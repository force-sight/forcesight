#!/usr/bin/env python3

import json
from aiohttp import web
from robot.kdl_wrapper import KdlWrapper, KdlType
from typing import Optional, Dict

##############################################################################

kdl_wrapper_cam = KdlWrapper.make(
        KdlType.FULL_ROBOT_6DOF,
        "robot/stretch_robot.urdf",
        "egocam_link",
    )

kdl_wrapper_gripper = KdlWrapper.make(
        KdlType.FULL_ROBOT_6DOF,
        "robot/stretch_robot.urdf",
        "link_grasp_center",
    )

##############################################################################

async def handle_fk_calc(request):
    try:
        data = await request.json()

        print("Calc FK, data", data)
        arm_seg = data[2]/4
        joints = [
                data[0], data[1],
                arm_seg, arm_seg, arm_seg, arm_seg,
                data[3], data[4], data[5],
            ]
        origin_pose = kdl_wrapper_cam.forward_kinematics(joints)
        response = {'res': origin_pose}
        return web.json_response(response)

    except json.JSONDecodeError:
        return web.Response(status=400, text='Bad Request: Invalid JSON data.')


async def handle_ik_calc(request):
    try:
        data = await request.json()

        print("Calc IK, data", data)
        target_joints = kdl_wrapper_gripper.inverse_kinematics(
            data, combine_arm_extension=True)
        response = {'res': target_joints}
        return web.json_response(response)

    except json.JSONDecodeError:
        return web.Response(status=400, text='Bad Request: Invalid JSON data.')


app = web.Application()
app.router.add_post('/ik_calc', handle_ik_calc)
app.router.add_post('/fk_calc', handle_fk_calc)


if __name__ == "__main__":
    web.run_app(app, host='localhost', port=8000)
