#!/usr/bin/env python3

import json
import requests

##############################################################################

def get_forward_kinematics(joint_list, url='http://localhost:8000/fk_calc'):
    """Return an optional list of target pose for the camera egocam_link"""
    if len(joint_list) != 6:
        print("Error! Expect 6 joints")
        return None

    headers = {'Content-Type': 'application/json'}
    l = []
    for j in joint_list:
        l.append(float(j))
    data = json.dumps(l)
    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        result = response.json()
        return result['res']
    else:
        print('Error:', response.status_code)
        return None

def get_inverse_kinematics(joint_list, url='http://localhost:8000/ik_calc'):
    """
    Return an optional list of joint angles to reach the link_grasp_center
    """
    if len(joint_list) != 6:
        print("Error! Expect 6 values, xyzrpy")
        return None

    headers = {'Content-Type': 'application/json'}
    l = []
    for j in joint_list:
        l.append(float(j))
    data = json.dumps(l)
    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        result = response.json()
        if result is not None:
            return result['res']
        else:
            return None
    else:
        print('Error:', response.status_code)
        return None

##############################################################################
if __name__ == "__main__":
    r = get_forward_kinematics([0]*6)
    print("r", r)
    r = get_inverse_kinematics(r)
    print("r", r)
    r = get_inverse_kinematics([1.0]*6)
    print("r", r) # this will fail
