import numpy as np
import time
from recording.ft import FTCapture
import math

def calibrate_ft(ft_obj):
    ft = ft_obj.get_ft()
    np.save('ft_calibration.npy', ft)


def get_ft_calibration():
    return np.load('ft_calibration.npy')


def ft_to_cam_rotation(custom_pitch=(10/90)*math.pi/2):
    """
    custom_pitch: camera pitch angle, down is -ve and up is +ve
    """
    custom_pitch = np.array([[1, 0, 0],
                [0, np.cos(custom_pitch), -np.sin(custom_pitch)],
                [0, np.sin(custom_pitch), np.cos(custom_pitch)]])
    return np.array([[0, 1, 0],
                     [-1, 0, 0],
                     [0, 0, 1]])@custom_pitch

##############################################################################

if __name__ == '__main__':
    robot = None
    ft_obj = FTCapture()
    calibrate_ft(ft_obj)
    offset = get_ft_calibration()
    frame_rotation = ft_to_cam_rotation()

    while True:
        ft = ft_obj.get_ft()
        ft = ft - offset
        # print('calibrated FT: ', ft)
        # rounding to 3 decimal places
        print('calibrated FT: ', np.round(ft, 3)[:3])
        print('calibrated FT in camera frame: ', np.round(ft[:3] @ frame_rotation, 3))
        time.sleep(0.4)
