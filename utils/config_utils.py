import argparse
import os
import yaml
import argparse
from types import SimpleNamespace

def load_config(config_name):
    config_path = os.path.join('./config', config_name + '.yml')

    with open(config_path, 'r') as stream:
        data = yaml.safe_load(stream)

    data_obj = SimpleNamespace(**data)
    data_obj.CONFIG_NAME = config_name
    return data_obj

def parse_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, default='default')
    parser.add_argument('--epoch', '-e', type=str, default='best', help='model epoch to load')
    parser.add_argument('--index', '-i', type=str, default='0', help='keeps track of training sessions using the same config')
    parser.add_argument('--folder', '-f', type=str, default=None, help='folder for data_capture or folder to pull data from if not live')
    parser.add_argument('--stage', '-s', type=str, default=None, help='train, test, or raw')
    parser.add_argument('--video_name', '-vname', type=str, default=False, help='video name')
    parser.add_argument('--speed', '-sp', type=int, default=1, help='general speed multiplier')
    parser.add_argument('--prompt', '-p', type=str, default='', help='text prompt for the model')
    parser.add_argument('--bipartite', '-bp', type=int, default=1, help='bipartite data capture')
    parser.add_argument('--save_all_frames', type=int, default=0, help='save non-keyframes')
    parser.add_argument('--ip', type=str, default='localhost', help='robot ip')
    parser.add_argument('--num_folders', type=int, default=0, help='number of folders to select for data experiments')
    parser.add_argument('--keypoints_per_folder', type=int, default=0, help='number of keypoints per folder for data experiments')

    # these are all technically boolean flags
    parser.add_argument('--record_video', '-rec', type=int, default=0, help='record video')
    parser.add_argument('--use_ft', type=int, default=1, help='use force torque sensor')
    parser.add_argument('--filter', type=int, default=0, help='delete bad data points')
    parser.add_argument('--view', '-v', type=int, default=1, help='view camera and graphs')
    parser.add_argument('--xbox', '-x', type=int, default=0, help='use xbox controller')
    parser.add_argument('--realsense_id', '-rs', type=str, default=None, help='realsense id')
    parser.add_argument('--live', '-lv', type=int, default=1, help='use camera feed instead of args.folder')
    
    # --flag that is default false
    parser.add_argument('--random_trans', action='store_true', help='apply data aug with random translation')
    parser.add_argument('--use_mock_ft', action='store_true', help='use mock ft sensor')
    parser.add_argument('--ignore_prefilter', action='store_true', help='ignore prefiltering')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--ros_viz', action='store_true', help='use ros viz')
    parser.add_argument('--evaluate', action='store_true', help='evaluate a trained model')
    parser.add_argument('--disable_auto_expose', action='store_true', help='disable realsense auto expose')
    parser.add_argument('--eval_folder', type=str, default=None, help='files to evaluate on when args.evaluate is True')
    parser.add_argument('--ablate_prompt', action='store_true', help='don\'t condition the model on text')
    parser.add_argument('--save_every_epoch', action='store_true', help='save a checkpoint every epoch')
    parser.add_argument('--binary_grip', action='store_true', help='binary gripper state')
    parser.add_argument('--ablate_force', action='store_true', help='dont use force goals')
    parser.add_argument('--ignore_robot', action='store_true', help='dont connect to the robot')

    args = parser.parse_args()
    return load_config(args.config), args
