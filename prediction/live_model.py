import torch
from torch.utils.data import DataLoader
from prediction.models import *
from prediction.loader import ActAffData
from utils.config_utils import *
from utils.pred_utils import *
from utils.data_pipeline import *
from utils.visualizer import *
from robot.robot_utils import *
from recording.ft import FTCapture, MockFTCapture, EEF_PITCH_WEIGHT_OFFSET
from stretch_remote.robot_utils import keyboard_teleop, get_pos_dict, move_to_target
from stretch_remote.remote_client import RemoteClient
from utils.aruco_detect import ArucoPoseEstimator, find_contact_markers

import numpy as np
import cv2
import time
from utils.realsense_utils import RealSense, CameraType

DEBUG_MODE = False

##################################################################################

def print_yellow(msg):
    print('\033[93m' + msg + '\033[0m')

def pprint(msg):
    """pretty print in color"""
    if DEBUG_MODE:
        print('\033[92m' + msg + '\033[0m')           

class LiveModel():
    """Live model to run the vision model and control the robot"""

    def __init__(self, use_robot=True):
        self.config, self.args = parse_config_args()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stage = 'test'

        self.realsense = RealSense(
            select_device=self.args.realsense_id, view=self.args.view,
            auto_expose=(not self.args.disable_auto_expose))
        if self.args.use_mock_ft:
            self.ft = MockFTCapture()
        elif self.args.use_ft:
            self.ft = FTCapture()
        time.sleep(1)
        if use_robot:
            self.rc = RemoteClient(ip=self.args.ip, port=5556)
            if self.rc.get_status() is None:
                raise Exception('Remote client not connected')
        else:
            self.rc = None

        cam_mat, cam_dist = self.realsense.get_camera_intrinsics(CameraType.COLOR)
        self.aruco_pose_estimator = ArucoPoseEstimator(cam_mat, cam_dist)

        self.enable_moving = True
        self.stop = False

        self.prompt = self.args.prompt
        self.prompt_mod = self.prompt # modified prompt

        # init robot if robot is in used
        if use_robot:
            self.rc.move({'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'gripper': 0}) # leveling robot
            time.sleep(1)

        self.recording_video = False

        if self.args.use_ft:
            self.ft_offset = self.ft.get_ft()

        if self.args.record_video:
            if self.args.live:
                fps=15
                self.video = MovieWriter('videos/{}_{}.avi'.format(self.args.config, self.args.video_name + '_live'), fps=fps)

            else:
                # open args.folder/fps.txt and read fps
                with open(self.args.folder + '/fps.txt', 'r') as f:
                    fps = float(f.readline())

                self.video = MovieWriter('videos/{}_{}.avi'.format(self.args.config, self.args.video_name + '_rendered'), fps=fps)
                

            print("FPS: ", fps)

        
        model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(
            self.args.config, self.args.index, self.args.epoch))
        self.model, self.tokenizer = load_model(self.config, model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("planning model loaded")

        self.grip_force_model = load_grip_force_model()
        self.curr_left_fingertip = np.array([0,0,0.1])
        self.curr_right_fingertip = np.array([0,0,0.1])
        self.pred_left_fingertip = np.array([0,0,0.1])
        self.pred_right_fingertip = np.array([0,0,0.1])
        self.pred_grip_force = 0
        self.pred_timestep_index = 0
        print("grip force model loaded")

        if self.args.live:
            self.teleop_mode = True
        else:
            self.teleop_mode = False
        
        # for subgoal index testing, only used if SUBGOAL_TEXT is TRUE
        self.current_subgoal_idx = 0

        # for recording rgb and depth images
        self.num_frames = 0


    def run_model(self, control_func=None):
        """
        control_func:   The control function that takes in the current state
                        of the robot and outputs a control action. An example:
                        visual servoing approach
        """
        while not self.stop:
            # for prompt, initial_data, final_data, rgb_paths in tqdm(loader):
            # Add padding to the text input
            if control_func:
                pos_dict = get_pos_dict(self.rc)
                if pos_dict is not None:
                    self.pos_dict = pos_dict

            if self.teleop_mode:
                # for teleop mode, always use original prompt
                self.prompt_mod = self.prompt

            self.num_frames += 1

            if self.args.live:
                self.rgb_image, self.depth_image = self.realsense.get_rgbd_image()
            else:
                rgb_path = self.args.folder + '/rgb/{}.png'.format(self.num_frames)
                depth_path = self.args.folder + '/depth/{}.png'.format(self.num_frames)
                prompt_path = self.args.folder + '/prompt/{}.txt'.format(self.num_frames + 1)

                self.rgb_image = cv2.imread(rgb_path)
                self.depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                with open(prompt_path, 'r') as f:
                    self.prompt_mod = f.readline()

            # save rgb and depth images
            if self.recording_video:
                # self.num_frames += 1
                cv2.imwrite('{}/{}.png'.format(self.rgb_folder, self.num_frames), self.rgb_image) # save rgb image
                cv2.imwrite('{}/{}.png'.format(self.depth_folder, self.num_frames), self.depth_image) # save depth image
                # saving a text file with self.prompt_mod
                with open('{}/{}.txt'.format(self.prompt_folder, self.num_frames), 'w') as f:
                    f.write(self.prompt_mod)
                print_yellow(f"NOW RECORDING FRAME {self.num_frames}")

            prompt_input = preprocess_prompt(self.config, [self.prompt_mod], self.tokenizer)

            # stacking rgb and depth on the channel dimension
            rgb_input, depth_input = preprocess_rgbd(self.config, self.rgb_image, self.depth_image)
            rgbd_input = torch.cat((rgb_input, depth_input), dim=0)
            rgbd_input = rgbd_input.to(self.device)
            rgbd_input = rgbd_input.unsqueeze(0)

            # visualize_datapoint(self.prompt, initial_data, final_data, pred=output, save_folder=save_folder)

            if self.config.MULTIMODAL_HEAD == 'classifier-free-guidance':
                pred = self.model(rgbd_input, texts=prompt_input, cond_scale=1.) # output is has keys ['left_fingertip'], ['right_fingertip'], ['force'], and ['pitch']
            else:
                pred = self.model(rgbd_input, texts=prompt_input)
            pred = postprocess_output(self.config, pred, stage='metrics')

            # printout the predicted timestep
            timestep = t2np(pred['timestep'])
            self.pred_timestep_index = np.argmax(timestep)

            fingers_pos = self.aruco_pose_estimator.get_fingertip_poses(self.rgb_image)
            if fingers_pos is not None:
                self.curr_left_fingertip, self.curr_right_fingertip = find_contact_markers(*fingers_pos)

            if hasattr(self.config, 'PIXEL_SPACE_OUTPUT') and self.config.PIXEL_SPACE_OUTPUT:
                cls_img_pred, reg_img_pred = recover_pixel_space_represention(
                    self.config, pred['cls_img'], pred['reg_img'])

                if hasattr(self.config, 'PIXEL_SPACE_CENTROID') and self.config.PIXEL_SPACE_CENTROID:
                    # classic pixel space representation for centroid
                    centroid_pred = pixel_space_to_centroid(
                        self.config, cls_img_pred, reg_img_pred, method="local_max",
                        threshold=0.002, avg_depth_within_radius=2)
                    if centroid_pred is None:
                        self.pred_left_fingertip = None
                        self.pred_right_fingertip = None
                    else:
                        if hasattr(self.config, 'LAMBDA_YAW') and self.config.LAMBDA_YAW:
                            self.pred_left_fingertip, self.pred_right_fingertip = centroid_to_fingertips(centroid_pred, t2float(pred['width']), t2float(pred['yaw']))
                        else:
                            self.pred_left_fingertip, self.pred_right_fingertip = centroid_to_fingertips(centroid_pred, t2float(pred['width']))
                else:
                    # classic pixel space representation for fingertips
                    l_f, r_f = pixel_space_to_contacts(
                        self.config, cls_img_pred, reg_img_pred, method='local_max')
                    if l_f is not None and r_f is not None:
                        self.pred_left_fingertip = l_f
                        self.pred_right_fingertip = r_f
                    else:
                        print("Not detecting anything!!")
                        self.pred_left_fingertip = None
                        self.pred_right_fingertip = None
            else:
                self.pred_left_fingertip = t2np(pred['left_fingertip'])
                self.pred_right_fingertip = t2np(pred['right_fingertip'])

            if control_func is None:
                self.grip_force = 0.0
            else:
                # This runs the grip force model, a simple linear model that takes in current
                # robot state and outputs a grip force
                grip_force = run_grip_force_model(
                        self.grip_force_model,
                        self.pos_dict,
                        self.curr_left_fingertip,
                        self.curr_right_fingertip
                    )
                self.grip_force = t2float(grip_force)
            pprint(f'grip force: {self.grip_force}')

            # bgr to rgb
            # self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)

            if self.args.use_ft:
                self.curr_force = self.ft.get_ft() - self.ft_offset
                # this applies the offset
                self.curr_force -= EEF_PITCH_WEIGHT_OFFSET
            else:
                self.curr_force = np.zeros(6)
                
            self.pred_force = t2np(pred["force"])
            self.pred_force -= EEF_PITCH_WEIGHT_OFFSET[:3]

            pprint(f'current force: {self.curr_force}')
            pprint(f'pred final force: {self.pred_force}')
            
            ft_origin = (self.curr_left_fingertip + self.curr_right_fingertip) / 2

            self.pred_grip_force = t2np(pred['grip_force'])
            pprint(f'pred grip force: {self.pred_grip_force}')

            pred_fingertips = [self.pred_left_fingertip, self.pred_right_fingertip]

            is_teleop_text = '[Teleop] ' if self.teleop_mode else '[Auto] '
            raw_img = self.rgb_image.copy()

            if self.curr_left_fingertip is not None and self.curr_right_fingertip is not None:
                viewable_img = visualize_points(self.rgb_image,
                                                [self.curr_left_fingertip, self.curr_right_fingertip],
                                                [(0, 126, 0), (0, 126, 0)],
                                                show_depth=False,
                                                show_point=False)
                # viewable_img = visualize_forces(viewable_img, self.curr_force, ft_origin, color=(0, 126, 126))
                # viewable_img = visualize_grip_force(viewable_img, self.grip_force, [self.curr_left_fingertip, self.curr_right_fingertip], (0, 126, 0))

                if pred_fingertips[0] is not None and pred_fingertips[1] is not None:
                #     ft_pred_origin = (self.pred_left_fingertip + self.pred_right_fingertip) / 2
                #     viewable_img = visualize_points(viewable_img,
                #                                     pred_fingertips,
                #                                     colors=[(0, 255, 0), (0, 255, 0)],
                #                                     show_point=False)
                #     viewable_img = visualize_forces(viewable_img, self.pred_force, ft_pred_origin, color=(0, 255, 255))
                #     viewable_img = visualize_grip_force(viewable_img, self.pred_grip_force, pred_fingertips, color=(0, 255, 0))

                    viewable_img_noprompt = viewable_img.copy()
                    viewable_img_prompt = visualize_prompt(viewable_img, is_teleop_text + self.prompt_mod)
                    without_heatmap = viewable_img.copy()

                    if hasattr(self.config, 'PIXEL_SPACE_OUTPUT') and self.config.PIXEL_SPACE_OUTPUT:
                        # creating different visualizations to save
                        cls_img_pred_normalized = (cls_img_pred - np.min(cls_img_pred)) / 0.012 # (np.max(cls_img_pred) - np.min(cls_img_pred))
                        cls_img_pred_normalized_2 = (cls_img_pred - np.min(cls_img_pred)) / (np.max(cls_img_pred) - np.min(cls_img_pred))
                        
                        cls_img_pred_rgb = cv2.applyColorMap((cls_img_pred_normalized*255).astype(np.uint8), cv2.COLORMAP_JET)
                        cls_img_pred_rgb_2 = cv2.applyColorMap((cls_img_pred_normalized_2*255).astype(np.uint8), cv2.COLORMAP_JET)

                        reg_img_pred_rgb_normalized = (reg_img_pred - np.min(reg_img_pred)) / (np.max(reg_img_pred) - np.min(reg_img_pred))
                        
                        # blurring the heatmap
                        reg_img_pred_rgb_normalized = cv2.GaussianBlur(reg_img_pred_rgb_normalized, (33, 33), 0)
                        reg_img_pred_rgb = cv2.applyColorMap((reg_img_pred_rgb_normalized*255).astype(np.uint8), cv2.COLORMAP_JET)

                        # Create a 4-channel image with the same color data
                        cls_img_pred_rgba = cv2.cvtColor(cls_img_pred_rgb, cv2.COLOR_BGR2BGRA)
                        cls_img_pred_rgba_2 = cv2.cvtColor(cls_img_pred_rgb_2, cv2.COLOR_BGR2BGRA)

                        # Set the alpha channel to zero wherever the grayscale value is zero
                        # rgba_image[image == 0, 3] = 0
                        cls_img_pred_rgba[cls_img_pred_normalized < 5e-3, 3] = 0
                        cls_img_pred_rgba_2[cls_img_pred_normalized_2 < 5e-3, 3] = 0

                        reg_img_pred_rgba = cv2.cvtColor(reg_img_pred_rgb, cv2.COLOR_BGR2BGRA)
                        reg_img_pred_rgba[cls_img_pred_normalized < 5e-3, 3] = 0

                        cls_img_pred_translucent = cls_img_pred_rgba.copy()
                        cls_img_pred_translucent[cls_img_pred_normalized > 5e-3, 3] = 127 # make the heatmap translucent
                        # blurring the heatmap
                        cls_img_pred_translucent = cv2.GaussianBlur(cls_img_pred_translucent, (15, 15), 0)

                        viewable_img_prompt = alphaMerge(cls_img_pred_translucent, viewable_img_prompt, 0, 0) # using special function to overlay rgba images
                        viewable_img_noprompt = alphaMerge(cls_img_pred_translucent, viewable_img_noprompt, 0, 0) # using special function to overlay rgba images

                        viewable_depth = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
                else:
                    viewable_img_prompt = visualize_prompt(viewable_img, is_teleop_text + self.prompt_mod)
                    viewable_img_noprompt = viewable_img.copy()
                    without_heatmap = viewable_img.copy()

                # TODO: check this
                # viewable_img = visualize_points(viewable_img, [self.curr_left_fingertip, self.curr_right_fingertip], [(0, 126, 0), (0, 126, 0)])
                # viewable_img = visualize_forces(viewable_img, self.curr_force, ft_origin, color=(0, 126, 126))
                # viewable_img = visualize_grip_force(viewable_img, self.grip_force, [self.curr_left_fingertip, self.curr_right_fingertip], (0, 126, 0))

            if pred_fingertips[0] is not None and pred_fingertips[1] is not None:
                ft_pred_origin = (self.pred_left_fingertip + self.pred_right_fingertip) / 2
                viewable_img_noprompt = visualize_points(viewable_img_noprompt, pred_fingertips, colors=[(0, 255, 0), (0, 255, 0)] )
                if not self.args.ablate_force:
                    viewable_img_noprompt = visualize_grip_force(viewable_img_noprompt, self.pred_grip_force, pred_fingertips, color=(0, 255, 0))
                    viewable_img_noprompt = visualize_forces(viewable_img_noprompt, ft_pred_origin, self.pred_force, color=(0, 255, 255))
                viewable_depth = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)

            cv2.imshow('rgb', viewable_img_noprompt)
                
            # print('left error (m): ', np.linalg.norm(pred['left_fingertip'].cpu().numpy()[0] - final_left_fingertip.cpu().numpy()))
            # print('right error (m): ', np.linalg.norm(pred['right_fingertip'].cpu().numpy()[0] - final_right_fingertip.cpu().numpy()))
            # left_error = np.linalg.norm(pred['left_fingertip'].cpu().numpy()[0] - final_left_fingertip.cpu().numpy())
            # right_error = np.linalg.norm(pred['right_fingertip'].cpu().numpy()[0] - final_right_fingertip.cpu().numpy())

            keycode = cv2.waitKey(1) & 0xFF

            if self.args.record_video:
                self.video.write_frame(viewable_img_noprompt)
                
                
            ###################################################################################################
            # if keycode == ord('g'):
            #     move_to_target(pred_fingertips, self.rc, retries=2)

            ###################################################################################################
            # if keycode == ord('g'):
            #     move_to_target(pred_fingertips, self.rc, retries=2)
            ###################################################################################################
            if keycode == ord('t'):
                pprint(f'\nTeleop is now {self.teleop_mode}, switch to {not self.teleop_mode}\n')
                self.teleop_mode = not self.teleop_mode
                self.current_subgoal_idx = 0 # reset subgoal idx
            ###################################################################################################
            if keycode == ord('f'):
                self.args.ablate_force = not self.args.ablate_force
            if keycode == ord('g'):
                self.args.binary_grip = not self.args.binary_grip
            ###################################################################################################
            elif keycode == ord('.') and hasattr(self.config, 'PIXEL_SPACE_OUTPUT') and self.config.PIXEL_SPACE_OUTPUT:
                datetime = time.strftime("%Y%m%d-%H%M%S")
                fig_dir = './figures/' + datetime
                if not os.path.exists('./figures'):
                    os.makedirs('./figures')
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)

                # saving the viewable image, original rgb image, cls_img_pred_rgb, and cls_img_pred_rgb_2
                cv2.imwrite(f'{fig_dir}/rgb_{datetime}.png', viewable_img_prompt)
                cv2.imwrite(f'{fig_dir}/rgb_noprompt_{datetime}.png', viewable_img_noprompt)
                cv2.imwrite(f'{fig_dir}/rgb_pred_{datetime}.png', without_heatmap)
                cv2.imwrite(f'{fig_dir}/cls_img_pred_rgb_{datetime}.png', cls_img_pred_rgba)
                cv2.imwrite(f'{fig_dir}/cls_img_pred_rgb_2_{datetime}.png', cls_img_pred_rgba_2)
                cv2.imwrite(f'{fig_dir}/reg_img_pred_rgb_{datetime}.png', reg_img_pred_rgb)
                cv2.imwrite(f'{fig_dir}/reg_img_pred_rgb_2_{datetime}.png', reg_img_pred_rgba)
                cv2.imwrite(f'{fig_dir}/raw_img_{datetime}.png', raw_img)
                cv2.imwrite(f'{fig_dir}/depth_img_{datetime}.png', viewable_depth)
                print(f'Images saved to ./figures/rgb/rgb_{datetime}.png, ./figures/rgb_pred/rgb_pred_{datetime}.png, \
                      ./figures/cls_img_pred_rgb/cls_img_pred_rgb_{datetime}.png, \
                        ./figures/cls_img_pred_rgb_2/cls_img_pred_rgb_2_{datetime}.png, \
                            ./figures/raw_img/raw_img_{datetime}.png', f'./figures/depth_img/depth_img_{datetime}.png')
            ###################################################################################################
            elif keycode == ord('v'):
                self.recording_video = not self.recording_video

                if self.recording_video:
                    datetime = time.strftime("%Y%m%d-%H%M%S")
                    self.results_folder = 'results/{}_{}_ablateforce_{}_binarygrip_{}'.format(self.args.config, datetime, self.args.ablate_force, self.args.binary_grip)
                    # self.rgb_folder = 'results/{}_{}/rgb'.format(self.args.config, datetime)
                    # self.depth_folder = 'results/{}_{}/depth'.format(self.args.config, datetime)
                    self.rgb_folder = '{}/rgb'.format(self.results_folder)
                    self.depth_folder = '{}/depth'.format(self.results_folder)
                    self.prompt_folder = '{}/prompt'.format(self.results_folder)

                    if not os.path.exists(self.results_folder):
                        os.makedirs(self.results_folder)
                        os.makedirs(self.rgb_folder)
                        os.makedirs(self.depth_folder)
                        os.makedirs(self.prompt_folder)

                    self.num_frames = 0
                    self.start_time = time.time()
                    print_yellow(f'Saving video to results/{self.args.config}_{datetime}')

                if self.num_frames > 0 and not self.recording_video:
                    print('Average FPS: ', self.num_frames / (time.time() - self.start_time))
                    # saving FPS to a text file for videos
                    fps = self.num_frames / (time.time() - self.start_time)
                    with open('{}/fps.txt'.format(self.results_folder), 'w') as f:
                        f.write(str(fps))
                    self.num_frames = 0
                    print_yellow(f'Saved video to {self.results_folder}')

            ###################################################################################################
            override_next_subgoal = False
            if keycode == ord('y'):
                override_next_subgoal = True

            self.control_action = None
            if control_func is not None:
                self.control_action, next_subgoal = control_func()

            if self.teleop_mode or control_func is None:
                keyboard_teleop(self.rc, self.config.ACTION_DELTA_DICT, keycode, self)
            else:
                if self.control_action is not None:
                    if self.args.live:
                        self.rc.move(self.control_action)

                    if hasattr(self.config, 'SUBGOAL_TEXT') and self.config.SUBGOAL_TEXT:
                        if next_subgoal or override_next_subgoal:
                            self.current_subgoal_idx += 1
                        if self.config.SUBGOAL_TEXT == 'named_action':
                            self.prompt_mod = self.prompt + timestep_to_subgoal_text(self.current_subgoal_idx, self.prompt)
                        else:
                            self.prompt_mod = self.prompt + timestep_to_subgoal_text(self.current_subgoal_idx)
                        print(f"Now to {self.prompt_mod}")

            # print('left error (m): ', np.linalg.norm(pred['left_fingertip'].cpu().numpy()[0] - final_left_fingertip.cpu().numpy()))
            # print('right error (m): ', np.linalg.norm(pred['right_fingertip'].cpu().numpy()[0] - final_right_fingertip.cpu().numpy()))
            # left_error = np.linalg.norm(pred['left_fingertip'].cpu().numpy()[0] - final_left_fingertip.cpu().numpy())
            # right_error = np.linalg.norm(pred['right_fingertip'].cpu().numpy()[0] - final_right_fingertip.cpu().numpy())
            # cv2.waitKey(1)

            # rgb_pred = visualize_points(self.rgb_image, pred_fingertips)

        if self.args.record_video:
            self.video.close()



##################################################################################

if __name__ == '__main__':
    live_model = LiveModel(use_robot=False)
    live_model.run_model(control_func=None)
