#!/usr/bin/env python3

from utils.realsense_utils import Intrinsic640, get_point_cloud, display_point_cloud
import os
import cv2
import torch
import numpy as np
from prediction.models import *
from prediction.deep_fusion import DeepFusion
from prediction.classifier_free_guidance import *
from prediction.grip_force_model import GripForceMLP
from robot.robot_utils import *
from utils.transform import *
from utils.data_pipeline import normalize_gripper_pos, normalize_gripper_effort

from transformers import T5Tokenizer, BertTokenizer

import os
import shutil
from pathlib import Path


def save_img(image, img_name):
    # Create a new directory if it doesn't exist
    # Normalize float img to [0,255] and convert to uint8
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    uint8_image = normalized_image.astype(np.uint8)
    directory = f'{os.path.expanduser("~")}/debug_imgs'
    if not os.path.exists(directory):
        os.makedirs(directory)

    image_path = os.path.join(directory, f'{img_name}.png')
    cv2.imwrite(image_path, uint8_image)


def t2np(tensor):
    return tensor.detach().cpu().numpy()[0]


def t2float(tensor):
    return tensor.detach().cpu().numpy().item()


def create_file_index(folder):
    # find the index of the next file to be saved
    if len(os.listdir(folder)) == 0:
        file_index = 0
    else:
        # e.g. if folder is 'data/rgb/rgb_1_2_0.png', then the new index will be 0 + 1 = 1
        # file_index = max(int(name.split('_')[-1]) for name in os.listdir(folder)) + 1
        file_index = max(int(name.split('_')[-1].split('.')[0])
                         for name in os.listdir(folder)) + 1
    return file_index


def load_model(config, checkpoint_path=None):
    print('loading model...')
    print('USE_RGBD: ', config.USE_RGBD)
    # load model from config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.IMAGE_MODEL.split('-')[0] == 'vit':

        if hasattr(config, 'PRETRAINED'):
            image_model = RGBDViT(size=config.IMAGE_MODEL.split('-')[1],
                                  config=config,
                                  num_classes=0,
                                  pretrained=config.PRETRAINED).to(device)  # tiny, small, base, large
        else:
            image_model = RGBDViT(size=config.IMAGE_MODEL.split('-')[1],
                                  config=config,
                                  num_classes=0,
                                  ).to(device)

        if config.FREEZE_IMAGE_MODEL:
            for param in image_model.parameters():
                param.requires_grad = False
        if config.USE_RGBD:
            # train the patch embedding layer
            image_model.model.patch_embed.proj.requires_grad = True

    elif config.IMAGE_MODEL.split('-')[0] == 'clip':
        clip_model = RGBDCLIP(
            size=config.IMAGE_MODEL.split('-')[1],
            config=config,
        ).to(device)
        image_model, text_model = clip_model, clip_model.model.encode_text
        tokenizer = clip.tokenize

        if config.FREEZE_IMAGE_MODEL:
            for param in clip_model.visual.parameters():
                param.requires_grad = False
        if config.USE_RGBD:
            # train the patch embedding layer
            clip_model.model.visual.conv1.requires_grad = True

    elif config.IMAGE_MODEL.split('-')[0] == 'dinov2':
        image_model = RGBDDinov2(
            size=config.IMAGE_MODEL.split('-')[1],
            config=config,
        ).to(device)

        if config.FREEZE_IMAGE_MODEL:
            for param in image_model.parameters():
                param.requires_grad = False
        if config.USE_RGBD:
            # train the patch embedding layer
            image_model.model.patch_embed.proj.requires_grad = True
    else:
        print('Image model in config not recognized')

    # we don't need to load separate text models for this option
    if config.MULTIMODAL_HEAD != 'classifier-free-guidance':
        if config.TEXT_MODEL == 't5-small':
            # text_model = T5Model.from_pretrained("t5-small").encoder.to(device)
            text_model = T5(size='small').to(device)
            tokenizer = T5Tokenizer.from_pretrained("t5-small")
        elif config.TEXT_MODEL == 't5-base':
            # text_model = T5Model.from_pretrained("t5-base").encoder.to(device)
            text_model = T5(size='base').to(device)
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
        elif config.TEXT_MODEL == 't5-large':
            # text_model = T5Model.from_pretrained("t5-large").encoder.to(device)
            text_model = T5(size='large').to(device)
            tokenizer = T5Tokenizer.from_pretrained("t5-large")

        elif config.TEXT_MODEL == 'bert-base':
            text_model = Bert(size='base').to(device)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif config.TEXT_MODEL == 'bert-large':
            text_model = Bert(size='large').to(device)
            tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            print('text model: ', text_model)
        else:
            print('Text model in config not recognized')

        if config.IMAGE_MODEL.split('-')[0] == 'clip' and config.FREEZE_TEXT_MODEL:
            for param in clip_model.model.transformer.parameters():  # freeze the text model
                param.requires_grad = False
            for param in clip_model.model.ln_final.parameters():
                param.requires_grad = False
            for param in clip_model.model.token_embedding.parameters():
                param.requires_grad = False
        elif config.IMAGE_MODEL.split('-')[0] == 'clip' and not config.FREEZE_TEXT_MODEL:
            for param in clip_model.model.transformer.parameters():  # don't freeze the text model
                param.requires_grad = True
            for param in clip_model.model.ln_final.parameters():
                param.requires_grad = True
            for param in clip_model.model.token_embedding.parameters():
                param.requires_grad = True
        elif config.FREEZE_TEXT_MODEL:
            for param in text_model.parameters():
                param.requires_grad = False
        else:
            for param in text_model.parameters():
                param.requires_grad = True

    if config.MULTIMODAL_HEAD == 'concat-linear-attn-mlp':
        model = ConcatLinearAttnMLP(image_model, text_model)
    elif config.MULTIMODAL_HEAD == 'vision-only-linear':
        model = VisionOnlyLinear(image_model)
    elif config.MULTIMODAL_HEAD == 'vision-only-mlp':
        model = VisionOnlyMLP(image_model)
    elif config.MULTIMODAL_HEAD == 'vision-only-threeheads':
        model = ThreeHeadMLP(image_model)
    elif config.MULTIMODAL_HEAD == 'deep-fusion':
        model = DeepFusion(image_model, text_model)
    elif config.MULTIMODAL_HEAD == 'classifier-free-guidance' and config.IMAGE_MODEL.split('-')[0] == 'vit':
        # model = ClassifierFreeGuidance(image_model, text_model)
        # model = ConditionedVisionTransformer(image_model, text_model='t5') # text model can only be t5 or clip for now
        model = ConditionedVisionTransformer(
            image_model,
            text_model='t5',  # text model can only be t5 or clip for now
            config=config,
            hidden_dim=256,
        )
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
    elif config.MULTIMODAL_HEAD == 'classifier-free-guidance' and config.IMAGE_MODEL.split('-')[0] == 'clip':
        # modified cfg repo to use openai ViT-L/14 model
        model = ConditionedCLIP(
            image_model, text_model='clip', config=config, hidden_dim=256)
    elif config.MULTIMODAL_HEAD == 'classifier-free-guidance' and config.IMAGE_MODEL.split('-')[0] == 'dinov2':
        model = ConditionedDinov2(
            image_model, text_model='t5', config=config, hidden_dim=256)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    return model, tokenizer


class MovieWriter:
    def __init__(self, path, fps=30):
        self.writer = None
        self.path = path
        self.fps = fps

    def write_frame(self, frame):
        if self.writer is None:
            self.mkdir(self.path, cut_filename=True)
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(
                'M', 'J', 'P', 'G'), self.fps, (frame.shape[1], frame.shape[0]))
        self.writer.write(frame)

    def mkdir(self, path, cut_filename=False):
        if cut_filename:
            path = os.path.dirname(os.path.abspath(path))
        Path(path).mkdir(parents=True, exist_ok=True)

    def close(self):
        self.writer.release()


def load_grip_force_model():
    # to avoid multiprocessing issues since grip_force_model is tiny and running in the data loader
    device = torch.device('cpu')

    grip_force_model = GripForceMLP(num_inputs=3, num_outputs=1)
    grip_force_model.load_state_dict(
        torch.load(
            'grip_force_checkpoints/grip_force_dist_pos_effort_5_25_4/model_best.pth',
            map_location=device))
    grip_force_model.eval()
    return grip_force_model


def run_grip_force_model(grip_force_model, pos_dict, curr_left_fingertip, curr_right_fingertip):
    with torch.no_grad():
        curr_left_fingertip = torch.tensor(
            curr_left_fingertip).float().unsqueeze(0)
        curr_right_fingertip = torch.tensor(
            curr_right_fingertip).float().unsqueeze(0)
        gripper_pos = torch.tensor(
            pos_dict['gripper']).float().unsqueeze(0).unsqueeze(0)
        gripper_effort = torch.tensor(
            pos_dict['gripper_effort']).float().unsqueeze(0).unsqueeze(0)

        fingertip_dist = torch.norm(
            curr_left_fingertip - curr_right_fingertip, dim=1).float().unsqueeze(0)
        gripper_pos = normalize_gripper_pos(gripper_pos)
        gripper_effort = normalize_gripper_effort(gripper_effort)

        model_input = torch.cat((gripper_pos.unsqueeze(
            1), gripper_effort.unsqueeze(1), fingertip_dist.unsqueeze(1)), dim=1)
        # model_input = torch.cat((gripper_pos.unsqueeze(1), fingertip_dist.unsqueeze(1)), dim=1)
        # model_input = gripper_pos.unsqueeze(1)
        grip_force = grip_force_model(model_input)

        grip_force = grip_force.reshape((-1, 1))

        return grip_force


def copy_folder_contents(src_folder, dest_folder):
    # check if destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # if not, create it

    for filename in os.listdir(src_folder):
        file_path = os.path.join(src_folder, filename)

        if os.path.isfile(file_path):
            shutil.copy(file_path, dest_folder)  # copy files
        else:
            shutil.copytree(file_path, os.path.join(
                dest_folder, filename))  # copy directories
