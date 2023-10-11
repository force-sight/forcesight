#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
from prediction.loader import ActAffData
from tqdm import tqdm
from utils.config_utils import *
from utils.pred_utils import *
from utils.data_pipeline import *
import wandb
from prediction.grip_force_model import GripForceMLP

def train_epoch(model, optimizer, train_loader, criterion):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_sum = 0
    finger_dist_sum_old = 0
    finger_dist_sum = 0
    force_rmse_sum = 0
    correct_class = 0

    for prompt, initial_data, final_data, rgb_paths in tqdm(train_loader):
        # initial_data['rgb'] = initial_data['rgb'].to(device).float()
        # initial_data['depth'] = initial_data['depth'].to(device).float()
        initial_data['state'] = initial_data['state'].to(device)
        initial_data['left_fingertip'] = initial_data['left_fingertip'].to(device)
        initial_data['right_fingertip'] = initial_data['right_fingertip'].to(device)
        initial_data['ft'] = initial_data['ft'].to(device)

        # final_data['rgb'] = final_data['rgb'].to(device)
        # final_data['depth'] = final_data['depth'].to(device)
        # final_data['state'] = final_data['state'].to(device)
        # final_data['left_fingertip'] = final_data['left_fingertip'].to(device).float()
        # final_data['right_fingertip'] = final_data['right_fingertip'].to(device).float()
        # final_data['ft'] = final_data['ft'].to(device).float()

        # self.config.ROBOT_STATES= [x, y, z, roll, pitch, yaw, gripper]
        # initial state:  torch.Size([batch_size, 7])
        # print('initial state: ', initial_data['state'].shape)
        gripper_pos = initial_data['state'][:, config.ROBOT_STATES.index('gripper')].to(device).float()
        # print('gripper_pos: ', gripper_pos)
        gripper_pos = normalize_gripper_pos(gripper_pos)
        # print('normalized gripper_pos: ', gripper_pos)


        gripper_effort = initial_data['state'][:, config.ROBOT_STATES.index('gripper_effort')].to(device).float()
        # print('gripper_effort: ', gripper_effort)
        gripper_effort = normalize_gripper_effort(gripper_effort)   
        # print('normalized gripper_effort: ', gripper_effort)

        fingertip_dist = torch.norm(initial_data['left_fingertip'] - initial_data['right_fingertip'], dim=1).to(device).float()
        # print ('fingertip_dist: ', fingertip_dist)

        force_norm = torch.norm(initial_data['ft'], dim=1).unsqueeze(1).to(device).float()

        model_input = torch.cat((
            gripper_pos.unsqueeze(1),
            gripper_effort.unsqueeze(1),
            fingertip_dist.unsqueeze(1)), dim=1)
        # model_input = torch.cat((gripper_pos.unsqueeze(1), fingertip_dist.unsqueeze(1)), dim=1)
        # model_input = gripper_pos.unsqueeze(1)

        optimizer.zero_grad()

        output = model(model_input)
        loss = criterion(output, force_norm)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    return loss_sum / len(train_loader)

def val_epoch(model, val_loader, criterion):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_sum = 0
    finger_dist_sum_old = 0
    finger_dist_sum = 0
    force_rmse_sum = 0
    correct_class = 0

    with torch.no_grad():
        for prompt, initial_data, final_data, rgb_paths in tqdm(val_loader):
            # initial_data['rgb'] = initial_data['rgb'].to(device).float()
            # initial_data['depth'] = initial_data['depth'].to(device).float()
            initial_data['state'] = initial_data['state'].to(device)
            initial_data['left_fingertip'] = initial_data['left_fingertip'].to(device)
            initial_data['right_fingertip'] = initial_data['right_fingertip'].to(device)
            initial_data['ft'] = initial_data['ft'].to(device)

            # final_data['rgb'] = final_data['rgb'].to(device)
            # final_data['depth'] = final_data['depth'].to(device)
            # final_data['state'] = final_data['state'].to(device)
            # final_data['left_fingertip'] = final_data['left_fingertip'].to(device).float()
            # final_data['right_fingertip'] = final_data['right_fingertip'].to(device).float()
            # final_data['ft'] = final_data['ft'].to(device).float()

            # self.config.ROBOT_STATES= [x, y, z, roll, pitch, yaw, gripper]
            # initial state:  torch.Size([batch_size, 7])
            # print('initial state: ', initial_data['state'].shape)
            gripper_pos = initial_data['state'][:, config.ROBOT_STATES.index('gripper')].to(device).float()
            # print('gripper_pos: ', gripper_pos)
            gripper_pos = normalize_gripper_pos(gripper_pos)
            # print('normalized gripper_pos: ', gripper_pos)

            gripper_effort = initial_data['state'][:, config.ROBOT_STATES.index('gripper_effort')].to(device).float()
            # print('gripper_effort: ', gripper_effort)
            gripper_effort = normalize_gripper_effort(gripper_effort)   
            # print('normalized gripper_effort: ', gripper_effort)

            fingertip_dist = torch.norm(initial_data['left_fingertip'] - initial_data['right_fingertip'], dim=1).to(device).float()
            # print ('fingertip_dist: ', fingertip_dist)

            force_norm = torch.norm(initial_data['ft'][:3], dim=1).unsqueeze(1).to(device).float()

            model_input = torch.cat((gripper_pos.unsqueeze(1), gripper_effort.unsqueeze(1), fingertip_dist.unsqueeze(1)), dim=1)
            # model_input = torch.cat((gripper_pos.unsqueeze(1), fingertip_dist.unsqueeze(1)), dim=1)
            # model_input = gripper_pos.unsqueeze(1)

            # print('model_input: ', model_input.shape)

            output = model(model_input)
            loss = criterion(output, force_norm)

            loss_sum += loss.item()

    return loss_sum / len(val_loader)

def criterion(output, force_norm):
    # mse
    # print('output: ', output)
    # print('output shape: ', output.shape)
    # print('force_norm: ', force_norm)
    # print('force_norm shape: ', force_norm.shape)
    return torch.nn.functional.mse_loss(output, force_norm, reduction='mean')

if __name__ == '__main__':
    config, args = parse_config_args()
    wandb.init(project='action-affordances')

    wandb.config.update(config)
    wandb.config.update(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model, tokenizer = load_model(config)
    model = GripForceMLP(num_inputs=3, num_outputs=1)
    model.to(device)

    print('model: ', model)
    print('number of parameters: ', sum(p.numel() for p in model.parameters()))
    print('number of trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_dataset = ActAffData(config.TRAIN_FOLDER, stage='train')
    val_dataset = ActAffData(config.TEST_FOLDER, stage='test')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    # number of files in ./checkpoints that contain args.config
    folder_index = len([f for f in os.listdir(config.MODEL_DIR) if f.startswith(args.config)])

    # creating the checkpoint folder structure
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)

    if not os.path.exists(os.path.join(config.MODEL_DIR, '{}_{}'.format(args.config, folder_index))):
        os.makedirs(os.path.join(config.MODEL_DIR, '{}_{}'.format(args.config, folder_index)))

    wandb.run.name = '{}_{}'.format(args.config, folder_index)

    best_train_loss = float('inf')
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(model, optimizer, train_loader, criterion)
        val_loss = val_epoch(model, val_loader, criterion)

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)

        print(f'Epoch {epoch} - Train Loss: {train_loss} - Val Loss: {val_loss}')

        # model_name = '{}_{}/model_{}'.format(args.config, folder_index, epoch)
        model_name = '{}_{}/model'.format(args.config, folder_index, epoch)
        model_path = os.path.join(config.MODEL_DIR, model_name)

        if val_loss < best_val_loss:

            torch.save(model.state_dict(), model_path + '_best.pth')
            print('Model saved to {}'.format(model_path + '_best.pth'))
            best_val_loss = val_loss

        torch.save(model.state_dict(), model_path + '_latest.pth')
        print('Model saved to {}'.format(model_path + '_latest.pth'))
