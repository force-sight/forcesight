import torch
from torch.utils.data import DataLoader
from prediction.models import *
from prediction.loader import ActAffData
from tqdm import tqdm
from utils.config_utils import *
from utils.pred_utils import *
from utils.data_pipeline import *
import timm
from transformers import T5Model, T5Tokenizer, BertModel, BertTokenizer
import wandb
from torchvision.ops.focal_loss import sigmoid_focal_loss
import cv2
import numpy as np
import gc 

##############################################################################

def pred_metrics(config, raw_output, initial_data, final_data, device):
    """
    This is to calculate the metrics for the prediction
    """
    metrics_output = postprocess_output(config, raw_output, stage='metrics')
    timestep_gt = initial_data['timestep'].to(device).float()
    left_fingertip_gt, right_fingertip_gt = final_data['left_fingertip'], final_data['right_fingertip']

    if hasattr(config, 'PIXEL_SPACE_OUTPUT') and config.PIXEL_SPACE_OUTPUT:
        cls_img_gt, reg_img_gt = recover_pixel_space_represention(
            config, initial_data['cls_img'].detach().squeeze(1), initial_data['reg_img'].detach().squeeze(1))
        cls_img_pred, reg_img_pred = recover_pixel_space_represention(
            config, metrics_output['cls_img'].detach().squeeze(1), metrics_output['reg_img'].detach().squeeze(1))

        if hasattr(config, 'PIXEL_SPACE_CENTROID') and config.PIXEL_SPACE_CENTROID:
            centroid_pred = pixel_space_to_centroid(config, cls_img_pred, reg_img_pred, threshold=-0.001)

            _yaw = t2float(metrics_output['yaw'][0]) if hasattr(config, 'LAMBDA_YAW') and config.LAMBDA_YAW else 0.0
            left_fingertip_pred, right_fingertip_pred = centroid_to_fingertips(centroid_pred, t2float(metrics_output['width'][0]), _yaw)
            centroid_pred = torch.from_numpy(centroid_pred).to(device).float()
            centroid_dist = torch.norm(centroid_pred - final_data['centroid'][0])
            centroid_dist_x = torch.abs(centroid_pred[0] - final_data['centroid'][0, 0])
            centroid_dist_y = torch.abs(centroid_pred[1] - final_data['centroid'][0, 1])
            centroid_dist_z = torch.abs(centroid_pred[2] - final_data['centroid'][0, 2])

            left_fingertip_pred = torch.from_numpy(left_fingertip_pred).to(device).float()
            right_fingertip_pred = torch.from_numpy(right_fingertip_pred).to(device).float()

            # cam_z = sqrt(x^2 + y^2 + z^2)
            pred_cam_z_dist = torch.norm(centroid_pred)
            gt_cam_z_dist = torch.norm(final_data['centroid'][0])
        else:
            # left_fingertip_gt, right_fingertip_gt = pixel_space_to_contacts(config, cls_img_gt, reg_img_gt, method='local_max')
            left_fingertip_pred, right_fingertip_pred = pixel_space_to_contacts(config, cls_img_pred, reg_img_pred, method='local_max')
            
            if left_fingertip_gt is None or right_fingertip_gt is None or left_fingertip_pred is None or right_fingertip_pred is None:
                # print('Fingertip is None')
                # TODO: make this as far away as possible?? or just default to 0, 0, 0
                left_fingertip_pred = torch.from_numpy(np.array([0.0, 0.0, 0.0])).to(device).float()
                right_fingertip_pred = torch.from_numpy(np.array([0.0, 0.0, 0.0])).to(device).float()
            else:
                left_fingertip_pred = torch.from_numpy(left_fingertip_pred).to(device).float()
                right_fingertip_pred = torch.from_numpy(right_fingertip_pred).to(device).float()
   
            _diff_centroid = (left_fingertip_pred + right_fingertip_pred)/2 - (left_fingertip_gt[0] + right_fingertip_gt[0])/2
            centroid_dist = torch.norm((_diff_centroid))
            centroid_dist_x = torch.abs(_diff_centroid[0])
            centroid_dist_y = torch.abs(_diff_centroid[1])
            centroid_dist_z = torch.abs(_diff_centroid[2])

            # to calc the cam_z_dist_diff, first get the centroid
            pred_cam_z_dist = torch.norm((left_fingertip_pred + right_fingertip_pred)/2)
            gt_cam_z_dist = torch.norm((left_fingertip_gt[0] + right_fingertip_gt[0])/2)

        finger_dist = 0.5 * (torch.norm(left_fingertip_pred - left_fingertip_gt[0]) + torch.norm(right_fingertip_pred - right_fingertip_gt[0])) # only takes first element of the batch
        finger_dist_x = 0.5 * (torch.norm(left_fingertip_pred[0] - left_fingertip_gt[0, 0]) + torch.norm(right_fingertip_pred[0] - right_fingertip_gt[0, 0]))
        finger_dist_y = 0.5 * (torch.norm(left_fingertip_pred[1] - left_fingertip_gt[0, 1]) + torch.norm(right_fingertip_pred[1] - right_fingertip_gt[0, 1]))
        finger_dist_z = 0.5 * (torch.norm(left_fingertip_pred[2] - left_fingertip_gt[0, 2]) + torch.norm(right_fingertip_pred[2] - right_fingertip_gt[0, 2]))

        cam_z_dist_diff = torch.mean(torch.abs(pred_cam_z_dist - gt_cam_z_dist))
        pixel_cross_entropy = torch.nn.functional.cross_entropy(metrics_output['cls_img'], initial_data['cls_img'])
    else:
        left_fingertip_pred, right_fingertip_pred = metrics_output['left_fingertip'], metrics_output['right_fingertip']
        finger_dist = 0.5 * (torch.mean(torch.norm(left_fingertip_pred - left_fingertip_gt, dim=1)) + torch.mean(torch.norm(right_fingertip_gt - right_fingertip_pred, dim=1)))
        finger_dist_x = 0.5 * (torch.mean(torch.norm(left_fingertip_pred[:, 0] - left_fingertip_gt[:, 0])) + torch.mean(torch.norm(right_fingertip_gt[:, 0] - right_fingertip_pred[:, 0])))
        finger_dist_y = 0.5 * (torch.mean(torch.norm(left_fingertip_pred[:, 1] - left_fingertip_gt[:, 1])) + torch.mean(torch.norm(right_fingertip_gt[:, 1] - right_fingertip_pred[:, 1])))
        finger_dist_z = 0.5 * (torch.mean(torch.norm(left_fingertip_pred[:, 2] - left_fingertip_gt[:, 2])) + torch.mean(torch.norm(right_fingertip_gt[:, 2] - right_fingertip_pred[:, 2])))

        _diff_centroid = (left_fingertip_pred + right_fingertip_pred)/2 - (left_fingertip_gt + right_fingertip_gt)/2
        centroid_dist = torch.mean(torch.norm((_diff_centroid), dim=1))
        centroid_dist_x = torch.mean(torch.abs(_diff_centroid[:, 0]))
        centroid_dist_y = torch.mean(torch.abs(_diff_centroid[:, 1]))
        centroid_dist_z = torch.mean(torch.abs(_diff_centroid[:, 2]))

        pixel_cross_entropy = torch.tensor(0.4420, device=device)  # not using pixel cross entropy here

        # to calc the cam_z_dist_diff, first get the centroid
        pred_cam_z_dist = torch.norm((left_fingertip_pred + right_fingertip_pred)/2)
        gt_cam_z_dist = torch.norm((left_fingertip_gt[0] + right_fingertip_gt[0])/2)
        cam_z_dist_diff = torch.mean(torch.abs(pred_cam_z_dist - gt_cam_z_dist))

    force_rmse = torch.sqrt(torch.nn.functional.mse_loss(metrics_output['force'], final_data['ft'][:, :3]))

    metrics_output['grip_force'] = metrics_output['grip_force'].view((-1, 1))
    final_data['grip_force'] = final_data['grip_force'].view((-1, 1))
    gripforce_rmse = torch.sqrt(torch.nn.functional.mse_loss(metrics_output['grip_force'], final_data['grip_force']))

    correct_class = torch.sum(torch.argmax(metrics_output['timestep'], dim=1) == torch.argmax(timestep_gt, dim=1))
    # return the value instead of a tensor
    # print( finger_dist, finger_dist_x, finger_dist_y, finger_dist_z, force_rmse, correct_class, pixel_cross_entropy)
    
    metrics = [t2float(finger_dist), t2float(finger_dist_x), t2float(finger_dist_y), t2float(finger_dist_z),
           t2float(force_rmse), t2float(gripforce_rmse), t2float(correct_class), t2float(pixel_cross_entropy),
           t2float(cam_z_dist_diff), t2float(centroid_dist), t2float(centroid_dist_x), t2float(centroid_dist_y), t2float(centroid_dist_z)]

    if hasattr(config, 'LAMBDA_WIDTH') and config.LAMBDA_WIDTH:
        width_rmse = torch.sqrt(torch.nn.functional.mse_loss(metrics_output['width'], final_data['width']))
        metrics += [t2float(width_rmse)]
    if hasattr(config, 'LAMBDA_YAW') and config.LAMBDA_YAW:
        yaw_rmse = torch.sqrt(torch.nn.functional.mse_loss(metrics_output['yaw'], final_data['yaw']))
        metrics += [t2float(yaw_rmse)]
    return metrics

##############################################################################

def train_epoch(model, optimizer, train_loader, criterion, tokenizer):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_sum = 0
    finger_dist_sum = 0
    finger_dist_sum_x = 0
    finger_dist_sum_y = 0
    finger_dist_sum_z = 0
    centroid_dist_sum = 0
    centroid_dist_sum_x = 0
    centroid_dist_sum_y = 0
    centroid_dist_sum_z = 0
    force_rmse_sum = 0
    correct_class = 0
    pixel_cross_entropy = 0
    grip_force_rmse_sum = 0
    cam_z_dist_diff_sum = 0
    width_rmse_sum = 0
    yaw_rmse_sum = 0

    for prompt, initial_data, final_data, rgb_paths in tqdm(train_loader):
    
        initial_data['rgb'] = initial_data['rgb'].to(device).float()
        initial_data['depth'] = initial_data['depth'].to(device).float()
        # initial_data['state'] = initial_data['state'].to(device).float()
        # initial_data['left_fingertip'] = initial_data['left_fingertip'].to(device).float()
        # initial_data['right_fingertip'] = initial_data['right_fingertip'].to(device).float()
        # initial_data['ft'] = initial_data['ft'].to(device).float()
        # initial_data['grip_force'] = initial_data['grip_force'].to(device).float()
        if hasattr(config, 'PIXEL_SPACE_OUTPUT') and config.PIXEL_SPACE_OUTPUT:
            initial_data['cls_img'] = initial_data['cls_img'].to(device).float()
            initial_data['reg_img'] = initial_data['reg_img'].to(device).float()

        if hasattr(config, 'LAMBDA_WIDTH') and config.LAMBDA_WIDTH:
            final_data['width'] = final_data['width'].to(device).float()

        if hasattr(config, 'LAMBDA_YAW') and config.LAMBDA_YAW:
            final_data['yaw'] = final_data['yaw'].to(device).float()

        if hasattr(config, 'PIXEL_SPACE_CENTROID') and config.PIXEL_SPACE_CENTROID:
            final_data['centroid'] = final_data['centroid'].to(device).float()

        final_data['state'] = final_data['state'].to(device)
        final_data['left_fingertip'] = final_data['left_fingertip'].to(device).float()
        final_data['right_fingertip'] = final_data['right_fingertip'].to(device).float()
        final_data['ft'] = final_data['ft'].to(device).float()
        final_data['grip_force'] = final_data['grip_force'].to(device).float()
        timestep = initial_data['timestep'].to(device).float()

        optimizer.zero_grad()

        # Add padding to the text input
        prompt_input = preprocess_prompt(config, prompt, tokenizer)

        # stacking rgb and depth on the channel dimension
        rgbd_input = torch.cat((initial_data['rgb'], initial_data['depth']), dim=1)

        raw_output = model(rgbd_input, texts=prompt_input) # output is has keys ['left_fingertip'], ['right_fingertip'], ['force'], and ['grip_force']
        output = postprocess_output(config, raw_output)
        loss = criterion(output, initial_data, final_data, timestep)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        
        v = pred_metrics(config, raw_output, initial_data, final_data, device)
        finger_dist_sum += v[0]
        finger_dist_sum_x += v[1]
        finger_dist_sum_y += v[2]
        finger_dist_sum_z += v[3]
        force_rmse_sum += v[4]
        grip_force_rmse_sum += v[5]
        correct_class += v[6]
        pixel_cross_entropy += v[7]
        cam_z_dist_diff_sum += v[8]
        centroid_dist_sum += v[9]
        centroid_dist_sum_x += v[10]
        centroid_dist_sum_y += v[11]
        centroid_dist_sum_z += v[12]

        if hasattr(config, 'LAMBDA_WIDTH') and config.LAMBDA_WIDTH:
            width_rmse_sum += v[13]
        if hasattr(config, 'LAMBDA_YAW') and config.LAMBDA_YAW:
            yaw_rmse_sum += v[14]

        # make sure to delete the variables to free up memory
        del initial_data, final_data, prompt_input, rgbd_input, raw_output, output, loss, v
        gc.collect()

    size = len(train_loader)
    train_finger_dist_v2 = finger_dist_sum / size
    train_finger_dist_v2_x = finger_dist_sum_x / size
    train_finger_dist_v2_y = finger_dist_sum_y / size
    train_finger_dist_v2_z = finger_dist_sum_z / size

    train_centroid_dist = centroid_dist_sum / size
    train_centroid_dist_x = centroid_dist_sum_x / size
    train_centroid_dist_y = centroid_dist_sum_y / size
    train_centroid_dist_z = centroid_dist_sum_z / size

    wandb.log({'train_finger_dist_v2': train_finger_dist_v2}, step=epoch)
    wandb.log({'train_finger_dist_v2_x': train_finger_dist_v2_x}, step=epoch)
    wandb.log({'train_finger_dist_v2_y': train_finger_dist_v2_y}, step=epoch)
    wandb.log({'train_finger_dist_v2_z': train_finger_dist_v2_z}, step=epoch)

    wandb.log({'train_centroid_dist':   train_centroid_dist}, step=epoch)
    wandb.log({'train_centroid_dist_x': train_centroid_dist_x}, step=epoch)
    wandb.log({'train_centroid_dist_y': train_centroid_dist_y}, step=epoch)
    wandb.log({'train_centroid_dist_z': train_centroid_dist_z}, step=epoch)

    wandb.log({'train_force_rmse': force_rmse_sum / size}, step=epoch)
    wandb.log({'train_grip_force_rmse': grip_force_rmse_sum / size}, step=epoch)
    wandb.log({'train_timestep_acc': correct_class / (size*config.BATCH_SIZE)}, step=epoch)
    wandb.log({'train_pixel_cross_entropy': pixel_cross_entropy / size}, step=epoch)
    wandb.log({'train_cam_z_dist_diff': cam_z_dist_diff_sum / size}, step=epoch)

    print(f'train finger distance: {train_finger_dist_v2:.4f}')
    print(f'train finger distance xyz: {train_finger_dist_v2_x:.4f}, {train_finger_dist_v2_y:.4f}, {train_finger_dist_v2_z:.4f}')
    print(f'train centroid distance: {train_centroid_dist:.4f}')
    print(f'train centroid distance xyz: {train_centroid_dist_x:.4f}, {train_centroid_dist_y:.4f}, {train_centroid_dist_z:.4f}')
    print('train timestep classification accuracy: ', correct_class / (size*config.BATCH_SIZE))
    print('train force rmse: ', force_rmse_sum / size)
    print('train grip force rmse: ', grip_force_rmse_sum / size)
    print('train cam z dist diff: ', cam_z_dist_diff_sum / size)

    if hasattr(config, 'LAMBDA_WIDTH') and config.LAMBDA_WIDTH:
        print('train width rmse: ', width_rmse_sum / size)
        wandb.log({'train_width_rmse': width_rmse_sum / size}, step=epoch)

    if hasattr(config, 'LAMBDA_YAW') and config.LAMBDA_YAW:
        print('train yaw rmse: ', yaw_rmse_sum / size)
        wandb.log({'train_yaw_rmse': yaw_rmse_sum / size}, step=epoch)

    return loss_sum / size

##############################################################################

def val_epoch(model, val_loader, criterion, tokenizer):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_sum = 0
    finger_dist_sum = 0
    finger_dist_sum_x = 0
    finger_dist_sum_y = 0
    finger_dist_sum_z = 0
    centroid_dist_sum = 0
    centroid_dist_sum_x = 0
    centroid_dist_sum_y = 0
    centroid_dist_sum_z = 0
    force_rmse_sum = 0
    correct_class = 0
    pixel_cross_entropy = 0
    grip_force_rmse_sum = 0
    cam_z_dist_diff_sum = 0
    width_rmse_sum = 0
    yaw_rmse_sum = 0

    with torch.no_grad():
        for prompt, initial_data, final_data, rgb_paths in tqdm(val_loader):
            initial_data['rgb'] = initial_data['rgb'].to(device)
            initial_data['depth'] = initial_data['depth'].to(device)

            if hasattr(config, 'PIXEL_SPACE_OUTPUT') and config.PIXEL_SPACE_OUTPUT:
                initial_data['cls_img'] = initial_data['cls_img'].to(device).float()
                initial_data['reg_img'] = initial_data['reg_img'].to(device).float()

            if hasattr(config, 'LAMBDA_WIDTH') and config.LAMBDA_WIDTH:
                final_data['width'] = final_data['width'].to(device).float()

            if hasattr(config, 'LAMBDA_YAW') and config.LAMBDA_YAW:
                final_data['yaw'] = final_data['yaw'].to(device).float()

            if hasattr(config, 'PIXEL_SPACE_CENTROID') and config.PIXEL_SPACE_CENTROID:
                final_data['centroid'] = final_data['centroid'].to(device).float()

            # final_data['rgb'] = final_data['rgb'].to(device)
            # final_data['depth'] = final_data['depth'].to(device)
            final_data['state'] = final_data['state'].to(device)
            final_data['left_fingertip'] = final_data['left_fingertip'].to(device)
            final_data['right_fingertip'] = final_data['right_fingertip'].to(device)
            final_data['ft'] = final_data['ft'].to(device)
            final_data['grip_force'] = final_data['grip_force'].to(device).float()
            timestep = initial_data['timestep'].to(device)

            # Add padding to the text input
            prompt_input = preprocess_prompt(config, prompt, tokenizer)

            # stacking rgb and depth on the channel dimension
            rgbd_input = torch.cat((initial_data['rgb'], initial_data['depth']), dim=1).to(device)

            raw_output = model(rgbd_input, texts=prompt_input) # output is has keys ['left_fingertip'], ['right_fingertip'], ['force'], and ['grip_force']
            output = postprocess_output(config, raw_output)
            loss = criterion(output, initial_data, final_data, timestep)
            loss_sum += loss.item()

            v = pred_metrics(config, raw_output, initial_data, final_data, device)
            finger_dist_sum += v[0]
            finger_dist_sum_x += v[1]
            finger_dist_sum_y += v[2]
            finger_dist_sum_z += v[3]
            force_rmse_sum += v[4]
            grip_force_rmse_sum += v[5]
            correct_class += v[6]
            pixel_cross_entropy += v[7]
            cam_z_dist_diff_sum += v[8]
            centroid_dist_sum += v[9]
            centroid_dist_sum_x += v[10]
            centroid_dist_sum_y += v[11]
            centroid_dist_sum_z += v[12]

            if hasattr(config, 'LAMBDA_WIDTH') and config.LAMBDA_WIDTH:
                width_rmse_sum += v[13]
            if hasattr(config, 'LAMBDA_YAW') and config.LAMBDA_YAW:
                yaw_rmse_sum += v[14]

            # make sure to delete the variables to free up memory
            del initial_data, final_data, prompt_input, rgbd_input, raw_output, output, loss, v
            gc.collect()

        size = len(val_loader)
        val_finger_dist_v2 = finger_dist_sum / size
        val_finger_dist_v2_x = finger_dist_sum_x / size
        val_finger_dist_v2_y = finger_dist_sum_y / size
        val_finger_dist_v2_z = finger_dist_sum_z / size

        val_centroid_dist = centroid_dist_sum / size
        val_centroid_dist_x = centroid_dist_sum_x / size
        val_centroid_dist_y = centroid_dist_sum_y / size
        val_centroid_dist_z = centroid_dist_sum_z / size

        wandb.log({'val_finger_dist_v2': val_finger_dist_v2}, step=epoch)
        wandb.log({'val_finger_dist_v2_x': val_finger_dist_v2_x}, step=epoch)
        wandb.log({'val_finger_dist_v2_y': val_finger_dist_v2_y}, step=epoch)
        wandb.log({'val_finger_dist_v2_z': val_finger_dist_v2_z}, step=epoch)

        wandb.log({'val_centroid_dist': val_centroid_dist}, step=epoch)
        wandb.log({'val_centroid_dist_x': val_centroid_dist_x}, step=epoch)
        wandb.log({'val_centroid_dist_y': val_centroid_dist_y}, step=epoch)
        wandb.log({'val_centroid_dist_z': val_centroid_dist_z}, step=epoch)

        wandb.log({'val_force_rmse': force_rmse_sum / size}, step=epoch)
        wandb.log({'val_grip_force_rmse': grip_force_rmse_sum / size}, step=epoch)
        wandb.log({'val_timestep_acc': correct_class / (size*config.BATCH_SIZE)}, step=epoch)
        wandb.log({'val_pixel_cross_entropy': pixel_cross_entropy / size}, step=epoch)
        wandb.log({'val_cam_z_dist_diff': cam_z_dist_diff_sum / size}, step=epoch)


        print(f'val finger distance: {val_finger_dist_v2:.4f}')
        print(f'val finger distance xyz: {val_finger_dist_v2_x:.4f}, {val_finger_dist_v2_y:.4f}, {val_finger_dist_v2_z:.4f}')
        print(f'val centroid distance: {val_centroid_dist:.4f}')
        print(f'val centroid distance xyz: {val_centroid_dist_x:.4f}, {val_centroid_dist_y:.4f}, {val_centroid_dist_z:.4f}')
        print('val timestep classification accuracy: ', correct_class / (size*config.BATCH_SIZE))
        print('val force rmse: ', force_rmse_sum / size)
        print('val grip force rmse: ', grip_force_rmse_sum / size)
        print('val cam z dist diff: ', cam_z_dist_diff_sum / size)

        if hasattr(config, 'LAMBDA_WIDTH') and config.LAMBDA_WIDTH:
            val_width_rmse = width_rmse_sum / size
            wandb.log({'val_width_rmse': val_width_rmse}, step=epoch)
            print('val width rmse: ', val_width_rmse)

        if hasattr(config, 'LAMBDA_YAW') and config.LAMBDA_YAW:
            wandb.log({'val_yaw_rmse': yaw_rmse_sum / size}, step=epoch)
            print('val yaw rmse: ', yaw_rmse_sum / size)
    return loss_sum / size

##############################################################################

def criterion(output, initial_data, final_data, timestep):
    # # visualizing the output
    # cls_viewable = output['cls_img'].cpu().detach().numpy().transpose(0, 2, 3, 1)
    # reg_viewable = output['reg_img'].cpu().detach().numpy().transpose(0, 2, 3, 1)
    # cls_gt_viewable = initial_data['cls_img'].cpu().detach().numpy().transpose(0, 2, 3, 1)
    # reg_gt_viewable = initial_data['reg_img'].cpu().detach().numpy().transpose(0, 2, 3, 1)
    # rgb_viewable = initial_data['rgb'].cpu().detach().numpy().transpose(0, 2, 3, 1)

    # print('cls_viewable', cls_viewable.shape)
    # print('reg_viewable', reg_viewable.shape)
    # print('cls_gt_viewable', cls_gt_viewable.shape)
    # print('reg_gt_viewable', reg_gt_viewable.shape)
    # print('cls_viewable dtype', cls_viewable.dtype)
    # print('reg_viewable dtype', reg_viewable.dtype)
    # print('cls_viewable max', np.max(cls_viewable))
    # print('reg_viewable max', np.max(reg_viewable))
    # print('cls_viewable min', np.min(cls_viewable))
    # print('reg_viewable min', np.min(reg_viewable))
    
    # # viewing cls and depth images
    # cv2.imshow('cls', cls_viewable[0])
    # cv2.imshow('reg', reg_viewable[0])
    # cv2.imshow('cls_gt', cls_gt_viewable[0])
    # cv2.imshow('reg_gt', reg_gt_viewable[0])
    # cv2.imshow('rgb', rgb_viewable[0])
    # cv2.waitKey(0)

    if hasattr(config, 'PIXEL_SPACE_OUTPUT') and config.PIXEL_SPACE_OUTPUT:
        left_loss = 0
        right_loss = 0

        # applying weighted binary cross entropy on the cls_probs
        pixel_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.PIXEL_LABEL_WEIGHT]).to(device))(output['cls_img'], initial_data['cls_img'].float())
        
        # masking so that only the pixels that are not 0 are used in the loss
        depth_mask = initial_data['cls_img'] != 0 # (batch, 1, 224, 224) tensor that is 1 if the gt cls is not 0
        if not hasattr(config, 'DEPTH_LOSS') or config.DEPTH_LOSS == 'L1': # default to L1
            depth_loss = torch.nn.functional.l1_loss(output['reg_img'] * depth_mask, initial_data['reg_img'])
        elif config.DEPTH_LOSS == 'L2':
            depth_loss = torch.nn.functional.mse_loss(output['reg_img'] * depth_mask, initial_data['reg_img'])
    elif config.FINGERTIP_LOSS == 'L1':
        left_loss = torch.nn.functional.l1_loss(output['left_fingertip'], final_data['left_fingertip']) # left fingertip
        right_loss = torch.nn.functional.l1_loss(output['right_fingertip'], final_data['right_fingertip']) # right fingertip
    elif config.FINGERTIP_LOSS == 'L2':
        left_loss = torch.nn.functional.mse_loss(output['left_fingertip'], final_data['left_fingertip']) # left fingertip
        right_loss = torch.nn.functional.mse_loss(output['right_fingertip'], final_data['right_fingertip']) # right fingertip
    
    if config.FORCE_LOSS == 'L1':
        ft_loss = torch.nn.functional.l1_loss(output['force'], final_data['ft'][:, :3])
    elif config.FORCE_LOSS == 'L2':
        ft_loss = torch.nn.functional.mse_loss(output['force'], final_data['ft'][:, :3]) # forces only
    # print('ft loss: ', ft_loss)
    
    # if config.PITCH_LOSS == 'L1':
    #     pitch_loss = torch.nn.functional.l1_loss(output['pitch'], final_data['state'][:, 3])
    # elif config.PITCH_LOSS == 'L2':
    #     pitch_loss = torch.nn.functional.mse_loss(output['pitch'], final_data['state'][:, 3]) # x, y, z, pitch
    # print('pitch loss: ', pitch_loss)

    if not hasattr(config, 'GRIP_LOSS'):
        config.GRIP_LOSS = 'L2'
    if not hasattr(config, 'LAMBDA_GRIP'):
        config.LAMBDA_GRIP = 0.


    output['grip_force'] = output['grip_force'].view((-1, 1))
    final_data['grip_force'] = final_data['grip_force'].view((-1, 1))

    if config.GRIP_LOSS == 'L1':
        grip_loss = torch.nn.functional.l1_loss(output['grip_force'], final_data['grip_force'])
    elif config.GRIP_LOSS == 'L2':
        grip_loss = torch.nn.functional.mse_loss(output['grip_force'], final_data['grip_force'])
    # print('grip force:', output['grip_force'], final_data['grip_force'])

    # print('CLASSIFY TIMESTEPS')
    # print('timestep: ', timestep)
    # print('output timestep: ', output['timestep'])

    if hasattr(config, 'LAMBDA_WIDTH'):
        if not hasattr(config, 'WIDTH_LOSS'):
            config.WIDTH_LOSS = 'L2'
        if config.WIDTH_LOSS == 'L1':
            width_loss = torch.nn.functional.l1_loss(output['width'], final_data['width'])
        elif config.WIDTH_LOSS == 'L2':
            width_loss = torch.nn.functional.mse_loss(output['width'], final_data['width'])
    # print('width: ', output['width'], final_data['width'])

    if hasattr(config, 'LAMBDA_YAW'):
        if not hasattr(config, 'YAW_LOSS'):
            config.YAW_LOSS = 'L2'
        if config.YAW_LOSS == 'L1':
            yaw_loss = torch.nn.functional.l1_loss(output['yaw'], final_data['yaw'])
        elif config.WIDTH_LOSS == 'L2':
            yaw_loss = torch.nn.functional.mse_loss(output['yaw'], final_data['yaw'])

    if config.CLASSIFICATION_LOSS == 'cross_entropy':
        timestep_loss = torch.nn.functional.cross_entropy(output['timestep'], timestep)
    elif config.CLASSIFICATION_LOSS == 'focal':
        timestep_loss = sigmoid_focal_loss(output['timestep'], timestep, alpha=0.25, gamma=2, reduction='mean')

    # print('left loss: ', left_loss * config.LAMBDA_FINGERTIPS)
    # print('right loss: ', right_loss * config.LAMBDA_FINGERTIPS)
    # print('pixel loss: ', pixel_loss * config.LAMBDA_PIXEL)
    # print('depth loss: ', depth_loss * config.LAMBDA_DEPTH)
    # print('ft loss: ', ft_loss * config.LAMBDA_FORCE)
    # print('grip loss: ', grip_loss * config.LAMBDA_GRIP)
    # print('timestep loss: ', timestep_loss * config.LAMBDA_TIMESTEP)
    # print('width loss: ', width_loss * config.LAMBDA_WIDTH)

    # print('timestep loss: ', timestep_loss * config.LAMBDA_TIMESTEP)
    if hasattr(config, 'PIXEL_SPACE_OUTPUT') and config.PIXEL_SPACE_OUTPUT:
        loss = config.LAMBDA_FINGERTIPS * (left_loss + right_loss) \
            + config.LAMBDA_PIXEL * pixel_loss \
            + config.LAMBDA_DEPTH * depth_loss \
            + config.LAMBDA_FORCE * ft_loss \
            + config.LAMBDA_GRIP * grip_loss \
            + config.LAMBDA_TIMESTEP * timestep_loss
        if hasattr(config, 'LAMBDA_WIDTH'):
            loss += config.LAMBDA_WIDTH * width_loss
        if hasattr(config, 'LAMBDA_YAW'):
            loss += config.LAMBDA_YAW * yaw_loss
        return loss
    else:
        return config.LAMBDA_FINGERTIPS * (left_loss + right_loss) \
            + config.LAMBDA_FORCE * ft_loss \
            + config.LAMBDA_TIMESTEP * timestep_loss \
            + config.LAMBDA_GRIP * grip_loss
            # + config.LAMBDA_PITCH * pitch_loss \

##############################################################################

if __name__ == '__main__':
    config, args = parse_config_args()
    if args.num_folders or args.keypoints_per_folder:
        wandb.init(project='force-sight-dataset')
    else:
        wandb.init(project='force-sight-fixed_depth_6-6')

    wandb.config.update(config)
    wandb.config.update(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.pretrained_model_path:
        print('loading pretrained model from: ', args.pretrained_model_path)

    model, tokenizer = load_model(config, args.pretrained_model_path)

    # print('model: ', model)
    print('number of parameters: ', sum(p.numel() for p in model.parameters()))
    print('number of trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if config.LAMBDA_TIMESTEP > 0:
        print('CLASSIFYING TIMESTEP')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)    

    # number of files in ./checkpoints that contain args.config
    folder_index = len([f for f in os.listdir(config.MODEL_DIR) if f.startswith(args.config)])

    # creating the checkpoint folder structure
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)

    if not os.path.exists(os.path.join(config.MODEL_DIR, '{}_{}'.format(args.config, folder_index))):
        os.makedirs(os.path.join(config.MODEL_DIR, '{}_{}'.format(args.config, folder_index)))

    if args.num_folders or args.keypoints_per_folder:
        wandb.run.name = '{}_{}_{}_folders_{}_per_folder'.format(args.config, folder_index, args.num_folders, args.keypoints_per_folder)
    elif args.evaluate:
        wandb.run.name = 'evaluate_{}_{}'.format(args.config, folder_index)
    else:
        wandb.run.name = '{}_{}'.format(args.config, folder_index)

    best_train_loss = float('inf')
    best_val_loss = float('inf')

    if args.evaluate:
        epoch = 0
        val_dataset = ActAffData(args.eval_folder, stage='test')
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

        val_loss = val_epoch(model, val_loader, criterion, tokenizer)

    else:
        train_dataset = ActAffData(config.TRAIN_FOLDER, stage='train')
        val_dataset = ActAffData(config.TEST_FOLDER, stage='test')
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
        
        for epoch in range(config.NUM_EPOCHS):
            train_loss = train_epoch(model, optimizer, train_loader, criterion, tokenizer)
            val_loss = val_epoch(model, val_loader, criterion, tokenizer)

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

        if args.save_every_epoch:  
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, '{}_{}/model_{}.pth'.format(args.config, folder_index, epoch)))
