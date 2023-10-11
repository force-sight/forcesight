import torch
from torch.utils.data import DataLoader
from prediction.models import *
from prediction.loader import ActAffData
from tqdm import tqdm
from utils.config_utils import *
from utils.pred_utils import *
from utils.data_pipeline import *
from utils.visualizer import *
from utils.realsense_utils import PCDViewer
import timm
from transformers import T5Model, T5Tokenizer, BertModel, BertTokenizer
import numpy as np
import cv2

def view_preds(folder, stage='test'):
    config, args = parse_config_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ActAffData(folder=folder, stage=stage)
    # loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch))
    model, tokenizer = load_model(config, model_path)
    model = model.to(device)
    model.eval()

    save_folder = os.path.join(os.getcwd(), 'results/{}_{}_{}/'.format(args.config, args.index, args.epoch))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # pcd_view = PCDViewer(blocking=True)
    pcd_view = None
    if args.ros_viz:
        from ros_scripts.ros_viz import RosVizInterface
        pcd_view = RosVizInterface()

    with torch.no_grad():
        for prompt, initial_data, final_data, rgb_paths in tqdm(loader):
            prompt = prompt
            initial_data['rgb'] = initial_data['rgb'].to(device)
            initial_data['depth'] = initial_data['depth'].to(device)

            final_data['state'] = final_data['state'].to(device)
            final_data['left_fingertip'] = final_data['left_fingertip'].to(device)
            final_data['right_fingertip'] = final_data['right_fingertip'].to(device)
            final_data['ft'] = final_data['ft'].to(device)

            # Add padding to the text input
            prompt_input = preprocess_prompt(config, prompt, tokenizer)

            # stacking rgb and depth on the channel dimension
            rgbd_input = torch.cat((initial_data['rgb'], initial_data['depth']), dim=1)

            output = model(rgbd_input, texts=prompt_input) #, cond_scale=1.) # output is has keys ['left_fingertip'], ['right_fingertip'], ['force'], and ['pitch']
            output = postprocess_output(config, output, stage='metrics')
            visualize_datapoint(prompt, initial_data, final_data,
                                config=config, pred=output, save_folder=save_folder,
                                viz_3d=pcd_view)

if __name__ == '__main__':
    config, args = parse_config_args()
    view_preds(folder=args.folder, stage=args.stage)
