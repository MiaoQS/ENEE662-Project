import os
import sys
import torch
import torch.nn as nn
import json
import numpy as np
sys.path.append('.')
from config import Config


def depth_to_disp(depth, min_depth, max_depth):
    scalar = 5.4
    min_disp=1/max_depth
    max_disp=1/min_depth
    scaled_disp = 1 / torch.clip(torch.clip(depth, 0, max_depth) / scalar, min_depth, max_depth)
    disp = (scaled_disp - min_disp) / (max_disp-min_disp)
    return disp

file_dir = os.path.dirname(os.path.realpath(__file__))
file_dir = os.path.dirname(file_dir)

    
def load_and_preprocess_intrinsics(intrinsics_path, resize_width, resize_height):
    K = np.eye(4)
    with open(intrinsics_path, 'r') as f:
        K[:3, :3] = np.array(json.load(f))

    # Convert normalised intrinsics to 1/4 size unnormalised intrinsics.
    # (The cost volume construction expects the intrinsics corresponding to 1/4 size images)
    K[0, :] *= resize_width // 4
    K[1, :] *= resize_height // 4

    invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
    K = torch.Tensor(K).unsqueeze(0)

    if torch.cuda.is_available():
        return K.cuda(), invK.cuda()
    return K, invK

class DepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(DepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        # print(disp.shape)
        return disp

class SQLdepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(SQLdepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        disp = nn.functional.interpolate(disp, input_image.shape[-2:], mode='bilinear', align_corners=True)
        # print(disp.shape)
        disp = depth_to_disp(disp, 0.1, 100)
        return disp

class PlaneDepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(PlaneDepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_color):
        grid = torch.meshgrid(torch.linspace(-1, 1, Config.input_W_PD), torch.linspace(-1, 1, Config.input_H_PD), indexing="xy")
        # grid = torch.meshgrid(torch.linspace(-1, 1, Config.input_W_PD), torch.linspace(-1, 1, Config.input_H_PD))
        # grid = [_.T for _ in grid]
        grid = torch.stack(grid, dim=0)
        grids = grid[None, ...].expand(input_color.shape[0], -1, -1, -1).cuda()
        output = self.decoder(self.encoder(input_color), grids)
        pred_disp = output["disp"]
        # pred_disp = output["disp"][:, 0]
        # print(pred_disp.shape)
        pred_disp = (pred_disp - 0.7424) / 741.6576
        return pred_disp


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def import_depth_model(scene_size, model_type='monodepth2'):
    """
    import different depth model to attack:
    possible choices: monodepth2
    """
    if scene_size == (320, 1024):
        if model_type == 'monodepth2':
            model_name = 'mono+stereo_1024x320'
            code_path = os.path.join(file_dir, 'DepthNetworks', 'monodepth2')
            depth_model_dir = os.path.join(code_path, 'models')
            sys.path.append(code_path)
            import networks
        elif model_type == 'depthhints':
            model_name = 'DH_MS_320_1024'
            code_path = os.path.join(file_dir, 'DepthNetworks', 'depth-hints')
            depth_model_dir = os.path.join(code_path, 'models')
            sys.path.append(code_path)
            import networks
        else:
            raise RuntimeError("depth model unfound")
    else:
        raise RuntimeError(f"scene size undefined! {scene_size}")
    model_path = os.path.join(depth_model_dir, model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    if model_type == 'monodepth2' or model_type == 'depthhints':
        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        encoder = networks.ResnetEncoder(18, False)
        
        # extract the height and width of image that this model was trained with
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)

        
        print("   Loading pretrained decoder")
        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        depth_decoder.load_state_dict(loaded_dict)

        depth_model = DepthModelWrapper(encoder, depth_decoder)
    return depth_model