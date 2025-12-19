import torch

class Config(object):
    device = torch.device("cuda")
    kitti_dataset_root = "/path/to/dataset/KITTI/flow/"
    log_dir = "/path/to/logs"
    ori_H = 375
    ori_W = 1242
    input_W = 1024
    input_H = 320
    input_W_PD = 1280
    input_H_PD = 384
    train_scenes = 20
    input_H_GoogleAPI = 256
    input_W_GoogleAPI = 192
    model_scene_sizes_WH = {
        'monodepth2': (input_H, input_W),
        'depthhints': (input_H, input_W),
    }
    blacklight_shape = (20, 64)
    threshold_betwSquare = {
        'monodepth2': 1, # 10 default
        'depthhints': 1,

    }
    threshold_inSquare = {
        'monodepth2': 1, 
        'depthhints': 1,
    }
    eps = 1e-10
    init_noise_weight = 0.1
    min_noise_weight = 0.03
    lr = 0.1 # 0.1 default
    white_lr = 0.01
    beta1 = 0.5 # 0.5 default for optimizor
    beta2 = 0.5 # 0.5 default for optimizor
    gap = 1 # gap for calculate current loss
    AdaptiveTrail = False
    
    fixed_Noiseweight = False
    topk = False # False: One Way, True: Best K. Note: One way is better than top 1. this parameter is for hardbeat
    minus_mean = False # should always be set to False
    noise_type = 'discrete' # better than continues noise
    prob_norm_times = False
    AdaptiveWeight = 'V1' # V1 is better than V2
    Weight_Normalization = True # True is better than false

    UseAdam = True
    Oneway = True # true is better than false
    hardbeat_oneway = True

    # parameters of countermeasure Blacklight
    window_size = 20
    hash_kept = 50
    roundto = 50
    step_size = 1
    workers = 5
    tracker_threshold = 25
    disturbance_weight = 0.05
    benign_rate = False
