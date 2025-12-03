import os
import argparse
import torch
import numpy as np
from torchvision import transforms
from my_utils import set_random_seed, get_mean_depth_diff, EPE
from config import Config
from my_utils import get_patch_area
import matplotlib.pyplot as plt
from attack.dataset import KittiDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

def parse():
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model_name', type=str, choices=['monodepth2', 'planedepth', 'depthhints', 'SQLdepth', 'FlowNetC', 'PWC-Net','FlowNet2'], required=True, help='name of the subject model.')
    parser.add_argument('--attack_method', type=str, choices=['ours', 'GA_attack', 'S-RS', 'hardbeat', 'whitebox'], required=True, help='name of the attack method.')
    parser.add_argument('--patch_path', type=str, required=True, help='path of the adversarial patch.')
    parser.add_argument('--seed', type=int, default=1, help='Random Seed')
    parser.add_argument('--n_batch', type=int, default=50, help='number of ppictures for evaluation')
    # parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
    parser.add_argument('--patch_ratio', type=float, default=0.02, help='patch ratio')
    args = parser.parse_args()
    return args

def disp_viz(disp: torch.tensor, path, difference=False):
    disp = disp.detach().cpu().squeeze().numpy()
    if difference:
        plt.imshow(disp, cmap='magma')
        plt.axis('off')
        plt.savefig('eval/MDE/disp_difference_Scaled.png', bbox_inches='tight', pad_inches=0)
    vmax = np.percentile(disp, 95)
    plt.imshow(disp, cmap='magma', vmax=vmax, vmin=0)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    
    plt.close()

def flow_visualize(flow: torch.tensor, path, difference=False):
    flow = flow[0].permute(1,2,0).detach().cpu().numpy()
    flow = flow_viz.flow_to_image(flow, difference=difference)
    plt.imshow(flow)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def flow_visualize_v2(frame1: torch.tensor, flow_ori: torch.tensor, flow_attack: torch.tensor, path):
    frame1 = transforms.ToPILImage()(frame1.squeeze())
    flow_ori = flow_ori[0].permute(1,2,0).detach().cpu().numpy()
    flow_attack = flow_attack[0].permute(1,2,0).detach().cpu().numpy()
    flow_diff = flow_ori - flow_attack
    flow_ori = flow_viz.flow_to_image(flow_ori, difference=True)
    flow_attack = flow_viz.flow_to_image(flow_attack, difference=True)
    flow_diff = flow_viz.flow_to_image(flow_diff, difference=True)
    # plt.imshow(flow)
    # plt.axis('off')
    # plt.savefig(path, bbox_inches='tight', pad_inches=0)
    # plt.close()
    fig: Figure = plt.figure(figsize=(8, 4)) # width, height
    plt.subplot(221); plt.imshow(flow_ori); plt.title('original'); plt.axis('off')
    plt.subplot(222); plt.imshow(flow_attack); plt.title('attack'); plt.axis('off')
    plt.subplot(223); plt.imshow(flow_diff); plt.title('difference'); plt.axis('off')
    plt.subplot(224); plt.imshow(frame1); plt.title('frame1'); plt.axis('off')
    fig.canvas.draw()
    plt.savefig(path, pad_inches=0)
    plt.close()


def main(args):

    set_random_seed(args.seed, deterministic=True)
    model_name  = args.model_name
    os.mkdir(f"visualization/{args.model_name}")

    if model_name in ['FlowNetC','PWC-Net','FlowNet2']:
        attack_task = 'OF'
    elif model_name in ['depthhints','monodepth2']:
        attack_task = 'MDE'
    else:
        raise RuntimeError('The attack model is not supported!')
    scene_size  = Config.model_scene_sizes_WH[model_name]
    patch_area = get_patch_area(attack_task, scene_size, args.patch_ratio)

    eval_dataset = KittiDataset(model_name, main_dir=Config.kitti_dataset_root, mode='testing')
    assert args.batch_size == 1
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    p_t, p_l, p_h, p_w = patch_area
    patch_slice = (slice(0, args.batch_size), slice(0, 3), slice(p_t, p_t + p_h), slice(p_l, p_l + p_w))

    if attack_task == 'MDE':
        # import depth model
        from attack.depth_model import import_depth_model
        model = import_depth_model(scene_size, model_name).to(Config.device).eval()
        patch = torch.load(args.patch_path)
        patch = torch.from_numpy(patch).unsqueeze(0)

        for i, (scene, _) in enumerate(tqdm(eval_loader)):

            if i == args.n_images:
                break
            with torch.no_grad():

                scene_save = f"visualization/{args.model_name}/scene_{i}.png"
                save_image(scene, scene_save)

                disp_ref = model(scene.to(Config.device))
                disp_ref_save = f"visualization/{args.model_name}/depthref_{i}.png"
                disp_viz(disp_ref, disp_ref_save)

                scene[patch_slice] = patch
                patched_scene = f"visualization/{args.model_name}/patched_scene_{i}.png"
                save_image(scene, patched_scene)

                disp = model(scene.to(Config.device))
                disp_save = f"visualization/{args.model_name}/depth_{i}.png"
                disp_viz(disp, disp_save)


    elif attack_task == 'OF':
        raise NotImplementedError


if __name__ == "__main__":
    args = parse()
    main(args)

'''
# eval
CUDA_VISIBLE_DEVICES=1 python eval.py \
    --model_name monodepth2 \
    --patch_path patch.pt \
    --attack_method ours \
    --n_images 50
    --patch_ratio 0.02
'''
