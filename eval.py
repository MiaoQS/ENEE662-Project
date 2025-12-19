import argparse
import torch
import numpy as np
# from torchvision.utils import save_image
from torchvision import transforms
from my_utils import set_random_seed, get_mean_depth_diff, EPE
from config import Config
from my_utils import get_patch_area
import matplotlib.pyplot as plt
from attack.dataset import KittiDataset
from torch.utils.data.dataloader import DataLoader


def parse():
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model_name', type=str, choices=['monodepth2', 'depthhints'], required=True, help='name of the subject model.')
    parser.add_argument('--attack_method', type=str, choices=['ours', 'GA_attack'], required=True, help='name of the attack method.')
    parser.add_argument('--patch_path', type=str, required=True, help='path of the adversarial patch.')
    parser.add_argument('--seed', type=int, default=1, help='Random Seed')
    parser.add_argument('--n_batch', type=int, default=40, help='number of ppictures for evaluation')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size for evaluation')
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


def main(args):
    set_random_seed(args.seed, deterministic=True)

    model_name  = args.model_name
    attack_task = 'MDE'
    
    scene_size  = Config.model_scene_sizes_WH[model_name]
    patch_area = get_patch_area(attack_task, scene_size, args.patch_ratio)

    eval_dataset = KittiDataset(model_name, main_dir=Config.kitti_dataset_root, mode='testing')
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    p_t, p_l, p_h, p_w = patch_area
    patch_slice = (slice(0, args.batch_size), slice(0, 3), slice(p_t, p_t + p_h), slice(p_l, p_l + p_w))

    error_patch_list = []
    error_whole_list = []

    
    # import depth model
    from attack.depth_model import import_depth_model
    model = import_depth_model(scene_size, model_name).to(Config.device).eval()
    patch = torch.load(args.patch_path)
    patch = torch.from_numpy(patch).unsqueeze(0)

    for i, (scene, _) in enumerate(eval_loader):
        print(i)
        if i == args.n_batch:
            break
        with torch.no_grad():
            disp_ref = model(scene.to(Config.device))
            # disp_viz(disp_ref, 'eval/MDE/original_disparity.png')

            scene[patch_slice] = patch
            # save_image(scene, 'eval/MDE/scene_patched.png')
            disp = model(scene.to(Config.device))
            
            patch_mask = torch.zeros_like(disp)
            patch_mask[patch_slice] = 1
            mean_depth_error_patch = get_mean_depth_diff(disp, disp_ref, patch_mask)
            mean_depth_error_whole = get_mean_depth_diff(disp, disp_ref)
            error_patch_list.append(mean_depth_error_patch)
            error_whole_list.append(mean_depth_error_whole)

    error_in_patch = torch.mean(torch.stack(error_patch_list)).item()
    error_in_scene = torch.mean(torch.stack(error_whole_list)).item()

    print(f'{args.attack_method}, attack model: {model_name}')
    print(f'{args.attack_method} mean depth error in patch area: {error_in_patch}')
    print(f'{args.attack_method} mean depth error in whole scene: {error_in_scene}')


if __name__ == "__main__":
    args = parse()
    main(args)

