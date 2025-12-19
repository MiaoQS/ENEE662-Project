import os
import sys

import options
import logging
from tensorboardX import SummaryWriter

import torch
from my_utils import set_random_seed, get_patch_area
from config import Config


def main():
    args = options.parse()
    set_random_seed(args.seed, deterministic=True)
    # prepare log
    log_dir =  os.path.join(Config.log_dir, args.test_name)
    logfile_path = os.path.join(log_dir, 'log.txt')

    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            # logging.FileHandler(logfile_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(str(args))
    tb_logger = SummaryWriter(log_dir)
    tb_logger.add_text("CLI_options", str(args), 0)

    model_name  = args.model_name
    attack_task = 'MDE'

    scene_size  = Config.model_scene_sizes_WH[model_name]
    patch_area = get_patch_area(attack_task, scene_size, args.patch_ratio)
    logging.info(f"patch_area: {patch_area}")

    tracker = None

    from attack.depth_model import import_depth_model
    model = import_depth_model(scene_size, model_name).to(Config.device).eval()


    # blackbox attack
    from attack.blackbox_patch import Blackbox_patch
    blackbox_patch = Blackbox_patch(model ,model_name, patch_area, args.n_batch,
                                args.batch_size, tb_logger, log_dir, tracker, args.targeted_attack)
    with torch.no_grad():
        if args.attack_method == 'ours':
            blackbox_patch.zoo(total_steps=args.n_iter, K=args.topk, num_pos=args.num_pos, num_init=args.init_iters, trail=args.trail, patch_only=args.patch_only)
        
        elif args.attack_method == 'GA_attack':
            blackbox_patch.GA_attack(num_generations = args.n_iter, num_parents_mating = 5, sol_per_pop = 20, patch_only=args.patch_only)
        else:
            raise RuntimeError(f'The attack method {args.attack_method} is not supported!')

if __name__ == "__main__":
    main()
    