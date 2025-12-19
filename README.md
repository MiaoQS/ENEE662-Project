# ENEE662 Project -- On Evaluating the Robustness of Monocular Depth Estimation Models: Zeroth-Order Adversarial Patch Optimization

Group Memebers: Yangfan Deng, Tianyang Chen, Zhaoyi Liu.

This is the official PyTorch implementation of our Project "On Evaluating the Robustness of Monocular Depth Estimation Models: Zeroth-Order Adversarial Patch Optimization".

## Table of Contents

1. [Prepare for the Code](#environment-preparation)
2. [Construct the Environment](#code-preparation)
3. [Set up the Dataset](#dataset-preparation)
5. [Prepare for the Configuration](#configuration-preparation)
6. [Launch black-box attacks](#launch-black-box-attacks)
7. [Evaluate the patch Performance](#evaluate-the-patch)
8. [Visualize the Attacking Performances](#attack-the-google-online-service)

## Code preparation

Clone this repository

```
cd ~
git clone https://github.com/MiaoQS/ENEE662-Project ENEE662-Project
cd ENEE662-Project
```

Prepare the target MDE networks ([Monodepth2](https://github.com/nianticlabs/monodepth2), [DepthHints](https://github.com/nianticlabs/depth-hints) following their official instructions and put them in the directory of `DepthNetworks`. Download their official pretrained model weights (with the highest input resolution and best performance) into a sub-folder named `models` inside each network's directory (e.g., `DepthNetworks/monodepth2/models`).

The directories should be organized as:
```
BadPart
├── DepthNetworks
    ├── depth-hints
    ├── monodepth2
```

## Environment preparation

Create a new conda environment using the environment.yml file:
```
conda env create -f environment.yml
conda activate BadPart
```



## Dataset preparation
You will need to download the [KITTI flow dataset](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) and organize it in the following way. Assume the path of the dataset is `/path/to/dataset/KITTI/flow/`.

```
KITTI
├── flow
    ├── testing
    ├── training
    ├── devkit
```


## Configuration preparation
Provide the log path and the dataset path in the file `config.py`:
```
kitti_dataset_root = "/path/to/dataset/KITTI/flow/"
log_dir = "/path/to/logs"
```

## Launch optimization

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name monodepth2 \
    --attack_method ours \
    --patch_ratio 0.02 \
    --batch_size 4 \
    --n_batch 1 \
    --n_iter 10001 \
    --trail 20 \
    --targeted_attack \
    --patch_only \
    --test_name monodepth2
```

You can change the target model with the option `--model_name` and available models are:
- MDE models: 
    - Monodepth2    -> `monodepth2`
    - DepthHints    -> `depthhints`

You can also change the attack method with the option `--attack_method` and available methods are:
- Ours   -> `ours`
- GenAttack -> `GA_attack`

For detailed explanations of each options, please refer to the file `options.py`
The generated patch file is saved to folder `/path/to/logs/name_for_this_test/` by default.
 
## Evaluate the patch

You can evaluate the attack performance of the generated universal adversarial patch by runnning `eval.py`:
```
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_name monodepth2 \
    --patch_path /path/to/your/patch \
    --patch_ratio 0.02 \
    --batch_size 5 \
```
Prepare your adversarial patch and replace `/path/to/your/patch` with the actual path to your patch.

You can still change the target model with the option `--model_name`, but this target should be the same as the target you used to generate the patch.

## Visualize the Attacking Performances

You can visualize the attack performance of the generated universal adversarial patch by runnning `visualization.py`:
```
CUDA_VISIBLE_DEVICES=0 python visualization.py \
    --model_name monodepth2 \
    --attack_method ours \
    --patch_path /path/to/your/patch \
    --patch_ratio 0.02 \
    --batch_size 5 \
```
Prepare your adversarial patch and replace `/path/to/your/patch` with the actual path to your patch.

You can still change the target model with the option `--model_name`, but this target should be the same as the target you used to generate the patch.

## Acknowledgements

The authors acknowledge the University of Maryland supercomputing resources (https://hpcc.umd.edu) made available for conducting the research reported in this paper.
