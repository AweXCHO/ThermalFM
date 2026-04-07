# TEXNet
This page contains the architecture, implementation details, and the code (in PyTorch) for "TeX-Net", a DNN to decompose heat cubes (hyperspectral thermal images) into Temperature, emissivity, and Texture maps.

## Contents

- [TEXNet](#texnet)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Implementation Details](#implementation-details)
    - [Dependencies](#dependencies)
    - [Train](#train)
    - [Validation](#validation)
  - [Results and Pre-trained Weights](#results-and-pre-trained-weights)
  - [Contacts](#contacts)

## Introduction

<p align="center">
  <img src="https://github.com/FanglinBao/HADAR/blob/main/TeXNet/TeXNet.png" />
</p>

Architecture of our TeX-Net for inverse TeX decomposition. TeX-Net is physics-inspired for three aspects. Firstly, TeX decomposition of heat cubes relies on both spatial patterns and spectral thermal signatures. This inspires the adoption of spectral and pyramid (spatial) attention layers in the UNet model. Secondly, due to TeX degeneracy, the mathematical structure has to be specified to ensure the uniqueness of inverse mapping, and hence it is essential to learn thermal lighting factors $V$ instead of texture <em>X</em>. That is, TeX-Net cannot be trained end-to-end. Here, α,β and γ are indices of objects, and ν is the wavenumber. X<sub>α</sub> is constructed with <em>V</em> and S<sub>βν</sub> indirectly, where S<sub>βν</sub> is the down-sampled S<sub>αν</sub> to approximate <em>k</em> most significant environmental objects. Thirdly, the material library <em>M</em> and its dimension are key to the network. TeX-Net can either be trained with ground truth <em>T</em>, <em>m</em>, and <em>V</em> in supervised learning, or alternatively, with material library <em>M</em>, Planck's law B<sub>ν</sub>(T<sub>α</sub>), and the mathematical structure of X<sub>αν</sub> in unsupervised learning. In supervised learning, the loss function is a combination of individual losses with regularization hyper-parameters. In unsupervised learning, the loss function defined on the re-constructed heat cube is based on physics models of the heat signal. In practice, a hybrid loss function with T, e, V contributions (50%) in addition to the physics-based loss (50%) is used.

## Implementation Details

### Dependencies
All the dependencies are included in a .yml file that can be downloaded from the following link:<br />
[OneDrive](https://purdue0-my.sharepoint.com/:u:/g/personal/sjape_purdue_edu/EfY-p4JvzfxJtVGYUKba0FMBZZwZCEPrXpAud05nsuo-pg?e=v0JgaQ)

```
conda env create -f texnet.yml
```

### Train
The following command can be treated as an example to train TeXNet on the HADAR databse

```
CUDA_VISIBLE_DEVICES=0 python main_single.py --ngpus 1 --backbone resnet50 --data_dir /mnt/dqdisk/Code/HADAR/TeXNet/hadar --workers 8 --epochs 40000 --checkpoint_dir supervised_crop_224_single --lr 1e-3 --weight-decay 1e-3 --train_T --train_v --no_log_images --eval_every 500 --res full --batch-size 10 --seed 42
```
CUDA_SIVIBLE_DEVICES=0 python main_mask2former.py --ngpus 1 --backbone resnet50 --data_dir /mnt/dqdisk/Code/HADAR/TeXNet/hadar --workers 8 --epochs 40000 --checkpoint_dir supervised_crop_224_mask2former_fpn --lr 1e-3 --weight-decay 1e-3 --train_T --train_v --no_log_images --eval_every 500 --res full --batch-size 20 --seed 42
### Validation
The following command can be treated as an example to validate the trained model

```
CUDA_VISIBLE_DEVICES=1 python main_mask2former_slide_val.py --ngpus 1 --backbone resnet50 --data_dir /mnt/dqdisk/Code/HADAR/TeXNet/hadar --workers 8 --epochs 40000 --checkpoint_dir ./slide_output/show --lr 1e-3 --weight-decay 1e-3 --train_T --train_v --no_log_images --eval_every 500 --res full --batch-size 20 --seed 42 --resume '/mnt/dqdisk/Code/HADAR/TeXNet/supervised_crop_224_mask2former_multy/checkpoints/epoch=38999-step=195000.ckpt' --eval
```
CUDA_VISIBLE_DEVICES=1 python main.py --ngpus 1 --backbone resnet50 --data_dir /mnt/dqdisk/Code/HADAR/TeXNet/hadar --workers 8 --epochs 40000 --checkpoint_dir supervised_check_crop_originmethod --lr 1e-3 --weight-decay 1e-3 --train_T --train_v --no_log_images --eval_every 500 --res full --batch-size 20 --seed 42 --resume '/mnt/dqdisk/Code/HADAR/TeXNet/supervised_crop_224_originmethod/checkpoints/epoch=37499-step=337500.ckpt' --eval

## Results and Pre-trained Weights

The TeX-Net outputs T, e, and V. Please run visualization after evaluating the TeX-Net to get the resMap. Please see the demo codes in 'Check_TeX_Vision_for_TeXNet_Outputs.txt' in this folder to get TeX vision images for TeX-Net. The TeX-Net performance on the HADAR database is shown below.

<p align="center">
  <img src="https://github.com/FanglinBao/HADAR/blob/main/TeXNet/Fig3.png" />
</p>

Pre-trained weights and training loss curves are available through the following links:<br />

| Pre-trained weights |  Training loss curves |
|---|---|
|[OneDrive](https://purdue0-my.sharepoint.com/:f:/g/personal/sjape_purdue_edu/Ei3BWYRaU3BFt7e6C1YDu2MByBIb3_3WA96FuR2bXWgoyg?e=gF28b4)|[OneDrive](https://purdue0-my.sharepoint.com/:f:/g/personal/sjape_purdue_edu/Ei3BWYRaU3BFt7e6C1YDu2MByBIb3_3WA96FuR2bXWgoyg?e=gF28b4)|

## Contacts
