<!-- # WeedsGalore Dataset :seedling::herb:

This is the official implementation of the WACV 2025 paper **WeedsGalore: A Multispectral and Multitemporal UAV-based Dataset for Crop and Weed Segmentation in Agricultural Maize Fields.**
WeedsGalore is a UAV-based multispectral dataset with dense annotations for crop and weed segmentation in maize fields.
This repository contains code and download links for the dataset and pretrained models.

[[`arXiv`](https://arxiv.org/abs/2502.13103)], [[`paper`](https://openaccess.thecvf.com/content/WACV2025/html/Celikkan_WeedsGalore_A_Multispectral_and_Multitemporal_UAV-Based_Dataset_for_Crop_and_WACV_2025_paper.html)], [[`dataset`](https://doidata.gfz.de/weedsgalore_e_celikkan_2024/)]

<a href="/img.png" target="_blank">
  <img src="/img.png" alt="WeedsGalore Preview" width="800"/>
</a>

## Dataset
### Download
Follow this [link](https://doidata.gfz.de/weedsgalore_e_celikkan_2024/) to download the dataset. The dataset (`weedsgalore-dataset`, 0.4GB) and full-field orthomosaics (`weedsgalore-orthomosaic`, 12GB, GeoTIFF) can be downloaded separately.

### Structure

```
weedsgalore-dataset
└── 2023-05-25
    └── images
    └── semantics
    └── instances
    └── logs
└── 2023-05-30
    └── images
    └── ...
└── ...
└── splits
    └── train.txt
    └── ...
└── LICENSE.txt
```


### Licence
WeedsGalore dataset is distributed under the [Creative Commons Attribution (CC BY) Licence](https://creativecommons.org/licenses/by/4.0/).
Please refer to the [full licence text](https://doidata.gfz.de/weedsgalore_e_celikkan_2024/) for details.

## Evaluation

### Requirements
Make sure to have the necessary dependencies installed. They are listed in `requirements.txt`.

### Install Packages
Example: Create a conda environment and install dependencies:
```
conda create -n weedsgalore python=3.7.12 -c conda-forge
conda activate weedsgalore
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install absl-py=1.3.0
conda install pillow=9.0.1
pip install torchmetrics==0.11.4
```

Run the evaluation script, replacing the flags with your paths and parameters:
```
python src/evaluate.py --dataset_path <weedsgalore-dataset_directory> --split test --ckpt <ckpt_directory> --in_channels 5 --num_classes 6
```

Inference with probabilistic model:
```
python src/evaluate_vimc.py --dataset_path <weedsgalore-dataset_directory> --split test --ckpt <ckpt_directory> --in_channels 5 --num_classes 3 --mc_samples=5
```

You can download pretrained models for DeepLabv3+ [here](https://doidata.gfz.de/weedsgalore_e_celikkan_2024/ckpts.zip) (for both MSI and RGB input, uni-weed and multi-weed case, deterministic and probabilistic variants).

## Training
Run the training script, replacing the flags with your paths and parameters (set `dlv3p_do=True` to run the probabilistic variant):
```
python src/train.py --dataset_path <weedsgalore-dataset_directory> --dataset_size_train 104 --in_channels 5 --num_classes 3 --dlv3p_do True --pretrained_backbone True --ckpt_resnet <path-to-backbone-weights> --batch_size 8 --num_workers 4 --lr 0.001 --epochs 100 --out_dir <output_directory> --log_interval 25 --ckpt_interval 100 -->

````

## Experiment Notes: 5-band SegFormer Refinement

This section records the architecture changes and the ablation results used in the
current experiments. The goal was to improve the small and thin `crop`/`weed`
regions while keeping the model much lighter than the ResNet50 baseline.

### Reference baseline

The original deterministic reference is a ResNet50 DeepLabV3+ model:

```text
5-band UAV input: R, G, B, NIR, RE
                |
                v
ResNet50 backbone (ImageNet pretrained)
                |
                +--> high-level feature
                |        |
                |        v
                |      ASPP (dense multi-scale convolutions)
                |
                +--> low-level skip feature used by the DeepLabV3+ decoder
                                 |
                                 v
                            Decoder
                                 |
                                 v
                    background / crop / weed
````

In the paper-style comparison, the main semantic path is often summarized as
`ResNet50 -> high-level feature -> ASPP -> decoder`. In this repository's
`DeepLabHeadV3Plus`, the low-level skip is also projected and fused in the decoder.
The baseline uses a dense ASPP and a much larger ResNet50 backbone.

### Architecture versions explored

#### 1. ResNet50 baseline

```text
RGB + NIR + RE -> ResNet50 -> ASPP -> DeepLabV3+ decoder -> 3 classes
```

The plain 5-channel ResNet path replaces `conv1` with a 5-channel convolution.
This is not the same as the spectral-guided path: NIR and RE are processed by
the whole RGB backbone rather than by a separate lightweight spectral encoder.

#### 2. ResNet50 with CBAM / SR-CBAM

CBAM was tested after ASPP. It adds channel and spatial attention, but increases
parameters, FLOPs and latency substantially. The attention variants did not give
a consistent improvement over the baseline, so they are not the final model.

#### 3. ResNet34 spectral-guided + Lite-ASPP

The first multispectral design separated RGB from the extra spectral bands:

```text
RGB -----------------------> ResNet34
NIR + RE (+ NDVI + NDRE) --> TinySpectralEncoder
                                                                            |
                                                                            v
                                                         spatial Spectral Feature Map
                                                                            |
                                                 +------------+------------+
                                                 v                         v
                                        Spectral Gate-low        Spectral Gate-high
                                                 |                         |
                                     low-level RGB             high-level RGB
                                                 |                         |
                                                 +------> Lite-ASPP -> Decoder
```

Important implementation details:

- `TinySpectralEncoder` uses one depthwise 3x3 convolution, one pointwise 1x1
  convolution, BatchNorm and ReLU.
- It keeps a spatial map `(N, 32, H, W)`; it does not use global average pooling.
- With `use_spectral_indices=True`, the encoder input is
  `[NIR, RE, NDVI-like, NDRE-like]`.
- The same spectral map is reused for both low-level and high-level gates.
- The original multiplicative gate was:

  $$F' = F \odot G_c \odot G_s$$

  where both gates use sigmoid. This can only suppress the RGB feature because
  $0 < G_c, G_s < 1$.

- A residual alternative was tested:

  $$
  F' = F + F \odot G_c \odot G_s
           = F \odot (1 + G_c \odot G_s)
  $$

  but it was not retained as the final winning configuration because the
  experiments did not show a reliable improvement.

- An ECA experiment was also tested on the high-level path. It lowered the
  result in the observed run, so ECA is not used in the final model.

Output-stride-8 dilation was tested for ResNet34 by adding dilation support to
`BasicBlock` and using `[False, True, True]`. It raised FLOPs from about 22.78G
to 95.53G and reduced throughput from about 80.86 to 22.17 img/s without
improving validation mIoU. The final SegFormer direction was therefore preferred.

#### 4. Custom 5-band SegFormer

The custom encoder in `src/nets/segformer.py` was initially trained from scratch:

```text
5-band input
        |
        v
Overlap patch stem: 5 -> 32 channels
        |
        +--> F1: 32 channels, 1/4 resolution
        +--> F2: 64 channels, 1/8 resolution
        +--> F3: 160 channels, 1/16 resolution
        +--> F4: 256 channels, 1/32 resolution
        |
        v
Linear projections -> upsample to F1 -> concatenate
        |
        v
Lightweight decoder -> 3-class classifier
```

Each transformer block contains efficient self-attention and a Mix-FFN with a
depthwise convolution. This version was very efficient but its randomly
initialized encoder underperformed the pretrained ResNet baseline.

#### 5. Pretrained MiT-B0 SegFormer

The custom encoder was then replaced by the official `nvidia/mit-b0` encoder
from Transformers. The decoder was kept in the repository and the input stem
was adapted from 3 to 5 channels:

```text
pretrained RGB filters:  R, G, B  <- original MiT-B0 weights
new spectral filters:    NIR, RE  <- mean of the pretrained RGB filters
```

This is implemented by `PretrainedMiTEncoder` in `src/nets/segformer.py`.
Only the first projection changes its input channel count; the MiT feature
widths remain `[32, 64, 160, 256]`. `pretrained_backbone=True` has a real
effect only for the SegFormer path after this integration. It does not load a
previous experiment's `best_val.pth`; it loads ImageNet-pretrained MiT-B0
weights from Hugging Face.

#### 6. Final refine-decoder

The decisive improvement was not another attention gate. It was the
`LowLevelRefine` branch and the two-step decoder in `src/nets/segformer.py`:

```text
MiT F1/F2/F3/F4 -> multi-scale projection -> fuse at 1/4 resolution
                                                                            |
                                                                            v
                                                         coarse decoder feature
                                                                            |
                                                 bilinear upsample 1/4 -> 1/2
                                                                            |
Raw RGB + NIR + RE -> LowLevelRefine (5 -> 48, stride 2)
                                                                            |
                                        concatenate coarse + raw low-level detail
                                                                            |
                                                            3x3 conv + DW 3x3
                                                                            |
                                                        classifier at 1/2 resolution
                                                                            |
                                                            final 2x upsample
                                                                            |
                                                             3-class output
```

Before refinement, the decoder predicted at 1/4 resolution and performed one
direct 4x bilinear upsample. The new decoder performs `2x upsample -> feature
fusion/refinement -> 2x upsample`. This preserves edge, texture and thin-object
information from the raw five-band image. It adds a small convolutional branch,
but it does not add a second spectral encoder or a global spectral gate.

The key change is therefore:

```text
Old:  coarse 1/4-resolution decoder -> bilinear 4x -> classifier
New:  coarse 1/4 -> bilinear 2x -> raw low-level 1/2 fusion -> refine -> bilinear 2x
```

This explains why `crop` and especially `weed` improved together: both classes
depend on boundaries and fine spatial detail. The result supports the observed
hypothesis that the main bottleneck was decoder spatial reconstruction rather
than a lack of another attention module.

### Loss and calibration changes

The training script now supports class-weighted CE and soft Dice:

$$L_{CE} = -w_y \log p_y$$

For Dice, probabilities are used instead of hard predictions:

$$
Dice_c = \frac{2\sum p_c y_c + \epsilon}
                                 {\sum p_c + \sum y_c + \epsilon}
$$

$$L_{Dice} = 1 - \frac{1}{C}\sum_c Dice_c$$

The combined objective is:

$$L = (1 - \lambda)L_{CE} + \lambda L_{Dice}$$

The winning refine-decoder run used:

```text
class weights: [background=0.9, crop=1.0, weed=1.3]
Dice weight:   lambda=0.3
```

The implementation is `soft_dice_loss` in `src/train.py`. Dice and class weights
improve region overlap, but they can make confidence less calibrated. Therefore
ECE is handled separately after training with `src/calibrate_temperature.py`.

Temperature scaling uses:

$$p_T = softmax(z / T)$$

It does not change `argmax`, so mIoU and IoU remain unchanged. The scalar $T$
is fitted on validation only, then reused unchanged on test to avoid test leakage.

### Ablation table

Values below are the recorded experiments. `val` and `test` must not be mixed
when comparing models.

| Experiment                  | Encoder / decoder                                 | Loss or attention                | Params |   FLOPs | Inference / throughput |   Val mIoU |   Val crop |   Val weed |  Test mIoU |              ECE |
| --------------------------- | ------------------------------------------------- | -------------------------------- | -----: | ------: | ---------------------- | ---------: | ---------: | ---------: | ---------: | ---------------: |
| Published baseline          | ResNet50 + dense ASPP + DeepLabV3+                | CE                               | 39.76M | 173.82G | 116.57 ms / 8.58 img/s |     82.90% |     73.93% |     77.31% |          - |           0.0018 |
| ResNet50 + CBAM             | ResNet50 + ASPP                                   | CBAM                             | 39.77M | 173.82G | 109.88 ms / 9.10 img/s |     84.10% |          - |          - |          - |                - |
| ResNet34 spectral, OS32     | ResNet34 + spectral branch + Lite-ASPP            | CE                               | 21.94M |  22.78G | 12.41 ms / 80.60 img/s |     82.24% |          - |          - |          - |                - |
| ResNet34 spectral + indices | ResNet34 + NIR/RE + NDVI/NDRE + gates + Lite-ASPP | CE                               | 21.94M |  22.78G | 12.37 ms / 80.86 img/s |     81.88% |     73.62% |     74.78% |     81.55% |           0.0057 |
| ResNet34 spectral + ECA     | Same, ECA added                                   | ECA                              | 21.94M |  22.78G | 14.02 ms / 71.34 img/s |     78.56% |          - |          - |          - |                - |
| ResNet34 spectral, OS8      | Dilated BasicBlock + Lite-ASPP                    | residual gate                    | 21.94M |  95.53G | 45.10 ms / 22.17 img/s |     80.65% |          - |          - |          - |                - |
| SegFormer scratch           | Custom MiT-like encoder + lightweight decoder     | CE                               |  3.38M |   3.84G | 6.78 ms / 147.53 img/s |     80.85% |     71.07% |     73.31% |          - |          0.00425 |
| SegFormer + pretrained MiT  | MiT-B0 + existing decoder                         | CE                               |  3.46M |   3.49G | 9.90 ms / 101.03 img/s |     82.07% |     73.71% |     74.37% |          - |          0.00544 |
| MiT + CE + Dice (lambda=.5) | MiT-B0 + existing decoder                         | weights [0.8,1.0,1.5]            |  3.46M |   3.49G | 9.83 ms / 101.72 img/s |     82.62% |     74.62% |     75.15% |          - |           0.0120 |
| MiT + Dice refine-decoder   | MiT-B0 + `LowLevelRefine`                         | weights [0.9,1.0,1.3], lambda=.3 |  3.66M |  17.09G | 16.53 ms / 60.49 img/s | **84.63%** | **76.72%** | **78.03%** | **85.64%** | test ECE 0.00506 |

The final row used `best_val.pth` selected on validation. Its validation
temperature was `T=1.2215`, giving ECE `0.00749` on val and `0.00506` on test;
the test mIoU remained `85.64%` because temperature scaling leaves argmax
predictions unchanged.

### Reproduction commands for the final model

Train:

```bash
python src/train.py \
    --dataset_path=/kaggle/working/weedsgalore-dataset \
    --dataset_size_train=104 \
    --segformer=True --spectral_guided=False \
    --use_spectral_indices=False --use_attention=False --dlv3p_do=False \
    --in_channels=5 --num_classes=3 --pretrained_backbone=True \
    --batch_size=8 --num_workers=4 --lr=0.0003 \
    --weight_bg=0.9 --weight_crop=1.0 --weight_weed=1.3 \
    --use_dice_loss=True --dice_weight=0.3 --epochs=100 \
    --out_dir=/kaggle/working/out_dir_segformer_refine1 \
    --log_interval=25 --ckpt_interval=100
```

Fit temperature on validation:

```bash
python src/calibrate_temperature.py \
    --dataset_path=/kaggle/working/weedsgalore-dataset \
    --ckpt=/kaggle/working/out_dir_segformer_refine1/best_val.pth \
    --segformer=True --pretrained_backbone=True \
    --spectral_guided=False --in_channels=5 --num_classes=3
```

Evaluate with the fitted temperature:

```bash
python src/evaluate.py \
    --dataset_path=/kaggle/working/weedsgalore-dataset \
    --split=test \
    --ckpt=/kaggle/working/out_dir_segformer_refine1/best_val.pth \
    --segformer=True --pretrained_backbone=True \
    --spectral_guided=False --in_channels=5 --num_classes=3 \
    --temperature=1.2215
```

### Final interpretation

The ablation supports the following conclusion:

1. Adding NDVI/NDRE or another attention module alone is not sufficient.
2. Pure multiplicative spectral gating can suppress useful RGB features and did
   not reliably improve the result.
3. The pretrained MiT encoder supplies useful ImageNet initialization for the
   RGB part of the five-band input.
4. The largest consistent gain came from the decoder refinement path, which
   restores high-resolution raw-band detail before the final prediction.
5. CE + Dice with moderate class weights improved overlap, while temperature
   scaling handled confidence calibration separately.

The current best model is therefore best described as:

```text
5-band UAV input
    -> pretrained MiT-B0 with 3-to-5-channel stem adaptation
    -> four-scale SegFormer features
    -> lightweight multi-scale decoder
    -> raw five-band 1/2-resolution LowLevelRefine branch
    -> convolutional refinement and final upsampling
    -> 3-class segmentation
```

<!-- ## License
This project is licensed under the Apache-2.0 License. See LICENSES folder for details.
```
   Copyright 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
   Copyright 2024 Ekin Celikkan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```


## Citation
If you use the dataset or code, please cite our paper:

```
@InProceedings{Celikkan_2025_WACV,
    author    = {Celikkan, Ekin and Kunzmann, Timo and Yeskaliyev, Yertay and Itzerott, Sibylle and Klein, Nadja and Herold, Martin},
    title     = {WeedsGalore: A Multispectral and Multitemporal UAV-Based Dataset for Crop and Weed Segmentation in Agricultural Maize Fields},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {4767-4777}
}
``` -->
