# SVL

Official implementation of **Revisiting Shadow Detection from a Vision-Language Perspective** for shadow detection.

## Code Structure

```text
SVL/
|-- train.py              # Training entry point
|-- test.py               # Testing / inference entry point
|-- cfg/
|   |-- sbu.yaml          # SBU training config
|   `-- istd.yaml         # ISTD training config
|-- tool/
|   |-- sbudata.py        # SBU dataset loader
|   |-- istddata.py       # ISTD dataset loader
|   `-- ucfdata.py        # UCF test dataset loader
|-- model/                # SVL / ShadowDINO model
|-- CLIP/                 # CLIP code and checkpoint
`-- dinov3/               # DINOv3 code and checkpoint
```

## Environment

Install the main dependencies:

```bash
pip install torch torchvision numpy pillow tqdm pyyaml scikit-image scikit-learn ftfy regex opencv-python pydensecrf
```

## Pretrained Weights

The code loads pretrained weights from the following default paths:

```text
CLIP/ckpt/ViT-L-14-336px.pt
dinov3/ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

## Dataset Paths

Dataset paths are in the following paths. Update them to match your local machine.

Training dataset paths are defined in:

```text
tool/sbudata.py
tool/istddata.py
```

Testing dataset paths are defined in:

```text
tool/sbudata.py
tool/istddata.py
tool/ucfdata.py
test.py
```

Expected dataset layouts:

```text
SBU-shadow/
|-- SBUTrain4KRecoveredSmall/
|   |-- ShadowImages/
|   `-- ShadowMasks/
`-- SBU-Test/
    |-- ShadowImages/
    `-- ShadowMasks/

ISTD/
|-- train/
|   |-- train_A/
|   `-- train_B/
`-- test/
    |-- test_A/
    `-- test_B/

UCF_shadow/
|-- InputImages/
`-- GroundTruth/
```

## Training

Train on SBU with multiple GPUs

```bash
torchrun --nproc_per_node=4 train.py --dataset sbu
```

Train on ISTD with multiple GPUs

```bash
torchrun --nproc_per_node=4 train.py --dataset istd
```

## Testing

Test on SBU:

```bash
python test.py --dataset sbu --ckpt_dir ./sbu/ckpt --ee_start 12000 --ee_end 20001 --ee_step 2000
```

Test on ISTD:

```bash
python test.py --dataset istd --ckpt_dir ./istd/ckpt --ee_start 12000 --ee_end 20001 --ee_step 2000
```

Test on UCF:

```bash
python test.py --dataset ucf --ckpt_dir ./sbu/ckpt --ee_start 12000 --ee_end 20001 --ee_step 2000
```
