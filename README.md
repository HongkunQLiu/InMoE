# InMoE

Official reviewer-facing code release for **InMoE: Interaction-Aware Graph Mixture of Experts for Trajectory Prediction**.

This repository contains the extracted training, validation, and testing code for the two configurations used in our experiments:

- `InMoE-Argoverse`: Argoverse 1 motion forecasting
- `InMoE-INTERACTION`: INTERACTION multi-agent motion forecasting

The repository is intentionally compact and includes only the files required to inspect the model, reproduce training, and run validation/testing for the released InMoE variants.

## Repository Structure

```text
InMoE/
├── InMoE-Argoverse/
│   ├── train.py
│   ├── val.py
│   ├── test.py
│   ├── model/
│   ├── modules/
│   ├── layers/
│   ├── datamodules/
│   ├── datasets/
│   ├── losses/
│   ├── metrics/
│   ├── transforms/
│   ├── utils/
│   └── visualization/
└── InMoE-INTERACTION/
    ├── train.py
    ├── val.py
    ├── test.py
    ├── model/
    ├── modules/
    ├── layers/
    ├── datamodules/
    ├── datasets/
    ├── losses/
    ├── metrics/
    ├── transforms/
    └── utils/
```

## Environment

We recommend the following environment as a close match to the one used in our experiments.

### 1. Create a conda environment

```bash
conda create -n InMoE python=3.8
conda activate InMoE
```

### 2. Install PyTorch and PyTorch Geometric

```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.3.1
conda install pytorch-lightning==2.0.3 -c conda-forge
```

### 3. Install dataset-specific dependencies

For Argoverse:

```bash
pip install argoverse
```

If needed, please also install the official [argoverse-api](https://github.com/argoverse/argoverse-api).

For INTERACTION:

```bash
pip install lanelet2==1.2.1
```

Additional common packages:

```bash
pip install numpy pandas tqdm transformers
```

## Datasets

### Argoverse 1

Please organize the Argoverse 1 dataset as:

```text
/path/to/Argoverse_root/
├── train/
│   └── data/
│       ├── 1.csv
│       ├── 2.csv
│       └── ...
├── val/
│   └── data/
│       ├── 1.csv
│       ├── 2.csv
│       └── ...
└── test/
    └── data/
        ├── 1.csv
        ├── 2.csv
        └── ...
```

### INTERACTION

Please organize the INTERACTION dataset as:

```text
/path/to/INTERACTION_root/
├── maps/
├── train/
│   ├── ...
├── val/
│   ├── ...
└── test_multi-agent/
    ├── ...
```

## Training

### Argoverse 1

```bash
cd InMoE-Argoverse

python train.py \
  --root /path/to/Argoverse_root \
  --train_batch_size 2 \
  --val_batch_size 2 \
  --devices 8
```

Common optional arguments:

- `--max_epochs 100`
- `--num_workers 4`
- `--sample_rate 1`
- `--ckpt_path /path/to/checkpoint.ckpt` for resumed training

### INTERACTION

```bash
cd InMoE-INTERACTION

python train.py \
  --root /path/to/INTERACTION_root \
  --train_batch_size 2 \
  --val_batch_size 2 \
  --devices 8
```

Common optional arguments:

- `--max_epochs 64`
- `--num_workers 4`
- `--sample_rate 1`
- `--ckpt_path /path/to/checkpoint.ckpt` for resumed training

## Validation

### Argoverse 1

```bash
cd InMoE-Argoverse

python val.py \
  --root /path/to/Argoverse_root \
  --val_batch_size 2 \
  --devices 8 \
  --ckpt_path /path/to/checkpoint.ckpt
```

### INTERACTION

```bash
cd InMoE-INTERACTION

python val.py \
  --root /path/to/INTERACTION_root \
  --val_batch_size 2 \
  --devices 8 \
  --ckpt_path /path/to/checkpoint.ckpt
```

## Testing

### Argoverse 1

```bash
cd InMoE-Argoverse

python test.py \
  --root /path/to/Argoverse_root \
  --test_batch_size 2 \
  --devices 1 \
  --ckpt_path /path/to/checkpoint.ckpt
```

### INTERACTION

```bash
cd InMoE-INTERACTION

python test.py \
  --root /path/to/INTERACTION_root \
  --test_batch_size 2 \
  --devices 1 \
  --ckpt_path /path/to/checkpoint.ckpt
```

## Main Model Files

The primary model definitions used in this release are:

- Argoverse:
  - `InMoE-Argoverse/model/InMoE_real2Shared8MoE_CoPRBias_stepLR_QLinear64_Lasttriplelayer.py`
  - `InMoE-Argoverse/modules/backbone_real2Shared8MoE_CoPRbias_QLinear64_Lasttriplelayer.py`

- INTERACTION:
  - `InMoE-INTERACTION/model/InMoE_sparselessMoE_triplelastlayer.py`
  - `InMoE-INTERACTION/modules/backbone_sparselessMoE_triplelastlayer.py`

## Notes

- This repository is a compact extraction intended for review and reproducibility.
- Preprocessing is performed automatically by the dataset classes when processed files are not found.
- The Argoverse and INTERACTION pipelines require their respective external dataset APIs and map dependencies.

## Citation

If you find this code useful, please cite the corresponding paper.

```bibtex
@article{inmoe2026,
  title   = {InMoE: Interaction-Aware Graph Mixture of Experts for Trajectory Prediction},
  author  = {Anonymous},
  journal = {Pattern Recognition},
  year    = {2026}
}
```
# InMoE
