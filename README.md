# TGH-DB: Template-Guided Heteroscedastic Diffusion Bridge for Brain MRI-to-PET Synthesis

This work proposes **Template-Guided Heteroscedastic Diffusion Bridge (TGH-DB)** for brain MRI-to-PET synthesis. The framework introduces a population-level probabilistic metabolic template (voxel-wise mean and standard deviation) to steer the diffusion process toward a physiologically plausible target distribution. Unlike conventional diffusion models that drift toward an isotropic Gaussian prior, our forward process progressively guides PET signals toward the metabolic template with spatially adaptive, heteroscedastic noise.
The **Time-Aware Evolutionary Denoiser (TAED)** integrates multimodal MRI (T1 + FLAIR) via parallel encoders, a time-aware fusion mechanism, and a time-modulated decoding pathway.

## Project Structure

```
TGH-DB/
├── main_A4.py          # Training script for A4 dataset
├── main_HABS.py        # Training script for HABS dataset
├── test_A4.py          # Testing script for A4 dataset
├── test_HABS.py        # Testing script for HABS dataset
├── model.py            # TAED (Time-Aware Evolutionary Denoiser) network
├── diffusion.py        # DiffusionBridge: forward process & DDIM sampling
├── datasets.py         # Data loading and preprocessing
└── README.md
```

## Environment

```bash
pip install torch torchvision
pip install monai nibabel torchio tqdm numpy
```

## Data Preparation

### Directory Layout

Place your data as follows:

```
project_root/
├── A4/
│   ├── train_pairs.txt    # Training pairs (one dict per line)
│   └── val_pairs.txt      # Validation pairs
├── HABS/
│   ├── train_pairs.txt
│   └── val_pairs.txt
└── Template/
    ├── A4_FBP/
    │   ├── Mean.nii.gz    # Population mean template
    │   └── SD.nii.gz      # Population std template
    └── HABS_FBB/
        ├── Mean.nii.gz
        └── SD.nii.gz
```

### Pairs File Format

Each line in `train_pairs.txt` / `val_pairs.txt` is a Python dict (use `ast.literal_eval` to parse), e.g.:

```python
{'t1': 'path/to/t1.nii.gz', 'flair': 'path/to/flair.nii.gz', 'pet': 'path/to/pet.nii.gz', 'subject_id': 'xxx', 'visit': 'bl', 'dataset': 'A4'}
```

## Training

**A4 dataset:**
```bash
python main_A4.py
```

**HABS dataset:**
```bash
python main_HABS.py
```

Outputs are saved under `experiments/A4_TGH_DB/` or `experiments/HABS_TGH_DB/` with timestamp subfolders, including:

- `config.json` – training config
- `train.log` – training log
- `best_model.pth` – best checkpoint
- `model_epoch_*.pth` – periodic checkpoints
- `epoch_*_vis/` – validation visualizations

### Main Config Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_steps` | Diffusion steps | 1000 |
| `lr` | Learning rate | 1e-4 |
| `batch_size` | Batch size | 8 |
| `patch_size` | 3D patch size | (224, 224, 224) |
| `epochs` | Training epochs | 300 |

## Testing

**A4:**
```bash
python test_A4.py --model_path <path_to_model.pth> --config_path <path_to_config.json> --test_txt A4/val_pairs.txt --template_dir Template/A4_FBP --batch_size 8
```

**HABS:**
```bash
python test_HABS.py --model_path <path_to_model.pth> --config_path <path_to_config.json> --test_txt HABS/val_pairs.txt --template_dir Template/HABS_FBB --batch_size 8
```
