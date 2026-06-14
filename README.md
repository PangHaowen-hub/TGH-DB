# TGH-DB: Template-Guided Heteroscedastic Diffusion Bridge for Brain MRI-to-PET Synthesis

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
TGH-DB/
├── A4/
│   ├── train_pairs.txt    # Training pairs (one dict per line)
│   ├── val_pairs.txt
│   └── test_pairs.txt
├── HABS/
│   ├── train_pairs.txt
│   ├── val_pairs.txt
│   └── test_pairs.txt
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
{'t1': 'path/to/t1.nii.gz', 'flair': 'path/to/flair.nii.gz', 'pet': 'path/to/pet.nii.gz', 'subject_id': 'xxx', 'visit': 'xxx', 'dataset': 'xxx'}
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

## Citation

If you use this code, please cite our paper. Please note that the paper has not yet been indexed or retrieved by academic search databases.

```bibtex
@inproceedings{pang2026tghdb,
  title={TGH-DB: Template-Guided Heteroscedastic Diffusion Bridge for Brain MRI-to-PET Synthesis},
  author={Pang, Haowen and Zhu, Yitao and Fu, Yingji and Ye, Chuyang and Qiu, Anqi},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2026},
  organization={Springer}
}
```
