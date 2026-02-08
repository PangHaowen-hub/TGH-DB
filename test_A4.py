import os
import torch
import torch.nn as nn
import json
import argparse
from tqdm import tqdm
import numpy as np
import nibabel as nib
import ast
import torchio as tio
from torch.utils.data import DataLoader
from diffusion import DiffusionBridge
from model import TAED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(x):
    return (x - 0.5) / 0.5


def denormalize(x):
    return (x * 0.5) + 0.5


def load_test_data(test_txt, template_dir, batch_size=1, patch_size=(224, 224, 224)):
    subjects_test_list = []
    with open(test_txt, "r") as f:
        for line in f.readlines():
            subjects_test_list.append(ast.literal_eval(line.strip('\n')))

    subjects_test = []
    for name in subjects_test_list:
        template_mean_path = os.path.join(template_dir, "Mean.nii.gz")
        template_sd_path = os.path.join(template_dir, "SD.nii.gz")
        
        pet_img = tio.ScalarImage(name['pet'])
        subject = tio.Subject(
            t1=tio.ScalarImage(name['t1']),
            flair=tio.ScalarImage(name['flair']),
            pet=pet_img, 
            template_mean=tio.ScalarImage(template_mean_path),
            template_sd=tio.ScalarImage(template_sd_path),
            subject_id=name['subject_id'], 
            visit=name['visit'], 
            dataset=name['dataset'],
            original_shape=np.array(pet_img.spatial_shape),
            original_affine=pet_img.affine
        )
        subjects_test.append(subject)

    transform = tio.Compose([
        tio.transforms.RescaleIntensity(include=('t1', 'flair'), out_min_max=(0, 1), percentiles=(0.5, 99.5)),
        tio.transforms.RescaleIntensity(include=('pet', 'template_mean', 'template_sd'), out_min_max=(0, 1), in_min_max=(0, 5)),
        tio.transforms.CropOrPad(patch_size) 
    ])

    test_subjects_ds = tio.SubjectsDataset(subjects_test, transform)
    print(f"Test set: {len(test_subjects_ds)} subjects")
    print(f"Patch Size: {patch_size}")

    return DataLoader(test_subjects_ds, batch_size=batch_size, shuffle=False, num_workers=4)


def test_model(model_path, config_path, test_txt, template_dir, batch_size=1):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    patch_size = tuple(config.get('patch_size', (224, 224, 224)))
    
    net = TAED(
        img_size=patch_size,
        in_channels=2,
        out_channels=1,
        feature_size=24
    )
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    
    if isinstance(state_dict, dict) and len(state_dict) > 0:
        first_key = list(state_dict.keys())[0]
        if first_key.startswith('module.'):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            state_dict = new_state_dict
    
    net.load_state_dict(state_dict, strict=False)
    
    net = net.to(DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for testing.")
        net = nn.DataParallel(net)
    net.eval()
    
    diffusion_bridge = DiffusionBridge(
        n_steps=config['n_steps'],
        gamma=config['gamma'],
        n_recursions=config['n_recursions'],
        consistency_threshold=config['consistency_threshold']
    )
    diffusion_bridge = diffusion_bridge.to(DEVICE)
    diffusion_bridge.eval()
    
    testloader = load_test_data(test_txt, template_dir, batch_size=batch_size, patch_size=patch_size)
    
    print(f"Starting test, {len(testloader)} batches in total...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(testloader, desc="Testing")):
            t1 = batch['t1']['data'].to(DEVICE)
            fl = batch['flair']['data'].to(DEVICE)
            pet = batch['pet']['data'].to(DEVICE)
            mu_atlas = batch['template_mean']['data'].to(DEVICE)
            sigma_atlas = batch['template_sd']['data'].to(DEVICE)
            
            pet = normalize(pet)
            mu_atlas = normalize(mu_atlas)
            t1 = normalize(t1)
            fl = normalize(fl)
            sigma_atlas = sigma_atlas / 0.5
            
            skip_steps = 100
            generated_pet = diffusion_bridge.ddim_predict_step(t1, fl, mu_atlas, sigma_atlas, net, skip_steps=skip_steps)
            
            for i in range(generated_pet.shape[0]):
                orig_shape = tuple(batch['original_shape'][i].tolist())
                orig_affine = batch['original_affine'][i].cpu().numpy()
                subject_id = batch['subject_id'][i]
                dataset = batch['dataset'][i]
                visit = batch['visit'][i]
                
                def restore(tensor, shape):
                    img = tio.ScalarImage(tensor=tensor.detach().cpu())
                    resizer = tio.CropOrPad(shape)
                    return resizer(img).data.squeeze().numpy()

                gen_img = restore(denormalize(generated_pet[i]), orig_shape)
                gen_img = gen_img.clip(0, 1)
                gen_img = gen_img * 5
                
                pred_path = os.path.join(model_path[:-4] + f'_skip_steps_{skip_steps}', dataset, subject_id, f'{visit}.nii.gz')
                os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                nib.save(nib.Nifti1Image(gen_img, orig_affine), pred_path)
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Diffusion Bridge model')
    parser.add_argument('--model_path', type=str, default='experiments/A4_TGH_DB/2026-01-12-03-29-30/model_epoch_300.pth', help='Path to model weights')
    parser.add_argument('--config_path', type=str, default='experiments/A4_TGH_DB/2026-01-12-03-29-30/config.json', help='Path to config file')
    parser.add_argument('--test_txt', type=str, default='A4/val_pairs.txt', help='Path to test data list')
    parser.add_argument('--template_dir', type=str, default='Template/A4_FBP', help='Path to template directory')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model_path,
        config_path=args.config_path,
        test_txt=args.test_txt,
        template_dir=args.template_dir,
        batch_size=args.batch_size
    )
