import os
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import time
import shutil
import logging
from model import TAED
import nibabel as nib
from datasets import load_data
from diffusion import DiffusionBridge

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logger(save_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = os.path.join(save_path, "train.log")
    formatter = logging.Formatter('%(levelname)s %(filename)s(%(lineno)d): %(message)s')
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def normalize(x):
    return (x - 0.5) / 0.5

def denormalize(x):
    return (x * 0.5) + 0.5

def train_and_test(diffusion_bridge, net, trainloader, valloader, optimizer, save_path, logger, epochs):
    
    criterion = nn.L1Loss() 
    best_loss = float('inf')
    
    accumulation_steps = 1 

    for epoch in range(1, epochs + 1):
        net.train()
        diffusion_bridge.train()
        epoch_loss = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
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

            t = torch.randint(1, diffusion_bridge.n_steps + 1, (pet.shape[0],), device=DEVICE)

            x_t = diffusion_bridge.q_sample(t, pet, mu_atlas, sigma_atlas)

            img_input_t1 = torch.cat([x_t, t1], dim=1)
            img_input_fl = torch.cat([x_t, fl], dim=1)
            
            pred_x0 = net(img_input_t1, img_input_fl, t)

            loss = criterion(pred_x0, pet)

            loss = loss / accumulation_steps
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': loss.item() * accumulation_steps})

        avg_train_loss = epoch_loss / len(trainloader)
        logger.info(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}")

        if epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                for val_batch in valloader:
                    t1_val = val_batch['t1']['data'].to(DEVICE)
                    fl_val = val_batch['flair']['data'].to(DEVICE)
                    pet_val = val_batch['pet']['data'].to(DEVICE)
                    mu_atlas_val = val_batch['template_mean']['data'].to(DEVICE)
                    sigma_atlas_val = val_batch['template_sd']['data'].to(DEVICE)
                    
                    pet_val = normalize(pet_val)
                    mu_atlas_val = normalize(mu_atlas_val)
                    t1_val = normalize(t1_val)
                    fl_val = normalize(fl_val)
                    sigma_atlas_val = sigma_atlas_val / 0.5
                    
                    generated_pet = diffusion_bridge.ddim_predict_step(t1_val, fl_val, mu_atlas_val, sigma_atlas_val, net)

                    vis_dir = os.path.join(save_path, f"epoch_{epoch}_vis")
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    affines = val_batch['pet']['affine']
                    subject_ids = val_batch['subject_id']
                    
                    for i in range(min(4, generated_pet.shape[0])):
                        sid = subject_ids[i]
                        affine = affines[i].cpu().numpy()
                        
                        gen_img = denormalize(generated_pet[i]).cpu().numpy().squeeze()
                        gt_img = denormalize(pet_val[i]).cpu().numpy().squeeze()
                        atlas_img = denormalize(mu_atlas_val[i]).cpu().numpy().squeeze()
                        t1_img = denormalize(t1_val[i]).cpu().numpy().squeeze()
                        
                        nib.save(nib.Nifti1Image(gen_img, affine), os.path.join(vis_dir, f"{sid}_gen.nii.gz"))
                        nib.save(nib.Nifti1Image(gt_img, affine), os.path.join(vis_dir, f"{sid}_gt.nii.gz"))
                        nib.save(nib.Nifti1Image(atlas_img, affine), os.path.join(vis_dir, f"{sid}_atlas.nii.gz"))
                        nib.save(nib.Nifti1Image(t1_img, affine), os.path.join(vis_dir, f"{sid}_t1.nii.gz"))
                    
                    val_mse = nn.MSELoss()(generated_pet, pet_val)
                    logger.info(f"[Epoch {epoch}] Val MSE: {val_mse.item():.6f}")
                    break

            torch.save(net.state_dict(), os.path.join(save_path, f"model_epoch_{epoch}.pth"))
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                torch.save(net.state_dict(), os.path.join(save_path, "best_model.pth"))
                logger.info(f"Saved Best Model at Epoch {epoch}")

if __name__ == "__main__":
    save_dir = 'experiments/A4_TGH_DB'
    now_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    save_path = os.path.join(save_dir, now_time)
    os.makedirs(save_path, exist_ok=True)
    logger = setup_logger(save_path)
    
    config = {
        "n_steps": 1000,
        "gamma": 1.0,
        "n_recursions": 2,
        "consistency_threshold": 1e-4,
        "lr": 1e-4,
        "batch_size": 8,
        "patch_size": (224, 224, 224),
        "num_workers": 8,
        "epochs": 300
    }
    
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    try:
        shutil.copy(os.path.abspath(__file__), save_path)
        if os.path.exists("diffusion.py"): shutil.copy("diffusion.py", save_path)
        if os.path.exists("datasets.py"): shutil.copy("datasets.py", save_path)
    except Exception as e:
        logger.warning(f"Backup failed: {e}")

    net = TAED(
        img_size=config['patch_size'],
        in_channels=2,
        out_channels=1,
        feature_size=24
    ).to(DEVICE)

    diffusion_bridge = DiffusionBridge(
        n_steps=config['n_steps'],
        gamma=config['gamma'],
        n_recursions=config['n_recursions'],
        consistency_threshold=config['consistency_threshold']
    )

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs.")
        net = nn.DataParallel(net)

    net = net.to(DEVICE)
    diffusion_bridge = diffusion_bridge.to(DEVICE)
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=1e-4)

    trainloader, valloader = load_data(
        r"A4/train_pairs.txt", 
        r"A4/val_pairs.txt", 
        r"Template/A4_FBP",
        batch_size=config['batch_size'],
        patch_size=config['patch_size'],
        num_workers=config['num_workers']
    )
    
    logger.info("Start Training...")
    train_and_test(
        diffusion_bridge,
        net,
        trainloader,
        valloader,
        optimizer,
        save_path,
        logger,
        epochs=config['epochs']
    )
