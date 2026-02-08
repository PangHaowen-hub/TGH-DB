import os
import ast
import torchio as tio
from torch.utils.data import DataLoader

def load_data(train_txt, val_txt, template_dir, batch_size, patch_size=(224, 224, 224), num_workers=4):
    subjects_train_list = []
    with open(train_txt, "r") as f:
        for line in f.readlines():
            subjects_train_list.append(ast.literal_eval(line.strip('\n')))

    subjects_train = []
    for name in subjects_train_list:
        template_mean_path = os.path.join(template_dir, "Mean.nii.gz")
        template_sd_path = os.path.join(template_dir, "SD.nii.gz")
        
        subject = tio.Subject(
            t1=tio.ScalarImage(name['t1']),
            flair=tio.ScalarImage(name['flair']),
            pet=tio.ScalarImage(name['pet']), 
            template_mean=tio.ScalarImage(template_mean_path),
            template_sd=tio.ScalarImage(template_sd_path),
            subject_id=name['subject_id'], 
            visit=name['visit'], 
            dataset=name['dataset']
        )
        subjects_train.append(subject)

    subjects_val_list = []
    with open(val_txt, "r") as f:
        for line in f.readlines():
            subjects_val_list.append(ast.literal_eval(line.strip('\n')))

    subjects_val = []
    for name in subjects_val_list:
        template_mean_path = os.path.join(template_dir, "Mean.nii.gz")
        template_sd_path = os.path.join(template_dir, "SD.nii.gz")
        
        subject = tio.Subject(
            t1=tio.ScalarImage(name['t1']),
            flair=tio.ScalarImage(name['flair']),
            pet=tio.ScalarImage(name['pet']), 
            template_mean=tio.ScalarImage(template_mean_path),
            template_sd=tio.ScalarImage(template_sd_path),
            subject_id=name['subject_id'], 
            visit=name['visit'], 
            dataset=name['dataset']
        )
        subjects_val.append(subject)

    transform = tio.Compose([
        tio.transforms.RescaleIntensity(include=('t1', 'flair'), out_min_max=(0, 1), percentiles=(0.5, 99.5)),
        tio.transforms.RescaleIntensity(include=('pet', 'template_mean', 'template_sd'), out_min_max=(0, 1), in_min_max=(0, 5)),
        tio.transforms.CropOrPad(patch_size) 
    ])

    train_subjects_ds = tio.SubjectsDataset(subjects_train, transform)
    val_subjects_ds = tio.SubjectsDataset(subjects_val, transform)

    print(f"Training set: {len(train_subjects_ds)} subjects, Validation set: {len(val_subjects_ds)} subjects")
    print(f"Patch Size: {patch_size}")

    return DataLoader(train_subjects_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers), \
           DataLoader(val_subjects_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)