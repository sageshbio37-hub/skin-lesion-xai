import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import cv2
import os
import numpy as np

IMG_DIR = 'data/ISIC_2019_Training_Input/ISIC_2019_Training_Input'
CSV_PATH = 'data/ISIC_2019_Training_GroundTruth.csv'
CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

# Data Augmentation
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'] + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = torch.tensor(
            [row[c] for c in CLASSES], dtype=torch.float32
        ).argmax().item()
        if self.transform:
            img = self.transform(img)
        return img, label

# Train/Val split
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(0.8 * len(df))
train_df = df[:split]
val_df = df[split:]

train_dataset = SkinLesionDataset(train_df, IMG_DIR, train_transforms)
val_dataset = SkinLesionDataset(val_df, IMG_DIR, val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")

# Test one batch
imgs, labels = next(iter(train_loader))
print(f"Batch shape: {imgs.shape}")
print(f"Labels: {labels[:8]}")
print("Dataset ready!")