import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# Paths
IMG_DIR = 'data/ISIC_2019_Training_Input/ISIC_2019_Training_Input'
CSV_PATH = 'data/ISIC_2019_Training_GroundTruth.csv'

df = pd.read_csv(CSV_PATH)
classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

# Show sample images per class
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Sample Images - Each Class', fontsize=16)

for idx, cls in enumerate(classes):
    sample = df[df[cls] == 1.0].iloc[0]['image']
    img_path = os.path.join(IMG_DIR, sample + '.jpg')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    ax = axes[idx//4][idx%4]
    ax.imshow(img)
    ax.set_title(cls)
    ax.axis('off')

plt.tight_layout()
plt.savefig('data/sample_images.png')
plt.show()
print("Sample images saved!")
import matplotlib.pyplot as plt
import os

# Load data
df_truth = pd.read_csv('data/ISIC_2019_Training_GroundTruth.csv')
df_meta = pd.read_csv('data/ISIC_2019_Training_Metadata.csv')

print("=== Dataset Info ===")
print(f"Total images: {len(df_truth)}")
print(f"\nColumns: {df_truth.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df_truth.head())

print("\n=== Class Distribution ===")
classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
for c in classes:
    count = df_truth[c].sum()
    print(f"{c}: {int(count)} images")

# Plot class distribution
counts = [int(df_truth[c].sum()) for c in classes]
plt.figure(figsize=(10, 5))
plt.bar(classes, counts, color='steelblue')
plt.title('ISIC 2019 - Class Distribution')
plt.xlabel('Lesion Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('data/class_distribution.png')
plt.show()
print("\nPlot saved!")