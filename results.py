import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
from dataset import val_loader
from model import get_efficientnet, get_vit

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

def evaluate_model(model, loader, model_name):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'data/{model_name}_confusion_matrix.png')
    plt.close()
    
    # Classification Report
    report = classification_report(all_labels, all_preds, 
                                   target_names=CLASSES, output_dict=True)
    
    print(f"\n=== {model_name} Results ===")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    print(f"Weighted Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Weighted Recall: {report['weighted avg']['recall']:.4f}")
    
    return report

# Load all 3 models
print("Loading models...")

# EfficientNet
eff_model = get_efficientnet()
eff_model.load_state_dict(torch.load('data/efficientnet_best.pth', map_location=DEVICE))
eff_model = eff_model.to(DEVICE)

# ViT
vit_model = get_vit()
vit_model.load_state_dict(torch.load('data/vit_best.pth', map_location=DEVICE))
vit_model = vit_model.to(DEVICE)

# KD EfficientNet
kd_model = get_efficientnet()
kd_model.load_state_dict(torch.load('data/kd_efficientnet_best.pth', map_location=DEVICE))
kd_model = kd_model.to(DEVICE)

print("All models loaded! Evaluating...")

eff_report = evaluate_model(eff_model, val_loader, 'EfficientNet')
vit_report = evaluate_model(vit_model, val_loader, 'ViT')
kd_report = evaluate_model(kd_model, val_loader, 'KD_EfficientNet')

# Comparison Bar Chart
models = ['EfficientNet', 'ViT', 'KD-EfficientNet']
f1_scores = [
    eff_report['weighted avg']['f1-score'],
    vit_report['weighted avg']['f1-score'],
    kd_report['weighted avg']['f1-score']
]
accuracies = [
    eff_report['accuracy'],
    vit_report['accuracy'],
    kd_report['accuracy']
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#2196F3', '#FF9800', '#4CAF50']

axes[0].bar(models, f1_scores, color=colors)
axes[0].set_title('Weighted F1 Score Comparison')
axes[0].set_ylabel('F1 Score')
axes[0].set_ylim(0, 1)
for i, v in enumerate(f1_scores):
    axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

axes[1].bar(models, accuracies, color=colors)
axes[1].set_title('Accuracy Comparison')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0, 1)
for i, v in enumerate(accuracies):
    axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('data/model_comparison.png')
plt.show()
print("\nAll results saved!")