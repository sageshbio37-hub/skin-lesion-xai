import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from dataset import train_loader, val_loader
from model import get_efficientnet, get_vit

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 20
LR = 1e-3

# ============================================
# Class weights for imbalance
# ============================================
class_counts = [4522, 12875, 3323, 867, 2624, 239, 253, 628]
total = sum(class_counts)
weights = torch.tensor([total/c for c in class_counts], dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)

# ============================================
# Training function
# ============================================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss/len(loader), correct/total

# ============================================
# Validation function
# ============================================
def validate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss/len(loader), correct/total, f1

# ============================================
# Train EfficientNet
# ============================================
def train_model(model, model_name, epochs=EPOCHS):
    model = model.to(DEVICE)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_f1 = 0
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[], 'val_f1':[]}
    
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"Device: {DEVICE}")
    print(f"{'='*50}")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc, val_f1 = validate(model, val_loader)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch [{epoch+1:2d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f'data/{model_name}_best.pth')
            print(f"  ✓ Best model saved! F1: {best_f1:.4f}")
    
    return history
if __name__ == '__main__':
    # EfficientNet already done — skip!
    # eff_model = get_efficientnet()
    # eff_history = train_model(eff_model, 'efficientnet', epochs=20)
    # print(f"EfficientNet Best F1: {max(eff_history['val_f1']):.4f}")
    
    # Train ViT only
    vit_model = get_vit()
    vit_history = train_model(vit_model, 'vit', epochs=20)
    print(f"ViT Best F1: {max(vit_history['val_f1']):.4f}")