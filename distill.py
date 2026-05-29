import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from dataset import train_loader, val_loader
from model import get_efficientnet, get_vit

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 20
TEMPERATURE = 4.0
ALPHA = 0.7

# Load teacher (ViT)
teacher = get_vit()
teacher.load_state_dict(torch.load('data/vit_best.pth', map_location=DEVICE))
teacher = teacher.to(DEVICE)
teacher.eval()

# Student (EfficientNet)
student = get_efficientnet()
student = student.to(DEVICE)

# Loss functions
ce_loss = nn.CrossEntropyLoss()

def distillation_loss(student_out, teacher_out, labels, T, alpha):
    soft_loss = F.kl_div(
        F.log_softmax(student_out / T, dim=1),
        F.softmax(teacher_out / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    hard_loss = ce_loss(student_out, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, student.parameters()), lr=1e-3
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_f1 = 0
print(f"\n{'='*50}")
print("Knowledge Distillation: ViT → EfficientNet")
print(f"Temperature: {TEMPERATURE}, Alpha: {ALPHA}")
print(f"Device: {DEVICE}")
print(f"{'='*50}\n")

for epoch in range(EPOCHS):
    student.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            teacher_out = teacher(imgs)
        student_out = student(imgs)
        loss = distillation_loss(student_out, teacher_out, labels, TEMPERATURE, ALPHA)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (student_out.argmax(1) == labels).sum().item()
        total += labels.size(0)

    # Validation
    student.eval()
    val_correct, val_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = student(imgs)
            preds = out.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    scheduler.step()

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
          f"Loss: {total_loss/len(train_loader):.4f} "
          f"Train Acc: {correct/total:.4f} | "
          f"Val Acc: {val_correct/val_total:.4f} F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(student.state_dict(), 'data/kd_efficientnet_best.pth')
        print(f"  ✓ Best KD model saved! F1: {best_f1:.4f}")

print(f"\nKnowledge Distillation Complete!")
print(f"Best F1: {best_f1:.4f}")