import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from model import get_efficientnet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

# Load model
model = get_efficientnet()
model.load_state_dict(torch.load('data/efficientnet_best.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Hook for gradients
gradients = []
activations = []

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def forward_hook(module, input, output):
    activations.append(output)

# Register hooks on last conv layer
target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

def generate_gradcam(img_path):
    # Load and preprocess
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    # Forward pass
    gradients.clear()
    activations.clear()
    output = model(input_tensor)
    pred_class = output.argmax().item()
    pred_name = CLASSES[pred_class]
    confidence = F.softmax(output, dim=1)[0][pred_class].item()

    # Backward pass
    model.zero_grad()
    output[0][pred_class].backward()

    # Grad-CAM
    grads = gradients[0].cpu().detach().numpy()[0]
    acts = activations[0].cpu().detach().numpy()[0]
    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img_resized + 0.4 * heatmap).astype(np.uint8)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_resized)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    axes[2].imshow(overlay)
    axes[2].set_title(f'Prediction: {pred_name} ({confidence:.1%})')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig('data/gradcam_result.png')
    plt.show()
    print(f"Prediction: {pred_name} | Confidence: {confidence:.1%}")

# Test with sample image
IMG_DIR = 'data/ISIC_2019_Training_Input/ISIC_2019_Training_Input'
import os
sample_img = os.path.join(IMG_DIR, 'ISIC_0000000.jpg')
generate_gradcam(sample_img)