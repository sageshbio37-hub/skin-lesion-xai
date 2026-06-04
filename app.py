import anthropic
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import get_efficientnet
import gdown
import os

# Download model weights if not present
os.makedirs('data', exist_ok=True)
if not os.path.exists('data/kd_efficientnet_best.pth'):
    gdown.download('https://drive.google.com/uc?id=1hoeplrX5VCMkRhkb5zczr3GHuOTYyjCg', 'data/kd_efficientnet_best.pth', quiet=False)
if not os.path.exists('data/vit_best.pth'):
    gdown.download('https://drive.google.com/uc?id=1vWTc1l3zzSHZxlBD_qK_cgW8y_Duar4T', 'data/vit_best.pth', quiet=False)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Professional Dark Blue Lab Theme
st.markdown("""
<style>
.stApp {
    background: 
        linear-gradient(135deg, rgba(10,14,39,0.85) 0%, rgba(13,27,75,0.85) 50%, rgba(10,22,40,0.85) 100%),
        url('https://images.unsplash.com/photo-1576086213369-97a306d36557?w=1920&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #e0e6ff;
}
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image: 
        radial-gradient(2px 2px at 10% 15%, #fff 0%, transparent 100%),
        radial-gradient(2px 2px at 25% 40%, #aad4ff 0%, transparent 100%),
        radial-gradient(1px 1px at 40% 10%, #fff 0%, transparent 100%),
        radial-gradient(2px 2px at 55% 60%, #fff 0%, transparent 100%),
        radial-gradient(1px 1px at 70% 25%, #aad4ff 0%, transparent 100%),
        radial-gradient(2px 2px at 80% 75%, #fff 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 45%, #fff 0%, transparent 100%),
        radial-gradient(2px 2px at 15% 70%, #aad4ff 0%, transparent 100%),
        radial-gradient(1px 1px at 35% 85%, #fff 0%, transparent 100%),
        radial-gradient(2px 2px at 60% 90%, #fff 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
    animation: twinkle 4s infinite alternate;
}
@keyframes twinkle {
    0% { opacity: 0.5; }
    100% { opacity: 1; }
}
h1, h2, h3 { color: #7eb3ff !important; text-shadow: 0 0 20px rgba(126,179,255,0.5); }
div[data-testid="metric-container"] {
    background: rgba(13,27,75,0.7) !important;
    border: 1px solid #2a4db5 !important;
    border-radius: 12px !important;
    padding: 15px !important;
}
div[data-testid="stFileUploader"] {
    background: rgba(13,27,75,0.5) !important;
    border: 2px dashed #2a4db5 !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] {
    background: rgba(13,27,75,0.5) !important;
    border: 1px solid #2a4db5 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)
CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_NAMES = {
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevus (Mole)',
    'BCC': 'Basal Cell Carcinoma',
    'AK': 'Actinic Keratosis',
    'BKL': 'Benign Keratosis',
    'DF': 'Dermatofibroma',
    'VASC': 'Vascular Lesion',
    'SCC': 'Squamous Cell Carcinoma'
}
CLASS_DESC = {
    'MEL': 'Most dangerous skin cancer. Irregular borders, multiple colors. Immediate medical attention required.',
    'NV': 'Common benign mole. Symmetric, uniform color. Usually harmless but monitor for changes.',
    'BCC': 'Most common skin cancer. Grows slowly, rarely spreads. Early treatment is highly effective.',
    'AK': 'Precancerous lesion caused by sun damage. Can develop into SCC if untreated.',
    'BKL': 'Non-cancerous skin growth. Common in older adults. No treatment usually needed.',
    'DF': 'Benign fibrous nodule. Common on legs. Harmless, no treatment needed.',
    'VASC': 'Benign blood vessel abnormality. Usually harmless. Cosmetic treatment available.',
    'SCC': 'Second most common skin cancer. Can spread if untreated. Early detection is key.'
}
DANGEROUS = ['MEL', 'BCC', 'SCC', 'AK']
RISK_LEVEL = {
    'MEL': '🔴 High Risk',
    'BCC': '🟠 Medium-High Risk',
    'SCC': '🟠 Medium-High Risk',
    'AK': '🟡 Medium Risk',
    'NV': '🟢 Low Risk',
    'BKL': '🟢 Low Risk',
    'DF': '🟢 Low Risk',
    'VASC': '🟢 Low Risk'
}

@st.cache_resource
def load_model():
    model = get_efficientnet()
    model.load_state_dict(torch.load('data/kd_efficientnet_best.pth', map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

gradients = []
activations = []

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def forward_hook(module, input, output):
    activations.append(output)

def generate_gradcam(model, img_tensor, img_display):
    gradients.clear()
    activations.clear()
    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)
    output = model(img_tensor)
    pred_class = output.argmax().item()
    model.zero_grad()
    output[0][pred_class].backward()
    grads = gradients[0].cpu().detach().numpy()[0]
    acts = activations[0].cpu().detach().numpy()[0]
    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img_display + 0.4 * heatmap).astype(np.uint8)
    h1.remove()
    h2.remove()
    return heatmap, overlay, pred_class

# Page config
st.set_page_config(page_title="DermaXAI", page_icon="🔬", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 3rem; margin: 0; }
    .main-header p { font-size: 1.1rem; opacity: 0.85; margin: 0.5rem 0 0; }
    .info-card {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        font-size: 0.85rem;
        margin-top: 2rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 DermaXAI</h1>
    <p>Explainable AI-Powered Skin Lesion Classification | Knowledge Distilled EfficientNet + Grad-CAM</p>
    <p style="font-size:0.9rem; opacity:0.7;">Trained on ISIC 2019 | 25,333 images | 8 Classes | F1: 0.7574</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.error("⚠️ **Medical Disclaimer:** DermaXAI is a research tool only. It does NOT replace professional medical diagnosis. Always consult a certified dermatologist for medical advice.")

# About section
with st.expander("ℹ️ About DermaXAI — How it works?"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🧠 AI Model
        - **Knowledge Distilled EfficientNet-B0**
        - Teacher: Vision Transformer (ViT-B16)
        - Student: EfficientNet-B0 (4M params)
        - 21x smaller than ViT, higher accuracy!
        """)
    with col2:
        st.markdown("""
        ### 🗺️ Grad-CAM XAI
        - Visualizes **where** the AI looks
        - Red = High attention area
        - Blue = Low attention area
        - Helps doctors **trust** AI decisions
        """)
    with col3:
        st.markdown("""
        ### 📊 Performance
        - Accuracy: **75.67%**
        - F1 Score: **0.7574**
        - Precision: **0.7621**
        - Dataset: ISIC 2019
        """)

st.divider()

# Skin lesion guide
with st.expander("📚 Skin Lesion Types Guide"):
    cols = st.columns(4)
    for i, (cls, name) in enumerate(CLASS_NAMES.items()):
        with cols[i % 4]:
            risk = RISK_LEVEL[cls]
            st.markdown(f"""
            **{name}** ({cls})
            {risk}
            _{CLASS_DESC[cls]}_
            """)

st.divider()

# Upload section
st.markdown("## 📤 Upload Dermoscopy Image")
st.markdown("Upload a skin lesion image for AI analysis. Best results with dermoscopy images.")

uploaded = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'],
                             help="Supported formats: JPG, JPEG, PNG")

if uploaded:
    model = load_model()
    img_resized = np.array(Image.open(uploaded).convert('RGB').resize((224, 224)))

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    img_pil = Image.open(uploaded).convert('RGB')
    img_pil = img_pil.resize((224, 224))
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)


    with torch.no_grad():
        output_prob = model(img_tensor.detach())
        probs = F.softmax(output_prob, dim=1)[0]

    heatmap, overlay, pred_class = generate_gradcam(model, img_tensor, img_resized)

    pred_name = CLASSES[pred_class]
    confidence = probs[pred_class].item()
    is_dangerous = pred_name in DANGEROUS

    st.divider()
    st.markdown("## 🔍 Analysis Results")

    # Result alert
    if is_dangerous:
        st.error(f"⚠️ **{CLASS_NAMES[pred_name]}** detected — {RISK_LEVEL[pred_name]} — Please consult a dermatologist immediately!")
    else:
        st.success(f"✅ **{CLASS_NAMES[pred_name]}** detected — {RISK_LEVEL[pred_name]} — Likely benign, but monitor for changes.")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prediction", CLASS_NAMES[pred_name])
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
    with col3:
        st.metric("Risk Level", RISK_LEVEL[pred_name])
    with col4:
        st.metric("Model", "KD-EfficientNet")

    st.divider()

    # Images
    st.markdown("### 🖼️ Visual Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📸 Original Image**")
        st.image(img_resized, use_column_width=True)
        st.caption("Input dermoscopy image")
    with col2:
        st.markdown("**🌡️ Grad-CAM Heatmap**")
        st.image(heatmap, use_column_width=True)
        st.caption("🔴 Red = AI focus area | 🔵 Blue = Low attention")
    with col3:
        st.markdown("**🔬 AI Attention Overlay**")
        st.image(overlay, use_column_width=True)
        st.caption("Heatmap overlaid on original image")

    st.divider()

    # Class probabilities
    st.markdown("### 📊 All Class Probabilities")
    prob_col1, prob_col2 = st.columns(2)
    sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    for i, (idx, prob) in enumerate(sorted_probs):
        cls = CLASSES[idx]
        col = prob_col1 if i < 4 else prob_col2
        with col:
            color = "🔴" if cls in DANGEROUS else "🟢"
            st.progress(float(prob), text=f"{color} {CLASS_NAMES[cls]}: {prob:.1%}")

    st.divider()

    # Clinical info
    st.markdown("### 🏥 Clinical Information")
    st.info(f"""
    **Detected:** {CLASS_NAMES[pred_name]} ({pred_name})
    
    **Description:** {CLASS_DESC[pred_name]}
    
    **Risk Level:** {RISK_LEVEL[pred_name]}
    
    **Recommended Action:** {'⚠️ Seek immediate medical attention from a certified dermatologist.' if is_dangerous else '✅ Monitor the lesion regularly. Consult a doctor if you notice any changes in size, color, or shape.'}
    """)

else:
    # How to use guide
    st.markdown("### 🚀 How to use DermaXAI?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### Step 1️⃣
        **Upload Image**
        Upload a clear dermoscopy or skin lesion image in JPG/PNG format
        """)
    with col2:
        st.markdown("""
        #### Step 2️⃣
        **AI Analysis**
        Our KD-EfficientNet model analyzes the image and generates Grad-CAM visualization
        """)
    with col3:
        st.markdown("""
        #### Step 3️⃣
        **Review Results**
        Check prediction, confidence score, heatmap and clinical information
        """)

# Footer
st.markdown("""
<div class="footer">
    🔬 DermaXAI | Knowledge Distillation from ViT to EfficientNet with Grad-CAM Explainability<br>
    Sagesh S & Abhay Srinivas Y.S | Arunai Engineering College | ISIC 2019 Dataset<br>
    ⚠️ For Research Purposes Only | Not for Clinical Use
</div>
""", unsafe_allow_html=True)
# AI Medical Analysis
st.divider()
st.markdown("## 🤖 AI Medical Analysis")

if st.button("Generate AI Medical Report"):
    with st.spinner("Analyzing..."):
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        
        prompt = f"""
        Classification Results:
        - Predicted Class: {CLASS_NAMES[CLASSES[pred_class]]}
        - Confidence: {confidence*100:.1f}%
        - Risk Level: {RISK_LEVEL[pred_name]}
        - Model: KD-EfficientNet
        
        All class probabilities:
        {chr(10).join([f"- {CLASS_NAMES[CLASSES[i]]}: {probs[i].item()*100:.1f}%" for i in range(len(CLASSES))])}
        
        Please provide a structured medical analysis.
        """
        
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system="""You are an expert AI medical writer specializing in digital dermatology. 
            Interpret classification outputs from DermaXAI system.
            Generate structured summary with Findings, Explainability, and Differential Context.
            ALWAYS start and end with: This is for research/educational purposes only and does not replace a certified dermatologist's clinical judgment.""",
            messages=[{"role": "user", "content": prompt}]
        )
        
        st.markdown(message.content[0].text)