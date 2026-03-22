import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import matplotlib.cm as cm

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

# ── Class names and descriptions ──────────────────────────────────────
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

descriptions = {
    'glioma': (
        "Gliomas are tumors that arise from glial cells in the brain or spine. "
        "They are the most common type of malignant brain tumor and can vary "
        "widely in aggressiveness. Early detection is critical for treatment planning."
    ),
    'meningioma': (
        "Meningiomas arise from the meninges — the membranes surrounding the brain "
        "and spinal cord. They are usually benign and slow-growing, and are the most "
        "common type of primary brain tumor overall."
    ),
    'notumor': (
        "No tumor was detected in this MRI scan. The brain tissue appears normal. "
        "Always consult a radiologist for a confirmed clinical diagnosis."
    ),
    'pituitary': (
        "Pituitary tumors form in the pituitary gland at the base of the brain. "
        "Most are benign (adenomas) and can affect hormone production. "
        "They are generally treatable with surgery or medication."
    )
}

# ── Load classifier ───────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(1536, 4)
    model.load_state_dict(
        torch.load("brain_tumor_model.pth", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model

# ── Load segmentation model ───────────────────────────────────────────
@st.cache_resource
def load_segmentor():
    model = smp.Unet(
        encoder_name    = "efficientnet-b3",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
    )
    model.load_state_dict(
        torch.load("segmentation_model.pth", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model

classifier = load_classifier()
segmentor  = load_segmentor()

# ── Transforms ────────────────────────────────────────────────────────
clf_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

seg_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Grad-CAM (manual, no opencv) ──────────────────────────────────────
def generate_gradcam(model, input_tensor, predicted_class_idx, original_image):
    model.eval()
    features  = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    handle_f = model.features[-1].register_forward_hook(forward_hook)
    handle_b = model.features[-1].register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    output[0, predicted_class_idx].backward()

    handle_f.remove()
    handle_b.remove()

    weights     = gradients.mean(dim=(2, 3), keepdim=True)
    cam         = (weights * features).sum(dim=1).squeeze()
    cam         = F.relu(cam)
    cam         = cam - cam.min()
    cam         = cam / (cam.max() + 1e-8)
    cam_np      = cam.cpu().numpy()

    cam_pil     = Image.fromarray((cam_np * 255).astype(np.uint8)).resize((300, 300))
    cam_colored = cm.jet(np.array(cam_pil) / 255.0)[:, :, :3]

    original_resized = np.array(original_image.resize((300, 300)), dtype=np.float32) / 255.0
    overlay          = (0.5 * original_resized + 0.5 * cam_colored)
    overlay          = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay

# ── Segmentation ──────────────────────────────────────────────────────
def generate_segmentation(model, image, gradcam_heatmap=None):
    input_tensor = seg_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        mask   = torch.sigmoid(output).squeeze().numpy()

    mask_binary      = (mask > 0.3).astype(np.uint8)
    original_resized = np.array(image.resize((256, 256)), dtype=np.uint8)

    if mask_binary.any() and mask_binary.sum() > 100:
        overlay  = original_resized.copy()
        red_mask = mask_binary > 0
        overlay[red_mask, 0] = 255
        overlay[red_mask, 1] = 0
        overlay[red_mask, 2] = 0
        blended = (0.45 * original_resized + 0.55 * overlay).astype(np.uint8)
        source  = "U-Net segmentation"
    elif gradcam_heatmap is not None:
        cam_resized = np.array(
            Image.fromarray(gradcam_heatmap).resize((256, 256))
        )
        cam_gray  = cam_resized[:, :, 0].astype(np.float32)
        cam_norm  = (cam_gray - cam_gray.min()) / (cam_gray.max() - cam_gray.min() + 1e-8)
        cam_mask  = (cam_norm > 0.5).astype(np.uint8)
        overlay   = original_resized.copy()
        overlay[cam_mask > 0, 0] = 255
        overlay[cam_mask > 0, 1] = 0
        overlay[cam_mask > 0, 2] = 0
        blended = (0.45 * original_resized + 0.55 * overlay).astype(np.uint8)
        source  = "Grad-CAM guided"
    else:
        blended = original_resized
        source  = "No region detected"

    return blended, mask_binary, source

# ── UI ────────────────────────────────────────────────────────────────
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown(
    "Upload a brain MRI image for **classification**, **Grad-CAM explanation**, "
    "and **tumor segmentation**."
)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI scan", width=300)
    st.markdown("---")

    with st.spinner("Analyzing MRI..."):

        clf_tensor = clf_transform(image).unsqueeze(0)
        with torch.no_grad():
            output     = classifier(clf_tensor)
            probs      = F.softmax(output, dim=1)[0]
            confidence, predicted_idx = torch.max(probs, 0)

        predicted_class     = class_names[predicted_idx.item()]
        predicted_class_idx = predicted_idx.item()
        confidence_pct      = confidence.item() * 100

        gradcam_image = generate_gradcam(
            classifier, clf_tensor, predicted_class_idx, image
        )

        seg_overlay, seg_mask, seg_source = generate_segmentation(
            segmentor, image, gradcam_image
        )

    if predicted_class == 'notumor':
        st.success(f"✅ Result: No Tumor Detected ({confidence_pct:.1f}% confidence)")
    else:
        st.warning(f"⚠️ Result: {predicted_class.capitalize()} detected ({confidence_pct:.1f}% confidence)")

    st.info(descriptions[predicted_class])

    st.markdown("#### Confidence Scores")
    for i, cls in enumerate(class_names):
        st.progress(
            float(probs[i].item()),
            text=f"{cls.capitalize()}: {probs[i].item()*100:.1f}%"
        )

    st.markdown("---")
    st.markdown("#### 🔬 Visual Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original MRI**")
        st.caption(" ")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("**🔍 Grad-CAM — Model Attention**")
        st.caption("Red/yellow = high attention regions")
        st.image(gradcam_image, use_container_width=True)

    with col3:
        st.markdown("**🎯 Tumor Segmentation**")
        st.caption("Red overlay = predicted tumor region")
        if predicted_class == 'notumor':
            st.image(image, use_container_width=True)
            st.caption("No tumor region predicted")
        else:
            st.image(seg_overlay, use_container_width=True)
            st.caption(f"Source: {seg_source}")

    st.markdown("---")
    st.caption(
        "⚠️ This tool is for educational purposes only and is not a substitute "
        "for professional medical diagnosis."
    )