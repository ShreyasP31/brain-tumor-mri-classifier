# 🧠 Brain Tumor MRI Classifier

A deep learning web application that classifies brain MRI scans into four categories with Grad-CAM explainability.

**Live Demo:** https://brain-tumor-mri-classifier-appxiz8iaoqp8mrby2cef8f.streamlit.app/

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 95.62% |
| Macro F1-Score | 0.96 |
| Macro Precision | 0.96 |
| Macro Recall | 0.96 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 1.00 | 0.84 | 0.91 |
| Meningioma | 0.89 | 0.99 | 0.94 |
| No Tumor | 0.96 | 1.00 | 0.98 |
| Pituitary | 0.99 | 1.00 | 0.99 |

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Training Loss
![Training Loss](training_loss.png)

### Sample MRI Images
![Sample Images](sample_images.png)

### U-Net Segmentation Results
![Segmentation Results](segmentation_results.png)

---

## Features

- **4-class classification** — Glioma, Meningioma, Pituitary Tumor, No Tumor
- **Grad-CAM explainability** — heatmap showing which brain regions influenced the prediction
- **Confidence scores** — probability breakdown across all 4 classes
- **Live web app** — deployed on Streamlit Cloud

---

## Model Architecture

- **Backbone:** EfficientNetB3 pretrained on ImageNet
- **Input size:** 300×300 RGB
- **Training:** 30 epochs, GPU (RTX 5060), mixed precision (AMP)
- **Loss:** CrossEntropyLoss with label smoothing (0.1)
- **Optimizer:** Adam (lr=0.0005) with CosineAnnealingLR scheduler
- **Explainability:** Grad-CAM via manual hook implementation

---

## Dataset

**Brain Tumor MRI Dataset** by Masoud Nickparvar (Kaggle)

| Class | Training | Testing | Total |
|-------|----------|---------|-------|
| Glioma | 1,321 | 400 | 1,721 |
| Meningioma | 1,339 | 400 | 1,739 |
| No Tumor | 1,595 | 400 | 1,995 |
| Pituitary | 1,457 | 400 | 1,857 |
| **Total** | **5,712** | **1,600** | **7,312** |

---

## Additional Work — U-Net Tumor Segmentation

A separate U-Net model with EfficientNetB3 encoder was trained for pixel-level tumor segmentation on the LGG MRI Segmentation dataset (Mateusz Buda, Kaggle).

- **Dataset:** 3,929 MRI + mask pairs from 110 patients
- **Dice Score:** 0.9045 on validation set
- **Note:** Segmentation performs best on axial LGG MRI scans matching the training distribution

Training notebook: `brain_tumor_segmentation.ipynb`

---

## Project Structure
```
brain-tumor-ai/
├── app.py                             # Streamlit web application
├── brain_tumor_classifier.ipynb       # Classification training notebook
├── brain_tumor_segmentation.ipynb     # U-Net segmentation training notebook
├── brain_tumor_model.pth              # EfficientNetB3 classifier weights
├── segmentation_model.pth             # U-Net segmentation weights
├── confusion_matrix.png               # Confusion matrix
├── training_loss.png                  # Training loss curve
├── segmentation_loss.png              # Segmentation training loss
├── segmentation_results.png           # Segmentation examples
├── sample_images.png                  # Sample MRI images
└── requirements.txt                   # Dependencies
```

---

## How to Run Locally
```bash
git clone https://github.com/ShreyasP31/brain-tumor-mri-classifier.git
cd brain-tumor-mri-classifier
pip install -r requirements.txt
streamlit run app.py
```

---

## Limitations

- Model trained on one dataset — performance may vary on MRIs from different scanners or protocols (distribution shift)
- Segmentation model trained on LGG axial MRIs only — not validated on other tumor types or views
- For educational purposes only — not validated for clinical diagnosis

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch 2.10 + CUDA 12.8 |
| Classification | EfficientNetB3 |
| Segmentation | U-Net + EfficientNetB3 encoder |
| Explainability | Grad-CAM |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |

---

## Changelog

**v2 — March 2026**
- Upgraded model from EfficientNetB0 to EfficientNetB3
- Added Grad-CAM explainability (manual implementation, no opencv dependency)
- Trained U-Net segmentation model — 0.90 Dice score on LGG dataset
- GPU training with mixed precision (AMP) on RTX 5060
- Improved accuracy from 94.31% to 95.62%

**v1 — March 2026**
- Initial release — EfficientNetB0, 94.31% accuracy
- Basic Streamlit web app

---

*Shreyas P | Biomedical Engineering | MIT Manipal | 2026*