# 🍎 FruitSense — Fruit Ripeness Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)
![License](https://img.shields.io/badge/License-MIT-green)

> Deep learning model to classify fruits as **Fresh**, **Rotten**, or **Unripe**
> using Transfer Learning (DenseNet121, VGG16, ResNet50, MobileNetV2).

---

## 📌 Problem Statement

Detecting fruit ripeness manually is time-consuming and error-prone.
FruitSense automates this using Convolutional Neural Networks trained on
real fruit images across 9 categories.

---

## 🗂️ Dataset

- **Source:** [Kaggle — Fruit Ripeness Dataset](https://www.kaggle.com/datasets/)
- **Classes (9):**

| Fresh | Rotten | Unripe |
|-------|--------|--------|
| Fresh Apple | Rotten Apple | Unripe Apple |
| Fresh Banana | Rotten Banana | Unripe Banana |
| Fresh Orange | Rotten Orange | Unripe Orange |

---

## 🏗️ Model Architecture

```
Input (224×224×3)
      ↓
Pretrained Base (DenseNet121 / VGG16 / ResNet50 / MobileNetV2)
      ↓
GlobalAveragePooling2D
      ↓
BatchNormalization
      ↓
Dense(512, ReLU) → Dropout(0.4)
      ↓
Dense(256, ReLU) → Dropout(0.3)
      ↓
Dense(9, Softmax)
```

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| DenseNet121 (FT) | ~97% | ~97% | ~97% | ~97% |
| ResNet50 | ~95% | ~95% | ~95% | ~95% |
| MobileNetV2 | ~94% | ~94% | ~94% | ~94% |
| VGG16 | ~93% | ~93% | ~93% | ~93% |

> FT = Fine-Tuned (last 50 layers unfrozen)

## 🚀 Run on Kaggle

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **New Notebook**
3. Import notebook from GitHub:
   - File → Import Notebook → GitHub tab
   - Paste: `https://github.com/YOUR_USERNAME/FruitSense`
4. Add dataset:
   - Right panel → **Add Data** → search fruit ripeness dataset
5. Click **Run All**

---

## 💻 Run Locally

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/FruitSense.git
cd FruitSense

# Install dependencies
pip install -r requirements.txt

# Run prediction on single image
python src/predict.py --image path/to/fruit.jpg
```

---

## 📦 Requirements

```
tensorflow>=2.12.0
opencv-python
scikit-learn
matplotlib
seaborn
pandas
numpy
```

# output 

# Model Acuuracy 
<img width="1920" height="1080" alt="Screenshot 2026-04-15 085840" src="https://github.com/user-attachments/assets/68b486d4-be30-46da-8f02-6994a6de0b6a" />

# comparing all models accuracy
<img width="1920" height="1080" alt="Screenshot 2026-04-15 085840" src="https://github.com/user-attachments/assets/0c851072-3b46-4244-91da-4e4ea325546f" />

# predicting the fruit
<img width="191" height="374" alt="image" src="https://github.com/user-attachments/assets/2174f8d7-cad9-48da-a41c-aecc3248f30b" />
