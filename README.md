# 🧠 Brain Tumor Detection System

A deep learning-based web application for detecting brain tumors from MRI images using a hybrid **CNN + MobileNetV2 ensemble model**. Built with **Streamlit**, this project provides an interactive interface for uploading MRI scans and receiving predictions with confidence scores and visualizations.

---

## 🚀 Live Demo

🔗 **View the App:**
👉 https://brain-tumor-detection-app1.streamlit.app/

---

## 🚀 Features

* 📤 Upload MRI brain scan images
* 🧠 Predict tumor type using deep learning
* 📊 Confidence scores with charts (Bar + Pie)
* 📋 Medical interpretation of results
* 📥 Downloadable analysis report
* ⚡ Fast and responsive UI (Streamlit)

---

## 🏥 Tumor Classes

The model classifies MRI scans into the following categories:

* **Glioma**
* **Meningioma**
* **Pituitary Tumor**
* **No Tumor**

---

## 🧠 Model Architecture

This project uses an **ensemble learning approach** combining:

* Custom **Convolutional Neural Network (CNN)**
* **MobileNetV2** (Transfer Learning)

### Why this approach?

* CNN captures spatial features
* MobileNetV2 improves generalization
* Ensemble improves overall accuracy (~95%)

---

## ⚙️ Tech Stack

* Python 🐍
* TensorFlow / Keras
* Streamlit
* NumPy
* Plotly
* PIL

---

## 🖼️ Image Preprocessing

Image preprocessing ensures consistency and improves model performance.

### Steps:

1. Convert image to RGB
2. Resize to **224 × 224**
3. Normalize pixel values (0–1)
4. Expand dimensions

```python
img = image.convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
```

---

## 🧪 Model Training & Development

* Developed using **Python in Google Colab**
* Used GPU acceleration for faster training
* Combined:

  * Custom CNN
  * MobileNetV2 (Transfer Learning)

### Dataset

* MRI brain scan dataset (multi-class)
* Data gathered, cleaned, and organized into:

  * Glioma
  * Meningioma
  * Pituitary
  * No Tumor

### Data Preparation

* Data cleaning and labeling
* Resizing and normalization
* Train-validation split
* Data augmentation

---

## 📈 Results

* ~95% validation accuracy
* Improved robustness using ensemble learning

---

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## ⚠️ Limitations

* May not perform well on:

  * Non-MRI images
  * Low-quality scans
  * Random internet images

* Accuracy depends on training data similarity

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It is **NOT a medical diagnostic tool**.
Always consult a medical professional for diagnosis.

---

## 🙌 Future Improvements

* Grad-CAM visualization
* Better preprocessing
* Larger dataset
* Improved generalization

---




