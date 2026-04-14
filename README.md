# AI Medical Image Analysis 🧠🩺

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

A deep learning-based AI project that classifies medical images (like X-rays) into **Normal** or **Disease**.  
It’s built to be easy to demo, easy to explain, and strong enough to include in a portfolio.

---

## Project Overview ✨

This project builds a complete pipeline for medical image classification using a **Convolutional Neural Network (CNN)**.  
It trains on labeled datasets and predicts whether an image is **Normal** or **Disease**.

You can run everything in a clean Streamlit dashboard that allows image upload, prediction, and visualization.

---

## Problem Statement 🎯

Medical image analysis is difficult to understand when it’s hidden inside code.  
Recruiters and non-technical users need something they can **see** and **interact with**.

This project provides a simple flow:  
**Upload Image → Get Prediction → Understand Result**

---

## Solution ✅

This system:

- Builds a **CNN model using TensorFlow/Keras**
- Trains on **Normal vs Disease datasets**
- Evaluates using graphs
- Provides a **Flask API**
- Provides a **Streamlit dashboard**

---

## Industry Relevance 🌍

Used in:

- AI-based radiology systems
- Disease detection from X-rays
- Medical imaging research
- Hospital diagnostic tools
- Healthcare AI startups

---

## Tech Stack 🧰

| Category       | Tool/Library       | Why it’s used       |
| -------------- | ------------------ | ------------------- |
| Language       | Python             | Core development    |
| Deep Learning  | TensorFlow/Keras   | CNN model training  |
| Image Handling | OpenCV             | Preprocessing       |
| Visualization  | Matplotlib/Seaborn | Graphs              |
| API            | Flask              | Prediction endpoint |
| Dashboard      | Streamlit          | UI for interaction  |
| Utilities      | NumPy, Joblib      | Data handling       |

---

## Folder Structure (project layout) 📁

```text
AI-Medical-Image-Analysis/
│
├── data/
│   ├── train/
│   │   ├── Normal/
│   │   └── Disease/
│   └── test/
│       ├── Normal/
│       └── Disease/
│
├── src/                         # Core ML logic
│
├── api/
│   └── app.py                   # Flask API (/predict)
│
├── dashboard/
│   └── app.py                   # Streamlit UI
│
├── models/
│   └── cnn_model.h5
│
├── outputs/                     # Graphs & results
├── images/                      # Sample images
│
├── main.py                      # Training script
├── requirements.txt
└── README.md



## Demo Video 🎥

[![Watch Demo](https://img.youtube.com/vi/D6HyNs9RiKI/0.jpg)](https://youtu.be/D6HyNs9RiKI)

👉 Click the image to watch the full demo

## Author 👤

Built by **[Sonia Thakur]**

- GitHub: `https://github.com/Sonia068`
- LinkedIn: `https://www.linkedin.com/in/sonia-thakur-6ab93b349/`


```
