# 😃 Emotion Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![License](https://img.shields.io/badge/License-MIT-red)

## 📌 Project Description

Emotion Detection is a **Deep Learning based Computer Vision project** that detects human emotions from facial expressions using a webcam.

The system uses **Convolutional Neural Networks (CNN)** to analyze facial features and classify emotions in real-time.

This project demonstrates how **Artificial Intelligence, Deep Learning, and Computer Vision** work together to understand human emotions.

---

## 🚀 Features

- 🎥 Real-time emotion detection using webcam
- 🧠 Deep Learning CNN model
- 😀 Detect multiple emotions
- 🖥️ Easy to run in VS Code
- ⚡ Fast predictions with TensorFlow/Keras
- 📊 Facial expression recognition

---

## 😊 Emotions Detected

The model can detect the following emotions:

| Emotion |
|-------|
| Happy 😀 |
| Sad 😢 |
| Angry 😠 |
| Surprise 😲 |
| Fear 😨 |
| Neutral 😐 |
| Disgust 🤢 |

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Project Structure

```
emotion-detection-project
│
├── dataset
│   ├── train
│   └── test
│
├── model
│   └── emotion_model.hdf5
│
├── emotion_detector.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/emotion-detection.git
```

### Step 2: Open Project Folder

```bash
cd emotion-detection
```

### Step 3: Install Libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Run emotion detection using webcam:

```bash
python emotion_detector.py
```

Your webcam will open and start detecting emotions.

---

## 🧠 Model Architecture

The system works in the following steps:

1️⃣ Capture image from webcam  
2️⃣ Detect face using OpenCV  
3️⃣ Preprocess image (resize + grayscale)  
4️⃣ Pass image to CNN model  
5️⃣ Predict emotion  
6️⃣ Display emotion on screen  

---

## 📊 Dataset

The model is trained using the **FER2013 Facial Emotion Dataset**.

Dataset contains **35,000+ labeled facial images** with different emotions.

---

## 📸 Sample Output

```
Face Detected → Emotion: sad>
<img width="569" height="567" alt="Screenshot 2026-04-10 143653" src="https://github.com/user-attachments/assets/997d1a14-240e-4618-90b0-e08939105dfa" />




The detected emotion appears above the face in real-time.

---

## 🔮 Future Improvements

- Improve accuracy using **Transfer Learning**
- Add **voice emotion detection**
- Deploy project using **Flask Web App**
- Build **mobile application**

---

## 👨‍💻 Author

**Madhav Shinde**

Data Science | AI | Machine Learning

---

## ⭐ Support

If you like this project, please ⭐ **star the repository**.
