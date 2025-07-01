# 🧠 Age Prediction from Facial Images

This project predicts the age of a person from their image using deep learning. It uses facial detection and a convolutional neural network (CNN) for regression to predict age from static images or webcam feed.

---

## 🚀 Tech Stack

- **Language:** Python 3.x  
- **Libraries:** OpenCV, dlib, NumPy, Matplotlib, PyTorch, Torchvision  
- **Deep Learning:** Custom CNN or pre-trained models  
- **Tools:** Jupyter Notebook, Docker (optional), TensorBoard (optional)  
- **Deployment:** CLI/Notebook-based execution  

---

## 📁 Project Structure

age-prediction/
├── data/ # Image data and CSVs
├── src/ # All source code
│ ├── model.py # CNN model for age prediction
│ ├── train.py # Model training script
│ ├── infer.py # Inference on new images
│ └── detect.py # Webcam demo + detection
├── architecture.png # System architecture diagram
├── requirements.txt # Required dependencies
├── Dockerfile # For containerization (optional)
└── README.md # Project documentation

---

## 🧠 System Architecture

1. **Image Input:** Either static image or webcam.  
2. **Face Detection:** Performed using OpenCV or dlib.  
3. **Preprocessing:** Resize, crop, normalize.  
4. **Age Prediction:** Forward pass through CNN model.  
5. **Result Output:** Age printed, logged, or overlayed on image/webcam feed.  

---

## 🧪 Model Workflow

**Training:**
```bash
python src/train.py --data_csv data/train.csv --epochs 50 --batch 32

