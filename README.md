# Fall Detection Using MediaPipe and Transformer-Based Deep Learning  

This project focuses on detecting different **human postures and fall events** using **MediaPipe-based pose data** and a **Transformer-based deep learning model**.  
The dataset consists of labeled **human posture classes** such as **falling, standing, sitting, lying down, and more** to train and evaluate the model for **real-time fall detection**.

---

##  Features
✅ **Data Preprocessing** – Normalizes and reshapes **time-series pose data** collected using **MediaPipe**.  
✅ **Transformer-Based Neural Network** – Implements a **custom Transformer model** for **sequential pose classification**.  
✅ **Evaluation Metrics** – Computes **accuracy, precision, recall, F1-score**, and visualizes **confusion matrices and ROC curves**.  
✅ **Real-Time Prediction Capability** – The model can **classify human postures and detect falls** in real-time.  

---

## Project Structure
- 📂 fall-detection-transformer 
- │── 📂 dataset/ # MediaPipe CSV files for pose data │ │── falling_df.csv # Data for falling events │ │── stand_up_df.csv # Data for standing up │ │── ... (more CSV files)
- │── 📂 saved_models/ # Directory for storing trained models │── transformer_model.py # Main script for data preprocessing, model training, and evaluation │──    fall_detection.py # Script for real-time fall detection │── requirements.txt # Dependencies │── README.md # Documentation


---

## 📊 Dataset Overview

The dataset contains **pose keypoints captured using MediaPipe**, representing different human postures.

| Class       | Description                        | Label |
|------------|------------------------------------|-------|
| **Falling**  | Fall event detected               | 0 |
| **Stand Up** | Transition from seated/lying      | 1 |
| **Standing** | Normal standing posture          | 2 |
| **Lie Down** | Moving to a lying position       | 3 |
| **Sit Down** | Sitting posture                  | 4 |
| **Lying**    | Fully lying down                 | 5 |

Each CSV file consists of **sequential pose keypoint data**, reshaped into **3D arrays** for input into the **Transformer model**.

---

## 🚀 Installation and Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Duncan1738/fall-detection-transformer.git
cd fall-detection-transformer

2️⃣ Install Required Dependencies

pip install -r requirements.txt
3️⃣ Run the Training Script

python transformer_model.py
4️⃣ Perform Real-Time Fall Detection (Using Camera)

python fall_detection.py
🔧 Workflow
1️⃣ Data Preprocessing
Loads pose keypoint data from CSV files.
Drops unnecessary columns (midpoint_1, midpoint_2, central_point).
Normalizes data using MinMaxScaler.
Reshapes pose keypoints into time-series sequences for Transformer input.
2️⃣ Model Architecture
Custom Transformer-based deep learning model for sequence classification.
Multi-head self-attention layers, followed by feed-forward layers.
Global average pooling and dense layers for classification.
3️⃣ Training and Evaluation
Splits dataset into training (80%) and testing (20%).
Implements early stopping to prevent overfitting.
Visualizes performance with: ✅ Confusion matrices
✅ ROC curves
✅ Accuracy/loss plots
📈 Example Output
🔹 Classification Report
              precision    recall    f1-score   support
falling          0.95       0.93       0.94       100
stand_up        0.92       0.91       0.91       100
standing        0.90       0.92       0.91       100
lie_down        0.94       0.95       0.94       100
sit_down        0.89       0.90       0.89       100
lie             0.92       0.90       0.91       100
📊 Confusion Matrix
(Replace with your actual confusion matrix image)

📉 Accuracy & Loss Plots
(Replace with training history plot)

📈 ROC Curve for Fall Detection
