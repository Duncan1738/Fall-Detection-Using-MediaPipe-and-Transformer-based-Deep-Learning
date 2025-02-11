# Fall-Detection-Using-MediaPipe-and-Transformer-based-Deep-Learning

📉 Fall Detection Using MediaPipe and Transformer-based Deep Learning
This project focuses on detecting different human postures and falls using MediaPipe-based data and a Transformer-based deep learning model. The dataset consists of several human posture labels such as falling, standing, sitting, lying down, and more, which are used to train and evaluate the model for real-time fall detection.

🌟 Features
Data Preprocessing: Normalization and reshaping of time-series pose data collected using MediaPipe.
Transformer-based Neural Network: Implements a custom Transformer model for sequential data classification.
Evaluation Metrics: Calculates accuracy, precision, recall, F1-score, and visualizes confusion matrices and ROC curves.
Real-time Prediction Capability: Model can classify and detect fall events based on pose data.
📂 Project Structure
bash
Copy
Edit
.
├── dataset/                      # MediaPipe CSV files for pose data
│   ├── falling_df.csv
│   ├── stand_up_df.csv
│   └── ... (more CSV files)
├── transformer_model.py           # Main script for data preprocessing, model building, and evaluation
├── saved_models/                  # Directory for saved trained models
└── README.md                      # Documentation
📊 Dataset Overview
The dataset contains pose keypoints captured using MediaPipe, representing various postures:

Class	Description	Label
Falling	Falling event detected	0
Stand Up	Standing up from a seated/lying position	1
Standing	Standing posture	2
Lie Down	Transition to lying down	3
Sit Down	Sitting posture	4
Lie	Lying down	5
Each CSV file consists of sequential pose keypoint data, which is reshaped into 3D arrays for input into the Transformer model.

🚀 Installation and Setup
1. Clone the Repository:
bash
Copy
Edit
git clone https://github.com/Duncan1738/fall-detection-transformer.git
cd fall-detection-transformer
2. Install Required Libraries:
bash
Copy
Edit
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
3. Run the Script:
bash
Copy
Edit
python transformer_model.py
🔧 Workflow
Data Preprocessing:

Loads pose data from CSV files.
Drops unnecessary columns (midpoint_1, midpoint_2, central_point).
Normalizes the data using MinMaxScaler.
Reshapes the data into time-series batches for input to the Transformer model.
Model Architecture:

Uses a custom Transformer-based architecture for sequence classification.
Multi-head self-attention layers are followed by feed-forward layers.
Applies global average pooling and dense layers for classification.
Training and Evaluation:

Splits the dataset into training (80%) and testing (20%) sets.
Implements early stopping with validation monitoring.
Visualizes model performance using confusion matrices, ROC curves, and accuracy/loss plots.
📈 Example Output
Classification Report:
markdown
Copy
Edit
              precision    recall  f1-score   support
   falling       0.95      0.93      0.94       100
   stand_up     0.92      0.91      0.91       100
   standing     0.90      0.92      0.91       100
   lie_down     0.94      0.95      0.94       100
   sit_down     0.89      0.90      0.89       100
   lie         0.92      0.90      0.91       100
Confusion Matrix:
(Replace with your actual confusion matrix image)

📊 Visualization Examples
Accuracy & Loss Plots
Confusion Matrix
ROC Curve for Fall Detection
✨ Future Improvements
Real-time Pose Detection: Integrate with a camera feed using MediaPipe to detect falls in real time.
Model Optimization: Experiment with different Transformer architectures for better accuracy and reduced computation time.
Multi-class Prediction on Imbalanced Data: Apply advanced sampling techniques for imbalanced data handling.
📜 License
This project is licensed under the MIT License.

📧 Contact
For questions or suggestions, feel free to reach out:
Duncan Kibet
GitHub Profile
