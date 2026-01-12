🧠 Week 3 – Introduction to Machine Learning
📌 Overview

This repository contains my Week 3 practical work focused on the fundamentals of Machine Learning, with an emphasis on regression algorithms.
The tasks cover implementing Linear Regression from scratch, using scikit-learn for multiple regression, understanding polynomial regression and overfitting, and learning model persistence techniques.

All implementations are done in Python and documented clearly for learning, reproducibility, and future reference.

🛠️ Technologies Used

Python 3

NumPy

Matplotlib

Scikit-learn

Pickle & Joblib

JSON

VS Code

Git & GitHub

📂 Project Structure
Week-3-Introduction-to-ML/
│
├── linear_regression_scratch.py
├── multiple_regression.py
├── polynomial_regression.py
├── model_persistence.py
├── load_and_predict.py
│
├── regression_line.png
├── cost_convergence.png
├── actual_vs_predicted.png
├── residuals.png
├── polynomial_models.png
│
├── model.pkl
├── model.joblib
├── weights.json
│
└── README.md

✅ Task 3.1: Simple Linear Regression from Scratch
📄 File: linear_regression_scratch.py
🔹 Description

This script implements Simple Linear Regression without using sklearn, relying purely on NumPy and mathematical principles.

🔹 Key Features

Synthetic dataset generation (y = 2x + 1 + noise)

Custom LinearRegression class

Mean Squared Error (MSE) cost function

Gradient Descent optimization

Manual R² score calculation

Visualization of:

Regression line

Cost function convergence

📈 Outputs

regression_line.png

cost_convergence.png

🎯 Learning Outcome

Deep understanding of how linear regression works internally

Practical exposure to gradient descent and optimization

✅ Task 3.2: Multiple Linear Regression with scikit-learn
📄 File: multiple_regression.py
🔹 Description

This task demonstrates Multiple Linear Regression using a real-world dataset and professional ML tools.

🔹 Key Steps

Loaded the California Housing Dataset

Performed train-test split

Trained LinearRegression model

Evaluated performance using:

MAE

MSE

RMSE

R² Score

Visualized:

Actual vs Predicted values

Residual errors

Printed feature coefficients and intercept

📈 Outputs

actual_vs_predicted.png

residuals.png

🎯 Learning Outcome

Understanding regression with multiple features

Model evaluation and interpretation

✅ Task 3.3: Polynomial Regression & Overfitting
📄 File: polynomial_regression.py
🔹 Description

This task explores model complexity, underfitting, and overfitting using polynomial regression.

🔹 Key Features

Synthetic non-linear dataset

Polynomial degrees tested: 1, 2, 3, 5, 10

Training and testing error comparison

Visualization of all polynomial models on a single graph

Identification of overfitting in high-degree models

📈 Outputs

polynomial_models.png

🎯 Learning Outcome

Bias–variance tradeoff

Effect of model complexity on generalization

✅ Task 3.4: Model Persistence – Saving & Loading Models
📄 Files:

model_persistence.py

load_and_predict.py

🔹 Description

This task demonstrates saving and loading Machine Learning models using different formats.

🔹 Model Saving Formats

Pickle (.pkl) – Python-native serialization

Joblib (.joblib) – Optimized for large NumPy arrays

JSON (.json) – Weights only (manual serialization)

🔹 Comparison Performed

Model loading time

File size comparison

Prediction consistency

🎯 Learning Outcome

Understanding trade-offs between persistence formats

Preparing models for deployment and reuse

📊 Model Persistence Comparison
Format	File Type	Speed	Size	Use Case
Pickle	.pkl	Fast	Medium	Python-only projects
Joblib	.joblib	Very Fast	Large	ML models with NumPy arrays
JSON	.json	Fast	Small	Model weights portability
🚀 How to Run the Project
1️⃣ Install Dependencies
pip install numpy matplotlib scikit-learn

2️⃣ Run Linear Regression from Scratch
python linear_regression_scratch.py

3️⃣ Run Multiple Regression
python multiple_regression.py

4️⃣ Run Polynomial Regression
python polynomial_regression.py

5️⃣ Save & Load Models
python model_persistence.py
python load_and_predict.py

📌 Conclusion

This repository demonstrates my understanding of Machine Learning regression techniques, both from theoretical and practical perspectives. By implementing algorithms from scratch and using industry-standard libraries, I gained a strong foundation that prepares me for more advanced topics such as classification, model tuning, and deployment.

👤 Author

Zeeshan Ali
Machine Learning Student

⭐ Acknowledgements

Scikit-learn documentation

California Housing Dataset

Python open-source community
