# 💳 Fraud Detection AI System

## 👤 Author
**Gameli Samuel Wordui**  
Thrive Africa – Machine Learning & AI Project  
Group 15  

---

## 📌 Project Overview
This project presents a **machine learning-based fraud detection system** designed to identify potentially fraudulent financial transactions.

The system integrates:
- Data preprocessing and feature engineering  
- Machine learning modelling (XGBoost)  
- Interactive deployment using Streamlit  

It enables users to input transaction details and receive **real-time fraud risk predictions** with adjustable sensitivity thresholds.

---

## 🚀 Key Features
- ✅ Real-time fraud prediction  
- 🎯 Adjustable fraud detection threshold  
- 📊 Risk classification (Low, Moderate, High)  
- 🧠 Model explanation interface  
- 📉 Visual probability indicator  
- 💻 Clean and interactive Streamlit dashboard  

---

## 🛠️ Technology Stack

### 🔹 Programming
- Python  

### 🔹 Development Environment
- Google Colab (model training, preprocessing, and experimentation)  
- VS Code (application development and integration)  

### 🔹 Data Processing
- Pandas  
- NumPy  

### 🔹 Machine Learning
- Scikit-learn  
- XGBoost  
- SMOTE (Imbalanced-learn)  

### 🔹 Model Persistence
- Pickle  
- Joblib  

### 🔹 Visualisation
- Matplotlib  
- Seaborn  

### 🔹 Deployment & Version Control
- Streamlit  
- Streamlit Cloud  
- GitHub  
- Git Bash  

---

## 📊 Model Performance

| Metric   | Score |
|----------|------|
| ROC-AUC  | 0.995 |
| F1 Score | 0.969 |
| MCC      | 0.938 |
| Log Loss | 0.095 |

> These results indicate strong classification performance. However, careful validation revealed sensitivity to class imbalance, highlighting the importance of robust evaluation strategies in fraud detection systems.

---

## ⚙️ How to Run the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt

### 2. Run the App
python -m streamlit run app.py

📁 Project Structure
fraud-detection-app/
│── app.py
│── train_model.py
│── preprocess_data.py
│── final_fraud_model.pkl
│── final_fraud_dataset.csv
│── requirements.txt
│── README.md

---

## ⚠️ Disclaimer
This system is designed for educational, research, and decision-support purposes only.  
Predictions are probabilistic and should not be used as the sole basis for financial, legal, or operational decisions.

---

## 🙏 Acknowledgement
This project was developed under the Thrive Africa Machine Learning & AI Programme.

---

## ⭐ Final Remark
This project demonstrates a complete machine learning pipeline—from data preprocessing and modelling to deployment—while critically addressing real-world challenges such as class imbalance, model generalisation, and decision threshold tuning.
