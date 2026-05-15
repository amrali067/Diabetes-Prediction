# 🩺 Diabetes Risk Predictor

A Machine Learning web application that predicts diabetes risk based on clinical data, built using **Random Forest** and deployed with **Streamlit**.

---

## 📁 Project Structure

```
📁 Diabetes-Prediction/
├── app.py                          # Streamlit web app
├── requirements.txt                # Required libraries
├── best_model.pkl                  # Trained Random Forest model
├── scaler.pkl                      # StandardScaler
├── EDA_and_Preprocessing.ipynb     # Data exploration & cleaning
└── Model_Training_and_evaluation.ipynb  # Model training & evaluation
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 📊 Models Compared

| Model | Accuracy | Recall | ROC-AUC |
|-------|----------|--------|---------|
| Logistic Regression | 0.708 | 0.556 | 0.830 |
| Random Forest ✅ | 0.864 | 0.889 | 0.933 |
| XGBoost | 0.870 | 0.815 | 0.955 |
| SVM | 0.857 | 0.759 | 0.875 |

> **Random Forest** was selected for deployment due to its highest **Recall (0.889)**, minimizing missed diabetes cases.

---

## 🔬 Features Used

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (μU/mL) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes hereditary score |
| Age | Age in years |

---

## 📦 Dataset

[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — UCI Machine Learning Repository

---

## 👨‍💻 Author

**Eng. Amr Samir** — Biomedical Engineer & Data Scientist &
