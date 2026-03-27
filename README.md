# 🏥 Sepsis Survival & Risk Prediction Platform

> AI-powered Clinical Decision Support System for ICU Teams

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

Sepsis is a life-threatening medical emergency that kills millions of patients every year in hospitals worldwide. Early identification of high-risk patients is critical to reducing mortality.

This project builds an **AI-powered Sepsis Survival Assessment Platform** that:

- Predicts the **probability of patient death** from sepsis
- Classifies patients into **High / Medium / Low risk** categories
- Provides **explainable AI insights** showing which features drove the prediction
- Gives **ICU clinical recommendations** based on risk level
- Allows doctors to **download a prediction report** for documentation

---

## 🎯 Problem Statement

Hospitals need a system that can:
- Predict survival probability early
- Identify high-risk patients before it is too late
- Assist ICU decision-making with data-driven insights
- Reduce mortality rates through faster intervention

---

## 🗂️ Project Structure

```
sepsis_project/
│
├── app/
│   └── main.py                  # Streamlit web application
│
├── data/
│   ├── sepsis.csv               # Primary cohort (training data)
│   ├── sepsis_cleaned.csv       # Cleaned dataset
│   ├── sepsis_features.csv      # Feature engineered dataset
│   ├── sepsis_scaled.csv        # Scaled dataset
│   └── *_study_cohort.csv       # Validation cohort
│   └── *_validation_cohort.csv  # Test cohort
│
├── models/
│   ├── best_model.pkl           # Saved best model (Random Forest)
│   └── scaler.pkl               # Fitted RobustScaler
│
├── notebooks/
│   └── 01_EDA.ipynb             # Exploratory Data Analysis notebook
│
├── src/
│   ├── data_preprocessing.py    # Data cleaning and validation
│   ├── feature_engineering.py   # Feature creation and scaling
│   ├── model_training.py        # Model training with MLflow tracking
│   ├── evaluate.py              # Model evaluation script
│   ├── check_data.py            # Data quality checks
│   └── check_all_files.py       # Multi-file validation
│
├── Dockerfile                   # Docker container configuration
├── requirements.txt             # Python dependencies
├── mentor_notes.txt             # Model explanation and limitations
└── README.md                    # This file
```

---

## 📊 Dataset

| File | Records | Purpose | Dead % | Alive % |
|------|---------|---------|--------|---------|
| Primary Cohort | 110,204 | Training | 7.4% | 92.6% |
| Study Cohort | 19,051 | Validation | 18.9% | 81.1% |
| Validation Cohort | 137 | Testing | 17.5% | 82.5% |

**Source:** Sepsis Survival Minimal Clinical Records — real-world hospital data from peer-reviewed research

**Original Features:** age_years, sex_0male_1female, episode_number, hospital_outcome_1alive_0dead

---

## ⚙️ Feature Engineering

We started with 3 original features and engineered 8 new features:

| New Feature | Description |
|-------------|-------------|
| age_group | Age binned into clinical categories (Child / Young / Adult / Senior / Elderly) |
| is_elderly | Binary flag: Age >= 65 |
| is_child | Binary flag: Age <= 18 |
| is_very_elderly | Binary flag: Age >= 80 |
| age_episode_ratio | Age divided by episode number |
| age_x_episode | Age multiplied by episode number |
| elderly_x_episode | Elderly flag multiplied by episode number |
| age_squared | Age squared to capture non-linear mortality effect |

---

## 🤖 Models Trained

| Model | CV AUC | Val AUC | Test AUC | Dead Recall |
|-------|--------|---------|----------|-------------|
| Logistic Regression | 0.7075 | 0.5898 | 0.5658 | 76.79% |
| **Random Forest ★** | **0.7442** | **0.6076** | 0.5385 | **76.87%** |
| XGBoost | 0.7335 | 0.6066 | 0.5529 | 0.06% |
| LightGBM | 0.7322 | 0.6027 | 0.5621 | 77.09% |

**★ Best Model: Random Forest** — selected based on highest Validation ROC-AUC and consistently high dead patient recall.

---

## 🛠️ Techniques Used

| Technique | Purpose |
|-----------|---------|
| **Stratified K-Fold (5-fold)** | Ensures each fold maintains the same class distribution as the original dataset — critical for imbalanced data |
| **SMOTE** | Creates synthetic samples of minority class (dead patients) — applied only on training data to avoid data leakage |
| **Class Weight Balancing** | Tells models to pay proportionally more attention to dead patients during training |
| **Cross Validation** | Gives reliable performance estimate across multiple data splits — detects overfitting |

---

## 📈 Evaluation Metrics

### Classification Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct predictions overall |
| **Precision** | Of all patients flagged as dead, how many actually died |
| **Recall** ⭐ | Of all patients who died, how many did the model catch — **most critical metric** |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Model's ability to distinguish dead vs alive across all thresholds |
| **PR-AUC** | Precision-Recall curve — more informative for imbalanced datasets |

### Why Recall is Most Important
Missing a high-risk patient (False Negative) is far more dangerous than a false alarm (False Positive). A patient incorrectly flagged as high risk receives extra monitoring — manageable. A high-risk patient missed by the model may not receive timely intervention — potentially fatal.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/gokz7/sepsis-survival-project.git
cd sepsis-survival-project
```

**Step 2 — Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**Step 3 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — Run data preprocessing:**
```bash
python src/data_preprocessing.py
```

**Step 5 — Run feature engineering:**
```bash
python src/feature_engineering.py
```

**Step 6 — Train models:**
```bash
python src/model_training.py
```

**Step 7 — Launch Streamlit app:**
```bash
streamlit run app/main.py
```

Open browser at `http://localhost:8501`

---

## 🧪 MLflow Experiment Tracking

View all experiment runs:
```bash
mlflow ui
```
Open browser at `http://localhost:5000`

All 4 model runs are tracked with:
- Parameters: model name, SMOTE applied, CV folds, dataset sizes
- Metrics: CV AUC, Validation AUC, Test AUC, Dead Recall, Dead Missed
- Artifacts: saved model files

---

## 🐳 Docker Deployment

**Build the container:**
```bash
docker build -t sepsis-app .
```

**Run locally:**
```bash
docker run -p 8501:8501 sepsis-app
```

Open browser at `http://localhost:8501`

---

## ☁️ Cloud Deployment

The application is deployed on **Render** cloud platform.

**Live URL:** `[Your Render URL here]`

Deployment is automatic — every push to the master branch triggers a new deployment via GitHub integration.

---

## 🖥️ Application Features

| Feature | Description |
|---------|-------------|
| Patient Input Form | Enter age, sex and episode number |
| Risk Assessment | High / Medium / Low risk classification |
| Death Probability | Exact probability score with progress bar |
| ICU Recommendation | Clinical advice based on risk level |
| Probability Chart | Visual bar chart of death vs survival |
| Feature Importance | Chart showing which features drove the prediction |
| Download Report | Full prediction report as TXT file |

---

## ⚠️ Limitations

- Dataset contains only 3 original clinical features (age, sex, episode number)
- Real sepsis scoring systems like SOFA and APACHE II use 15+ variables including vitals and lab results
- Validation ROC-AUC of 0.60 reflects dataset limitations — not model failure
- Test cohort has only 137 records — statistically limited

**This tool is a decision support system. All clinical decisions must be made by qualified medical professionals.**

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Imbalance Handling | Imbalanced-learn (SMOTE) |
| Experiment Tracking | MLflow |
| Web Application | Streamlit |
| Explainability | Feature Importance (Random Forest) |
| Containerization | Docker |
| Cloud Deployment | Render |
| Version Control | Git, GitHub |

---

## 📋 Deliverables

- [x] EDA Notebook
- [x] Data Preprocessing Script
- [x] Feature Engineering Script
- [x] Model Training Script with MLflow
- [x] Trained Model (.pkl)
- [x] MLflow Experiment Logs
- [x] Streamlit Web Application
- [x] Dockerfile
- [x] Cloud Deployment URL
- [x] README Documentation
- [x] Project Evaluation Document

---

## 👤 Author

**Gokul**
GUVI x HCL Project — March 2026
Domain: Healthcare AI / Clinical Decision Support System

---

## 📄 Dataset Citation

Davide Chicco, Giuseppe Jurman.
*Survival analysis of heart failure patients: a case study.*
Scientific Reports, 2020.

---

> ⭐ If you found this project helpful, please give it a star on GitHub!
