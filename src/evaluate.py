import pandas as pd                                    # for loading data
from sklearn.metrics import (confusion_matrix,         # for confusion matrix
                             classification_report)    # for detailed metrics per class
import joblib                                          # for loading saved model

# ── Load data and model ───────────────────────────────────────────────────────

df = pd.read_csv('data/sepsis_scaled.csv')             # loading scaled dataset
target = 'hospital_outcome_1alive_0dead'               # target column name
X = df.drop(columns=[target])                          # features
y = df[target]                                         # actual labels

model = joblib.load('models/best_model.pkl')           # loading saved best model
y_pred = model.predict(X)                              # making predictions

# ── Confusion Matrix ──────────────────────────────────────────────────────────

cm = confusion_matrix(y, y_pred)                       # calculating confusion matrix

print("=== Confusion Matrix ===")
print(f"                  Predicted Dead    Predicted Alive")
print(f"Actual Dead   :   {cm[0][0]:>10}        {cm[0][1]:>10}")   # true dead vs predicted
print(f"Actual Alive  :   {cm[1][0]:>10}        {cm[1][1]:>10}")   # true alive vs predicted

print("\n=== What This Means ===")
print(f"Correctly identified Dead  patients: {cm[0][0]:,}")        # true negatives
print(f"Missed Dead patients (dangerous!)  : {cm[0][1]:,}")        # false negatives
print(f"Correctly identified Alive patients: {cm[1][1]:,}")        # true positives
print(f"False alarms (predicted dead wrong): {cm[1][0]:,}")        # false positives

# ── Classification Report ─────────────────────────────────────────────────────

print("\n=== Classification Report ===")
print(classification_report(y, y_pred,
                            target_names=['Dead(0)', 'Alive(1)']))  # detailed report per class