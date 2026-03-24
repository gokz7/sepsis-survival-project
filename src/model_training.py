# ── All Imports Together at Top ───────────────────────────────────────────────
import pandas as pd                                    # for loading data
import numpy as np                                     # for numerical operations
import joblib                                          # for saving models
import warnings                                        # suppress warnings
import os                                              # folder operations
warnings.filterwarnings('ignore')                      # hide warnings

from sklearn.model_selection import StratifiedKFold, cross_val_score   # cross validation
from sklearn.linear_model import LogisticRegression    # logistic regression
from sklearn.ensemble import RandomForestClassifier    # random forest
from sklearn.preprocessing import RobustScaler         # feature scaling
from sklearn.metrics import (confusion_matrix,         # confusion matrix
                             classification_report,    # detailed metrics
                             roc_auc_score)            # roc-auc score
from xgboost import XGBClassifier                      # xgboost
from lightgbm import LGBMClassifier                    # lightgbm
from imblearn.over_sampling import SMOTE               # smote for imbalance
import mlflow                                          # experiment tracking
import mlflow.sklearn                                  # logging models

# ── Step 1: Load All 3 Files ──────────────────────────────────────────────────

print("=== Loading All 3 Datasets ===")
train_df = pd.read_csv('data/sepsis.csv')
val_df   = pd.read_csv('data/s41598-020-73558-3_sepsis_survival_study_cohort.csv')
test_df  = pd.read_csv('data/s41598-020-73558-3_sepsis_survival_validation_cohort.csv')

print(f"Train      : {train_df.shape}")
print(f"Validation : {val_df.shape}")
print(f"Test       : {test_df.shape}")

# ── Step 2: Feature Engineering Function ─────────────────────────────────────

def engineer_features(df):
    df = df.copy()                                     # copy to avoid modifying original

    df['age_group'] = pd.cut(df['age_years'],
                              bins=[0,18,40,60,80,100],
                              labels=[0,1,2,3,4],
                              include_lowest=True).astype(int)

    df['is_elderly']        = (df['age_years'] >= 65).astype(int)
    df['is_child']          = (df['age_years'] <= 18).astype(int)
    df['is_very_elderly']   = (df['age_years'] >= 80).astype(int)
    df['age_episode_ratio'] = df['age_years'] / df['episode_number']
    df['age_x_episode']     = df['age_years'] * df['episode_number']
    df['elderly_x_episode'] = df['is_elderly'] * df['episode_number']
    df['age_squared']       = df['age_years'] ** 2

    return df

train_df = engineer_features(train_df)                 # applying to train
val_df   = engineer_features(val_df)                   # applying to validation
test_df  = engineer_features(test_df)                  # applying to test

print(f"\n✅ Feature engineering applied to all 3 datasets")
print(f"Features after engineering: {train_df.shape[1]-1}")

# ── Step 3: Separate Features and Target ─────────────────────────────────────

target = 'hospital_outcome_1alive_0dead'

X_train = train_df.drop(columns=[target])              # train features
y_train = train_df[target]                             # train target

X_val   = val_df.drop(columns=[target])                # validation features
y_val   = val_df[target]                               # validation target

X_test  = test_df.drop(columns=[target])               # test features
y_test  = test_df[target]                              # test target

# ── Step 4: Scale Features ────────────────────────────────────────────────────

scaler         = RobustScaler()                        # initialise scaler
X_train_scaled = scaler.fit_transform(X_train)         # fit only on train then transform
X_val_scaled   = scaler.transform(X_val)               # transform val using train scaler
X_test_scaled  = scaler.transform(X_test)              # transform test using train scaler

os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')               # saving scaler for streamlit app
print("✅ Scaler fitted on train and saved")

# ── Step 5: SMOTE Only on Training Data ──────────────────────────────────────

print(f"\n=== Applying SMOTE on Training Data Only ===")
print(f"Before - Dead: {(y_train==0).sum():,}  Alive: {(y_train==1).sum():,}")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"After  - Dead: {(y_train_smote==0).sum():,}  Alive: {(y_train_smote==1).sum():,}")
print("Validation and Test data untouched ✅")

# ── Step 6: Define Models ─────────────────────────────────────────────────────

n_dead  = (y_train==0).sum()
n_alive = (y_train==1).sum()

models = {
    'Logistic_Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'),

    'Random_Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'),

    'XGBoost': XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        scale_pos_weight=n_alive/n_dead),

    'LightGBM': LGBMClassifier(
        n_estimators=100,
        random_state=42,
        verbose=-1,
        class_weight='balanced')
}

# ── Step 7: Stratified K-Fold ─────────────────────────────────────────────────

skf = StratifiedKFold(n_splits=5,
                       shuffle=True,
                       random_state=42)

# ── Step 8: MLflow Setup ──────────────────────────────────────────────────────

mlflow.set_experiment("Sepsis_Survival_3File_Split")
print("\n=== Starting MLflow Experiment Tracking ===")

# ── Step 9: Train and Evaluate ────────────────────────────────────────────────

results        = {}
best_auc       = 0
best_model     = None
best_model_name = None

for name, model in models.items():
    print(f"\n--- Training: {name} ---")

    with mlflow.start_run(run_name=name):

        # train on smote balanced data
        model.fit(X_train_smote, y_train_smote)

        # validation performance
        y_val_pred  = model.predict(X_val_scaled)
        y_val_prob  = model.predict_proba(X_val_scaled)[:,1]
        val_auc     = roc_auc_score(y_val, y_val_prob)
        val_cm      = confusion_matrix(y_val, y_val_pred)
        val_dead_recall = val_cm[0][0] / (val_cm[0][0] + val_cm[0][1])

        # test performance
        y_test_pred  = model.predict(X_test_scaled)
        y_test_prob  = model.predict_proba(X_test_scaled)[:,1]
        test_auc     = roc_auc_score(y_test, y_test_prob)
        test_cm      = confusion_matrix(y_test, y_test_pred)
        test_dead_recall = test_cm[0][0] / (test_cm[0][0] + test_cm[0][1])

        # cross validation on smote train data
        cv_auc = cross_val_score(model,
                                  X_train_smote,
                                  y_train_smote,
                                  cv=skf,
                                  scoring='roc_auc').mean()

        # log to mlflow
        mlflow.log_param("model_name", name)
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_metric("cv_roc_auc", cv_auc)
        mlflow.log_metric("val_roc_auc", val_auc)
        mlflow.log_metric("test_roc_auc", test_auc)
        mlflow.log_metric("val_dead_recall", val_dead_recall)
        mlflow.log_metric("test_dead_recall", test_dead_recall)
        mlflow.log_metric("val_dead_missed", int(val_cm[0][1]))
        mlflow.log_metric("test_dead_missed", int(test_cm[0][1]))
        mlflow.sklearn.log_model(model, name)

        results[name] = {
            'cv_auc'           : cv_auc,
            'val_auc'          : val_auc,
            'test_auc'         : test_auc,
            'val_dead_recall'  : val_dead_recall,
            'test_dead_recall' : test_dead_recall,
            'val_dead_missed'  : val_cm[0][1],
            'test_dead_missed' : test_cm[0][1]
        }

        print(f"  CV   ROC-AUC     : {cv_auc:.4f}")
        print(f"  Val  ROC-AUC     : {val_auc:.4f}")
        print(f"  Test ROC-AUC     : {test_auc:.4f}")
        print(f"  Val  Dead Recall : {val_dead_recall:.4f}")
        print(f"  Test Dead Recall : {test_dead_recall:.4f}")
        print(f"  Val  Dead Missed : {val_cm[0][1]:,}")
        print(f"  Test Dead Missed : {test_cm[0][1]:,}")

        if val_auc > best_auc:
            best_auc        = val_auc
            best_model      = model
            best_model_name = name

# ── Final Reports ─────────────────────────────────────────────────────────────

print("\n" + "="*70)
print(f"VALIDATION REPORT — {best_model_name}")
print("="*70)
print(classification_report(y_val,
                            best_model.predict(X_val_scaled),
                            target_names=['Dead(0)', 'Alive(1)']))

print("="*70)
print(f"TEST REPORT — {best_model_name}")
print("="*70)
print(classification_report(y_test,
                            best_model.predict(X_test_scaled),
                            target_names=['Dead(0)', 'Alive(1)']))

# ── Comparison Table ──────────────────────────────────────────────────────────

print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print(f"{'Model':<25} {'CV AUC':>8} {'Val AUC':>8} {'Test AUC':>9} {'Val Miss':>10} {'Test Miss':>10}")
print("-"*80)
for name, m in results.items():
    print(f"{name:<25} {m['cv_auc']:>8.4f} {m['val_auc']:>8.4f} "
          f"{m['test_auc']:>9.4f} {m['val_dead_missed']:>10,} {m['test_dead_missed']:>10,}")

# ── Save Best Model ───────────────────────────────────────────────────────────

joblib.dump(best_model, 'models/best_model.pkl')
print(f"\n🏆 Best Model : {best_model_name}")
print(f"   Val AUC   : {best_auc:.4f}")
print(f"\n✅ Best model saved to models/best_model.pkl")
print("✅ MLflow UI  : run 'mlflow ui' → open http://localhost:5000")