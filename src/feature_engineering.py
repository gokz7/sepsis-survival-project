import pandas as pd                                    # for loading and working with data
import numpy as np                                     # for numerical calculations
from sklearn.preprocessing import RobustScaler         # for scaling features (robust to outliers)
import joblib                                          # for saving the scaler to use later in app
import os                                              # for creating folders if they dont exist

# ── Load cleaned data ─────────────────────────────────────────────────────────

df = pd.read_csv('data/sepsis_cleaned.csv')            # loading the cleaned dataset
print("Loaded cleaned data:", df.shape)                # confirming shape

# ── 1. Age Grouping ───────────────────────────────────────────────────────────
# splitting age into clinical categories that doctors actually use

df['age_group'] = pd.cut(df['age_years'],              # cutting age column into bins
                          bins=[0, 18, 40, 60, 80, 100],        # age boundaries
                          labels=[0, 1, 2, 3, 4],     # 0=child, 1=young, 2=adult, 3=senior, 4=elderly
                          include_lowest=True)         # including age 0 in first bin
df['age_group'] = df['age_group'].astype(int)          # converting to integer

print("\nAge group distribution:")
print(df['age_group'].value_counts().sort_index())     # showing how many patients in each age group

# ── 2. Clinical Threshold Flags ───────────────────────────────────────────────
# creating binary flags based on clinically meaningful thresholds

df['is_elderly'] = (df['age_years'] >= 65).astype(int)         # 1 if patient is 65 or older
df['is_child'] = (df['age_years'] <= 18).astype(int)           # 1 if patient is 18 or younger
df['is_very_elderly'] = (df['age_years'] >= 80).astype(int)    # 1 if patient is 80 or older

print("\nElderly patients (65+):", df['is_elderly'].sum())      # printing count of elderly patients
print("Child patients (<=18):", df['is_child'].sum())          # printing count of child patients
print("Very elderly (80+):", df['is_very_elderly'].sum())      # printing count of very elderly

# ── 3. Risk Ratios ────────────────────────────────────────────────────────────
# creating ratio features that combine existing features

df['age_episode_ratio'] = df['age_years'] / df['episode_number']  # age divided by episode number
                                                                    # higher ratio = older patient with fewer episodes

# ── 4. Interaction Features ───────────────────────────────────────────────────
# combining two features to capture joint effects

df['age_x_episode'] = df['age_years'] * df['episode_number']   # age multiplied by episode number
                                                                 # captures combined effect of both

df['elderly_x_episode'] = df['is_elderly'] * df['episode_number']  # elderly flag times episode number
                                                                     # high value = elderly with multiple episodes

df['age_squared'] = df['age_years'] ** 2                        # age squared captures non-linear age effect
                                                                  # small children and very old both at risk

# ── 5. Feature Selection ──────────────────────────────────────────────────────
# keeping only the most useful features based on EDA findings

selected_features = [
    'age_years',               # original age - most important feature (0.9458 importance)
    'sex_0male_1female',       # original sex - keeping even though low importance
    'episode_number',          # original episode number
    'age_group',               # new - age category grouping
    'is_elderly',              # new - elderly flag (age >= 65)
    'is_child',                # new - child flag (age <= 18)
    'is_very_elderly',         # new - very elderly flag (age >= 80)
    'age_episode_ratio',       # new - age to episode ratio
    'age_x_episode',           # new - age times episode interaction
    'elderly_x_episode',       # new - elderly times episode interaction
    'age_squared',             # new - non linear age effect
    'hospital_outcome_1alive_0dead'  # target column - must keep this
]

df = df[selected_features]     # keeping only selected columns
print("\nFinal features selected:", df.shape)          # confirming final shape
print("Columns:", df.columns.tolist())                 # printing all column names

# ── 6. Scaling ────────────────────────────────────────────────────────────────
# scaling numerical features so no single feature dominates due to its scale

target = 'hospital_outcome_1alive_0dead'               # storing target column name
feature_cols = [c for c in df.columns if c != target]  # all columns except target

scaler = RobustScaler()                                # RobustScaler is better than StandardScaler
                                                        # because it handles outliers well
df_scaled = df.copy()                                  # making a copy to scale
df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])  # scaling all feature columns

# ── Save scaler ───────────────────────────────────────────────────────────────

os.makedirs('models', exist_ok=True)                   # creating models folder if it doesnt exist
joblib.dump(scaler, 'models/scaler.pkl')               # saving scaler to use later in streamlit app
print("\nScaler saved to models/scaler.pkl")           # confirming scaler saved

# ── Save engineered dataset ───────────────────────────────────────────────────

df.to_csv('data/sepsis_features.csv', index=False)     # saving unscaled version for reference
df_scaled.to_csv('data/sepsis_scaled.csv', index=False) # saving scaled version for model training
print("sepsis_features.csv saved in data folder")      # confirming features file saved
print("sepsis_scaled.csv saved in data folder")        # confirming scaled file saved

# ── Final Summary ─────────────────────────────────────────────────────────────

print("\n=== Feature Engineering Summary ===")
print(f"Original features  : 3")                       # we started with 3 features
print(f"Engineered features: {len(feature_cols)}")     # total features after engineering
print(f"New features added : {len(feature_cols) - 3}") # how many new features we created
print(f"Total records      : {len(df):,}")             # total patient records
print("\n✅ Feature engineering complete!")