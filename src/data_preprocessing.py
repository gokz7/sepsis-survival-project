import pandas as pd                              # importing pandas to work with tables/dataframes
import numpy as np                               # importing numpy for numerical calculations

# ── Load the dataset ──────────────────────────────────────────────────────────

df = pd.read_csv('data/sepsis.csv')              # reading the CSV file and storing it as a dataframe

print("Shape of dataset:", df.shape)             # printing how many rows and columns we have
print("Column names:", df.columns.tolist())      # printing the exact column names

# ── Basic Inspection ──────────────────────────────────────────────────────────

print("\nFirst 5 rows:")                         
print(df.head())                                 # showing first 5 rows to visually verify the data loaded correctly

print("\nData types:")
print(df.dtypes)                                 # checking if each column has correct data type (int, float, object)

print("\nBasic statistics:")
print(df.describe())                             # showing min, max, mean, count for each column to spot anything unusual

# ── Check Missing Values ──────────────────────────────────────────────────────

print("\nMissing values in each column:")
print(df.isnull().sum())                         # counting how many empty/null values exist in each column

missing_percent = (df.isnull().sum() / len(df)) * 100   # calculating what percentage of each column is missing
print("\nMissing percentage:")
print(missing_percent)                           # printing the percentage so we know how serious the missing data is

# ── Handle Missing Values ─────────────────────────────────────────────────────

for col in df.columns:                           # looping through each column one by one
    if df[col].dtype in ['float64', 'int64']:    # checking if the column contains numbers
        median_val = df[col].median()            # calculating the median value of that column
        df[col] = df[col].fillna(median_val)     # filling any empty cells with the median value

# ── Fix Data Types ────────────────────────────────────────────────────────────

df['age_years'] = df['age_years'].astype(int)                        # converting age to integer since age cant be a decimal
df['sex_0male_1female'] = df['sex_0male_1female'].astype(int)        # converting sex column to integer (0 or 1)
df['episode_number'] = df['episode_number'].astype(int)              # converting episode number to integer
df['hospital_outcome_1alive_0dead'] = df['hospital_outcome_1alive_0dead'].astype(int)  # converting target column to integer

# ── Check Class Balance ───────────────────────────────────────────────────────

print("\nTarget column distribution:")
print(df['hospital_outcome_1alive_0dead'].value_counts())            # counting how many alive vs dead patients we have

alive = (df['hospital_outcome_1alive_0dead'] == 1).sum()             # counting total alive patients
dead  = (df['hospital_outcome_1alive_0dead'] == 0).sum()             # counting total dead patients
print(f"\nAlive: {alive} ({alive/len(df)*100:.1f}%)")                # printing alive count with percentage
print(f"Dead : {dead} ({dead/len(df)*100:.1f}%)")                    # printing dead count with percentage

# ── Save Cleaned Data ─────────────────────────────────────────────────────────

df.to_csv('data/sepsis_cleaned.csv', index=False)   # saving the cleaned dataframe as a new CSV file
print("\n sepsis_cleaned.csv saved in data folder")  # confirming the file was saved successfully