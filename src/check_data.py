import pandas as pd                                    # importing pandas to read the csv file

df = pd.read_csv('data/sepsis_cleaned.csv')           # loading the cleaned dataset

print('=== Age suspicious values ===')
print('Age = 0 count:', (df['age_years'] == 0).sum())         # checking how many patients have age 0
print('Age < 0 count:', (df['age_years'] < 0).sum())          # checking how many patients have negative age

print('=== Episode suspicious values ===')
print('Episode = 0 count:', (df['episode_number'] == 0).sum()) # episode number should never be 0

print('=== Sex column unique values ===')
print(df['sex_0male_1female'].unique())                        # should only contain 0 and 1

print('=== Age outliers ===')
print('Min age:', df['age_years'].min())                       # minimum age in dataset
print('Max age:', df['age_years'].max())                       # maximum age in dataset
print('Ages below 1:', (df['age_years'] < 1).sum())           # infants - could be valid or data entry error