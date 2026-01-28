import pandas as pd

df = pd.read_csv(
    'c:/Storage/Personal/School/UVA/Spring 2026/DS 3001/Homework/HW2/wrangling/assignment/data/mn_police_use_of_force.csv'
)

# Clean subject_injury
df['subject_injury_clean'] = df['subject_injury'].map({'Yes': 'Yes', 'No': 'No'})

# Proportion of missing values

total_obs = len(df)
missing_count = df['subject_injury_clean'].isna().sum()
proportion_missing = missing_count / total_obs

print("Total observations:", total_obs)
print("Missing subject_injury values:", missing_count)
print("Proportion missing:", proportion_missing)

# 4. Cross-tabulation with force_type

crosstab = pd.crosstab(
    df['force_type'],
    df['subject_injury_clean'],
    dropna=False
)

print(crosstab)