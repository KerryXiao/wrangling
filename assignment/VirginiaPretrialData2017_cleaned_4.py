import pandas as pd

df = pd.read_csv('c:/Storage/Personal/School/UVA/Spring 2026/DS 3001/Homework/HW2/wrangling/assignment/data/October 2017 Cohort_Virginia Pretrial Data Project_Deidentified FINAL Update_10272021.csv') 

imposed_str = df["ImposedSentenceAllChargeInContactEvent"].astype(str).str.strip()
imposed_str = imposed_str.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
df["ImposedSentence_clean"] = pd.to_numeric(imposed_str, errors="coerce")

STRUCTURAL_ZERO_TYPES = [4, 9]

df.loc[
    df["SentenceTypeAllChargesAtConvictionInContactEvent"].isin(STRUCTURAL_ZERO_TYPES) & df["ImposedSentence_clean"].isna(),
    "ImposedSentence_clean"
] = 0

df["ImposedSentence_true_missing_flag"] = df["ImposedSentence_clean"].isna()

check = (
    df.groupby("SentenceTypeAllChargesAtConvictionInContactEvent")["ImposedSentence_clean"]
      .apply(lambda s: s.isna().mean())
      .rename("missing_rate")
      .reset_index()
)
print(check)