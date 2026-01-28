import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Open the shark attack file in Excel
df = pd.read_excel("GSAF5.xls")

# 2. Drop columns that contain no data
df = df.dropna(axis=1, how="all")

# 3. Clean Year variable
df["Year_clean"] = pd.to_numeric(df["Year"], errors="coerce")

# Describe range of years
year_min = df["Year_clean"].min()
year_max = df["Year_clean"].max()
print(f"Year range: {year_min}–{year_max}")

# Focus on attacks since 1940
df_1940 = df[df["Year_clean"] >= 1940]

attacks_per_year = df_1940.groupby("Year_clean").size()

plt.figure()
plt.plot(attacks_per_year.index, attacks_per_year.values)
plt.xlabel("Year")
plt.ylabel("Number of attacks")
plt.title("Shark Attacks Over Time (since 1940)")
plt.show()

# 4. Clean Age variable + histogram
df["Age_clean"] = pd.to_numeric(df["Age"], errors="coerce")

plt.figure()
plt.hist(df["Age_clean"].dropna(), bins=20)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution of Shark Attack Victims")
plt.show()

# 5. Proportion of victims that are male
df["Sex_clean"] = df["Sex "].str.strip().str.upper()
prop_male = (df["Sex_clean"] == "M").mean()
print(f"Proportion male: {prop_male:.2f}")

# 6. Clean Type variable
df["Type_clean"] = (
    df["Type"]
    .str.strip()
    .str.capitalize()
    .where(lambda x: x.isin(["Provoked", "Unprovoked"]), "Unknown")
)

prop_unprovoked = (df["Type_clean"] == "Unprovoked").mean()
print(f"Proportion unprovoked: {prop_unprovoked:.2f}")

# 7. Clean Fatal Y/N variable
df["Fatal_clean"] = (
    df["Fatal Y/N"]
    .str.strip()
    .str.upper()
    .where(lambda x: x.isin(["Y", "N"]), "Unknown")
)

# 8. Comparisons

# Unprovoked attacks by sex
unprovoked_by_sex = (
    df[df["Type_clean"] == "Unprovoked"]
    .groupby("Sex_clean")
    .size()
)
print("\nUnprovoked attacks by sex:")
print(unprovoked_by_sex)

# Fatality rate by attack type
fatal_by_type = (
    df.groupby("Type_clean")["Fatal_clean"]
    .apply(lambda x: (x == "Y").mean())
)
print("\nFatality rate by attack type:")
print(fatal_by_type)

# Fatality rate by sex
fatal_by_sex = (
    df.groupby("Sex_clean")["Fatal_clean"]
    .apply(lambda x: (x == "Y").mean())
)
print("\nFatality rate by sex:")
print(fatal_by_sex)

# 9. Proportion of attacks by white sharks
species_words = df["Species "].dropna().str.lower().str.split()
prop_white_shark = species_words.apply(lambda words: "white" in words).mean()

print(f"\nProportion of attacks by white sharks: {prop_white_shark:.2f}")
