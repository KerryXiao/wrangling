import pandas as pd


# Read CSV
df = pd.read_csv('c:/Storage/Personal/School/UVA/Spring 2026/DS 3001/Homework/HW2/wrangling/assignment/data/airbnb_hw.csv')

# Clean Price: remove commas and convert to numeric, errors='coerce' will turn invalid to NaN
df['Price'] = (
    df['Price']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
)

price_numeric = pd.to_numeric(df['Price'], errors='coerce')

# Check missing values
missing_price_count = price_numeric.isna().sum()