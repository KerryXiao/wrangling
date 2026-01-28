import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd  

df = pd.read_csv('c:/Storage/Personal/School/UVA/Spring 2026/DS 3001/Homework/HW2/wrangling/assignment/data/October 2017 Cohort_Virginia Pretrial Data Project_Deidentified FINAL Update_10272021.csv') 

df['released_pretrial_clean'] = (
    df['WhetherDefendantWasReleasedPretrial']
    .map({'Yes': 1, 'No': 0})
    .astype('float')   # ensures missing values are np.nan
)

df['released_pretrial_clean'].isna().sum()