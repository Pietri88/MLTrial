import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
df = pd.read_csv('survey.csv') 
# Use descriptive statistics to identify potential outliers
print(df.describe())

# Visualize data to spot outliers using box plots
df.boxplot(column=['Year', 'Value'])  # Replace with actual column names
plt.show()

# Calculate Z-scores to identify outliers
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))

# Find rows with Z-scores greater than 3
outliers = (z_scores > 3).all(axis=1)
print(df[outliers])

df_no_outliers = df[(z_scores < 3).all(axis=1)]

print(df[df_no_outliers])

# Check for unique values in categorical columns to identify inconsistencies
print(df['Variable_name'].unique())  # Replace with actual column name

# Use value counts to identify unusual or erroneous entries
print(df['Value'].value_counts())

# Check numeric columns for impossible values (e.g., negative ages)
print(df[df['Year'] < 0])  # Replace “Age” with the actual column name


print(df['Year'])
