import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load the dataset from a CSV file
#Texas salaries dataset includes variables such as 'HRS PER WK', 'MONTHLY', and 'Annual',
#and the categorical variable of 'Status' that is utilized to divide the dataset. 
df = pd.read_csv('/Users/Mine/Desktop/CIS607/texas_salaries.csv')

#Descriptive statistics for the entire dataset
#This gives it an overview so that mean, standard deviation, median, max, and min can be calculated. 
overall_stats = df.describe()

#Additional statistics: mean, standard deviation, median, max, min for Unit 1 Assignement number 1
overall_mean = df.mean(numeric_only=True)
overall_std = df.std(numeric_only=True)
overall_median = df.median(numeric_only=True)
overall_max = df.max(numeric_only=True)
overall_min = df.min(numeric_only=True)

#Descriptive statistics by the categorical variable 'STATUS'
status_stats = df.groupby('STATUS').describe()

#Additional statistics by 'STATUS'
#Division by Status variable into meaningful parts. 
status_mean = df.groupby('STATUS').mean(numeric_only=True)
status_std = df.groupby('STATUS').std(numeric_only=True)
status_median = df.groupby('STATUS').median(numeric_only=True)
status_max = df.groupby('STATUS').max(numeric_only=True)
status_min = df.groupby('STATUS').min(numeric_only=True)

#Box plot for Annual Salary by Status for Unit 1 Assignment number 2
#Shows the comparison of the 'ANNUAL' variable among different 'STATUS' categories.
plt.figure(figsize=(12, 8))
sns.boxplot(x='STATUS', y='ANNUAL', data=df)
plt.title('Box Plot of Annual Salary by Employment Status', fontsize=14)
plt.xlabel('Employment Status', fontsize=12)
plt.ylabel('Annual Salary', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()

#Histogram for HRS PER WK distribution Unit 1 Assignment number 3
#Visual representation of distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['HRS PER WK'], kde=True, bins=20)
plt.title('Histogram of Hours Per Week Distribution', fontsize=14)
plt.xlabel('Hours Per Week', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()

#Output for the overall statistics
print("Overall Statistics:")
print(f"Mean:\n{overall_mean}\n")
print(f"Standard Deviation:\n{overall_std}\n")
print(f"Median:\n{overall_median}\n")
print(f"Max:\n{overall_max}\n")
print(f"Min:\n{overall_min}\n")

# Output the statistics by 'STATUS'
print("\nStatistics by STATUS:")
print(f"Mean:\n{status_mean}\n")
print(f"Standard Deviation:\n{status_std}\n")
print(f"Median:\n{status_median}\n")
print(f"Max:\n{status_max}\n")
print(f"Min:\n{status_min}\n")

# Textual description of the findings
#This description explains the results from the calculations and visualizations, and discusses the implications of the data distribution for Unit 1 number 4

text_description = """
Textual Description:

- The overall mean, standard deviation, median, max, and min values provide a general understanding of the dataset.
- The statistics by 'STATUS' offer insights into how different employment statuses influence variables like 'HRS PER WK', 'MONTHLY', and 'ANNUAL'.
- The box plot shows the distribution of annual salaries across different employment statuses, highlighting any outliers or variations in salary distribution.
- The histogram of hours per week provides an overview of the work distribution among employees, with the kernel density estimate (KDE) showing the distribution's shape.
"""
print(text_description)