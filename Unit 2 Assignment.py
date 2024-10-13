import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv('/Users/Mine/Desktop/CIS607/data_scientists_salaries_from_reddit.csv')

#Drop rows with missing values
df = df.dropna()

#converting salary to numeric
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')


#Filter the data to create two groups based on education names
masters_education_names = ["Master", "MS", "M.A."]
phd_education_names = ["PhD", "Ph.D"]

masters_group = df[df['education'].str.contains('|'.join(masters_education_names), case=False, na=False)]['salary']
phd_group = df[df['education'].str.contains('|'.join(phd_education_names), case=False, na=False)]['salary']

#Perform t-test for Hypothesis 1
t_stat_education, p_value_education = ttest_ind(masters_group, phd_group, equal_var=False)

#Filter the data based on location - New York, NY, USA vs other locations
ny_data_scientists = df[df['location'] == 'New York, NY, USA']['salary']
other_locations_data_scientists = df[df['location'] != 'New York, NY, USA']['salary']

#Perform t-test for Hypothesis 2
t_stat_location, p_value_location = ttest_ind(ny_data_scientists, other_locations_data_scientists, equal_var=False)

#Print results
print("Hypothesis 1 - Ph.D. vs. Master's:")
print("T-statistic for Education Comparison:", t_stat_education)
print("P-value for Education Comparison:", p_value_education)

if p_value_education < 0.05:
    print("Reject the null hypothesis: There is a significant difference in average salaries.")
else:
    print("Fail to reject the null hypothesis: No significant difference in average salaries.")
    
print("\nHypothesis 2 - New York, NY, USA vs. Other Locations:")
print("T-statistic for Location Comparison:", t_stat_location)
print("P-value for Location Comparison:", p_value_location)

#Interpretation
if p_value_location < 0.05:
    print("Reject the null hypothesis: There is a significant difference in average salaries.")
else:
    print("Fail to reject the null hypothesis: No significant difference in average salaries.")
