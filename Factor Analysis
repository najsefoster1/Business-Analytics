import pandas as pd
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer

#dataset
df = pd.read_csv('/Users/Mine/Downloads/archive (1)/dataset.csv')

#Drop non-numeric columns and standardize the data
df_numeric = df.drop(columns=['GENDER', 'LUNG_CANCER'])
scaler = StandardScaler()
df_numeric_scaled = scaler.fit_transform(df_numeric)

#Factor Analysis with an initial guess of 5 factors
fa = FactorAnalyzer(n_factors=5, rotation="varimax")
fa.fit(df_numeric_scaled)

#Get Eigenvalues to decide on the number of factors
ev, v = fa.get_eigenvalues()
print("Eigenvalues:", ev)

#If Eigenvalues > 1 suggest the number of factors, re-run with the chosen number of factors
#Example: Let’s assume the result suggests 4 factors
fa = FactorAnalyzer(n_factors=4, rotation="varimax")
fa.fit(df_numeric_scaled)

#factor loadings and check the loading of each variable on the factors
loadings = fa.loadings_
print("Factor Loadings:\n", loadings)

#variance explained by each factor
variance = fa.get_factor_variance()
print("Variance explained by factors:", variance)
