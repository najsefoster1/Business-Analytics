import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#Load dataset
file_path = '/Users/Mine/Desktop/CIS607/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

#Fixing the 'TotalCharges' column: Convert to numeric, coerce errors (turn invalid data into NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
#Hypothesis 1: Regression Analysis on Contract Length and Churn
#Change 'Churn' to binary for regression: Yes = 1, No = 0
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
X_h1 = df[['tenure']]  
y_h1 = df['Churn']

#Split the data for Hypothesis 1
X_train_h1, X_test_h1, y_train_h1, y_test_h1 = train_test_split(X_h1, y_h1, test_size=0.2, random_state=42)
regression_h1 = LinearRegression()
regression_h1.fit(X_train_h1, y_train_h1)
r_squared_h1 = regression_h1.score(X_test_h1, y_test_h1)

#Hypothesis 2: Logistic Regression on Service Usage and Churn
X_h2 = df[['MonthlyCharges', 'TotalCharges']] 
y_h2 = df['Churn']

#Split the data for Hypothesis 2
X_train_h2, X_test_h2, y_train_h2, y_test_h2 = train_test_split(X_h2, y_h2, test_size=0.2, random_state=42)
log_reg_h2 = LogisticRegression(max_iter=1000)
log_reg_h2.fit(X_train_h2, y_train_h2)

#Test the logistic regression model
y_pred_h2 = log_reg_h2.predict(X_test_h2)
accuracy_h2 = accuracy_score(y_test_h2, y_pred_h2)
conf_matrix_h2 = confusion_matrix(y_test_h2, y_pred_h2)

#Hypothesis 3: Cluster Analysis on Demographics and Usage Patterns
X_h3 = df[['tenure', 'MonthlyCharges', 'TotalCharges']] 

#Apply clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_h3)

#view clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['tenure'], df['MonthlyCharges'], c=df['Cluster'], cmap='viridis', marker='o')
plt.title('Customer Segments Based on Tenure and Monthly Charges')
plt.xlabel('Tenure')
plt.ylabel('Monthly Charges')
plt.colorbar(label='Cluster')
plt.show()

#Results summary
print(f"R-squared for Hypothesis 1 (Tenure vs. Churn): {r_squared_h1:.2f}")
print(f"Accuracy for Hypothesis 2 (Service Usage vs. Churn): {accuracy_h2 * 100:.2f}%")
print(f"Confusion Matrix for Hypothesis 2:\n{conf_matrix_h2}")
