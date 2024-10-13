#Import necessary libraries for data manipulation and regression modeling
import pandas as pd  #for data handling
import numpy as np   #for numerical operations
from sklearn.model_selection import train_test_split  #to split the dataset
from sklearn.linear_model import LinearRegression  #to perform linear regression
from sklearn.metrics import mean_squared_error, r2_score  #for evaluation

#Load the CSV file into a DataFrame
data = pd.read_csv('/Users/Mine/Downloads/cs_students.csv')

#Let's take a look at the first few rows of the data
print("Initial Data Preview:")
print(data.head())

#Data Cleaning: transform the categorical values Python into numerical
#Remove any rows with missing values
#Assume proficiency levels are Strong = 3, Average = 2, Weak = 1
proficiency_map = {'Strong': 3, 'Average': 2, 'Weak': 1}
data['Python'] = data['Python'].map(proficiency_map)
data['SQL'] = data['SQL'].map(proficiency_map)
data['Java'] = data['Java'].map(proficiency_map)

#Drop any rows with missing data, if applicable
data = data.dropna()

#2. Input/Output Selection:
# GPA as the output variable (prediction)
# Selected 5 input variables
# Input variables: Age, Number of Projects, Python, SQL, and Java Proficiency

X = data[['Age', 'Projects', 'Python', 'SQL', 'Java']]  #input variables
y = data['GPA']  #output variable

# 3. Split the data into training and testing sets
#Training data will be used to train the model; testing data will evaluate it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Create and train the regression model
# We're using Linear Regression here
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Let's make predictions using the test set
y_pred = model.predict(X_test)

#6. Evaluate the model
#Look at metrics like Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#7. Display results
print("\nModel Coefficients (how much each feature impacts GPA):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

print(f"\nIntercept: {model.intercept_:.4f}")
print(f"\nMean Squared Error: {mse:.4f}")
print(f"R-squared (explains variance): {r2:.4f}")

#8. Predicting GPA for a new student:
#You can add new data for prediction
new_student = [[22, 5, 3, 2, 1]]  #Example: Age 22, 5 projects, Python=Strong, SQL=Average, Java=Weak
predicted_gpa = model.predict(new_student)
print(f"\nPredicted GPA for the new student: {predicted_gpa[0]:.2f}")

