import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

#dataset
file_path = '/Users/Mine/Downloads/creditcard.csv'
df = pd.read_csv(file_path)

#Print the column names
print(df.columns)

# Adjust based on actual column names
features = ['V1', 'V2', 'Amount']  
output = 'Class'

X = df[features]  #Input
y = df[output]    #Output

#Split the dataset into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

#Train the model on the training data
clf.fit(X_train, y_train)

#Plot the tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=features, class_names=['No Default', 'Default'], filled=True)
plt.show()

#predict the target on the test data
y_pred = clf.predict(X_test)

#Compare predicted output with actual output to calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
