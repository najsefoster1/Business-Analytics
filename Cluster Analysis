import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#dataset
file_path = '/Users/Mine/Downloads/archive (3) 2/CO2 Emissions_Canada.csv' 
df = pd.read_csv(file_path)

#data is loaded properly
print(df.head()) 
#Select relevant columns (Engine Size, Fuel Consumption, etc.)
df = df[['Engine Size(L)', 'Fuel Consumption City (L/100 km)']]

#Split dataset into 80% for training and 20% for testing
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

#Scale the data for clustering
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)

#K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(train_scaled)

#Cluster centroids
centroids = kmeans.cluster_centers_

#Add cluster labels to training data
train_data['Cluster'] = kmeans.labels_

#Print centroids
print("Centroids: ", centroids)
print(train_data.head())
#Predict clusters for 5 records from the test data
test_scaled = scaler.transform(test_data)
test_data['Cluster'] = kmeans.predict(test_scaled)

#cluster predictions for 5 test records
print(test_data.head(5))
