# Import necessary modules


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# read data from csv file
data = pd.read_csv('diabetes.csv')
# print(data.head())

x = data.drop(['Outcome'], axis=1)
# print(x.head())
y = data['Outcome']
# print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Feature scaling
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(x_train, y_train)

y_prediction = knn.predict(x_test)

# print(y_prediction)


print("Confusion Matrix: ",confusion_matrix(y_test, y_prediction))

print("Classification Report:",classification_report(y_test, y_prediction))

print("Accuracy:",accuracy_score(y_test,y_prediction)*100)




