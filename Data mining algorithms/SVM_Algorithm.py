import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# read data from csv file
data = pd.read_csv('diabetes.csv')
print(data.head())
print(data.describe())

# Splitting the dataset into training and testing sets.
x = data.drop(['Outcome'], axis=1)
# print(x.head())
y = data['Outcome']
# print(y.head())
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

# Creating the SVM model.
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)
y_prediction = classifier.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_prediction)*100)

print(confusion_matrix(y_test, y_prediction))
