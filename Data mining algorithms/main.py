# import libraries
import pandas as pd
import numpy as np

# read data from csv file
data = pd.read_csv('diabetes.csv')

print(data.head())

# preprocessing phase


# check missing values
data.isnull().sum()

# replace every 0 with null
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
data.isnull().sum()

# Replacing NaN with mean values through fillna method

data["Glucose"].fillna(data["Glucose"].mean(), inplace=True)
data["BloodPressure"].fillna(data["BloodPressure"].mean(), inplace=True)
data["SkinThickness"].fillna(data["SkinThickness"].mean(), inplace=True)
data["Insulin"].fillna(data["Insulin"].mean(), inplace=True)
data["BMI"].fillna(data["BMI"].mean(), inplace=True)

data.isnull().sum()

# ###############################################################################################################################

# visualize phase


import matplotlib.pyplot as plt
from seaborn import lineplot, distplot, scatterplot, boxplot

boxplot(data=data["BloodPressure"])
plt.show()

boxplot(data=data)
plt.show()

lineplot(data=data["BloodPressure"])
plt.show()

distplot(a=data["Insulin"])
plt.show

scatterplot(data=data["BMI"])
plt.show()
