import warnings
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import KNNImputer
import sklearn.metrics as sm
from sklearn.neural_network import MLPClassifier

"""
Authors:
Hubert Korzeniewski s19469
Adrian Szostak s19777

Input:
pima-indians-diabetes.csv

There are 4,898 observations with 11 input variables and one output variable. The variable names are as follows:
Fixed acidity.
Volatile acidity.
Citric acid.
Residual sugar.
Chlorides.
Free sulfur dioxide.
Total sulfur dioxide.
Density.
pH.
Sulphates.
Alcohol.
Quality (score between 0 and 10).

Output:
The program uses the Support Vector Classificator to classify data into two sets:
1. Not at risk of developing diabetes
2. At risk of developing diabetes

"""

def dataset(data):
    data = pd.DataFrame(data)

input_file = 'pima-indians-diabetes.csv'
data = pd.read_csv(input_file)

# Cleaning dataset with kNN-Imputer
# Replace 0 -> Null
data[['Glucose','BloodPressure','SkinThickness','Insuline','BMI']] = data[
    ['Glucose','BloodPressure','SkinThickness','Insuline','BMI']
    ].replace(0,np.NaN)

X, y = data.loc[:,[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insuline', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]], data.loc[:,['Outcome']]


# Using k-NN imputer replace NaN -> kNNValue
knn = KNNImputer()
knn.fit(X)
new_X = knn.transform(X)
new_X = pd.DataFrame(new_X)


# Scaling
new_X = preprocessing.minmax_scale(new_X)
new_X = pd.DataFrame(new_X)

y = y.astype(int).values
y = y.ravel()

# Train and test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train, y_train = new_X[:num_training], y[:num_training]

# Test data
X_test, y_test = new_X[num_training:], y[num_training:]


mlp = MLPClassifier(
    hidden_layer_sizes=(120, 84),
    max_iter=30,
    alpha=1e-4,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.1,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))