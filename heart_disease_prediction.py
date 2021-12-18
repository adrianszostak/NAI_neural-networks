import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.impute import KNNImputer
from sklearn import preprocessing
import warnings
from sklearn.exceptions import ConvergenceWarning

"""
Opracowanie:
    Autorzy: Hubert Korzeniewski
             Adrian Szostak
    Temat:   Sieci Neuronowe dla Klasyfikacji

Wejście:
    Heart Disease Prediction. W celu uruchomienia trzeba pobrać bazę z https://www.kaggle.com/gcdatkin/heart-disease-prediction/data
    i wypakować heart.csv w folderze głównym. Zbiór zawiera informacje o 
    1.age
    2.sex
    3.chest pain type (4 values)
    4.resting blood pressure
    5.serum cholestoral in mg/dl
    6.fasting blood sugar > 120 mg/dl
    7.resting electrocardiographic results (values 0,1,2)
    8.maximum heart rate achieved
    9.exercise induced angina
    10.oldpeak = ST depression induced by exercise relative to rest
    11.the slope of the peak exercise ST segment
    12.number of major vessels (0-3) colored by flourosopy
    13.thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
    14.heart disease: 1 = yes, 2 = no





Wyjście:
    Program wyświetla skuteczność przewidywania dla danych testowych

Wykorzystywane biblioteki:
    pandas - do analizy danych
    sklearn - do obróbki danych i neuronowej sztucznej inteligencjii
Dokumentacja kodu źródłowego:
    Python -> docstring (https://www.python.org/dev/peps/pep-0257/)
    pandas -> https://pandas.pydata.org/
    sklearn - https://scikit-learn.org/stable/user_guide.html
"""

input_file = 'heart.csv'
data = pd.read_csv(input_file, header=None, skiprows=[0])

X, y = data.loc[1:, 0:13], data.loc[1:, 13:]

# Cleaning dataset with kNN-Imputer
knn = KNNImputer()
knn.fit(X)
new_X = knn.transform(X)
new_X = pd.DataFrame(new_X)

y = y.astype(int).values
y = y.ravel()

# Train and test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training
# Training data
X_train, y_train = X[:num_training], y[:num_training]

# Test data
X_test, y_test = X[num_training:], y[num_training:]

mlp = MLPClassifier()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
