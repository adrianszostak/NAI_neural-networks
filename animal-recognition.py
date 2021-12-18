import warnings
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

"""
Authors:
Hubert Korzeniewski s19469
Adrian Szostak s19777


Wejście:
    W celu uruchomienia trzeba pobrać dane z https://www.openml.org/d/40927 i wypakować cifar-10.csv w folderze głównym.

    CIFAR-10 to oznaczony podzbiór [zbior danych 80 milionów małych obrazów].
    Składa się z kolorowych obrazów 32x32 reprezentujących 10 klas obiektów:
    0. airplane
    1. automobile
    2. bird
    3. cat
    4. deer
    5. dog
    6. frog
    7. horse
    8. ship
    9. truck

Wyjście:
    Program wyświetla iterację uczenia się oraz ostateczny wynik treningowy i testowy.

Wykorzystywane biblioteki:
    pandas - do analizy danych
    sklearn - do obróbki danych i neuronowej sztucznej inteligencji

Dokumentacja kodu źródłowego:
    Python -> docstring (https://www.python.org/dev/peps/pep-0257/)
    sklearn - https://scikit-learn.org/stable/user_guide.html
    pandas -> https://pandas.pydata.org/
"""

'''
Data loading
'''
input_file = 'cifar-10.csv'
data = pd.read_csv(input_file, header=None, skiprows=[0])

X, y = data.loc[0:, :3072], data.loc[0:, 3072:]
X = X / 255.0
y = y.astype(int).values
y = y.ravel()
print(data.loc[0])

'''
Train and test split
'''
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

'''
Training data
'''
X_train, y_train = X[:num_training], y[:num_training]

'''
Test data
'''
X_test, y_test = X[num_training:], y[num_training:]

'''
Neural Network Classifier
'''
mlp = MLPClassifier(
    hidden_layer_sizes=(1200,500,70),
    max_iter=3, #maksymalna liczba iteracji
    alpha=1e-4, #warunek regulujący
    solver="sgd", #odnosi się do stochastycznego spadku gradientu.
    verbose=True, #czy drukować komunikaty o postępie na standardowe wyjście.
    random_state=1, #Określa generowanie liczb losowych
    learning_rate_init=0.1, #Zastosowana początkowa szybkość uczenia się. Używane tylko, gdy solver=’sgd’ lub ‘adam’.
)
'''
Catching warnings from MLPClassifier
'''
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

'''
Printing score for train and test dataset
'''
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
