import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier

"""
Authors:
Hubert Korzeniewski s19469
Adrian Szostak s19777

Wejście:
    W celu uruchomienia trzeba pobrać dane z https://www.openml.org/d/40996 i wypakować fashion-dataset.csv w folderze głównym.
   

    Fashion-dataset to zbiór danych obrazów artykułów Zalando, składający się z zestawu treningowego (60 000 przykładów) 
    i zestawu testowego (10 000 przykładów). Każdy przykład to obrazek w skali szarości 28x28, powiązany z etykietą z 10 klas produktów.

    0 T-shirt/top
    1 Trouser
    2 Pullover
    3 Dress
    4 Coat
    5 Sandal
    6 Shirt
    7 Sneaker
    8 Bag
    9 Ankle boot

Wyjście:
    Program wyświetla iteracje uczenia się oraz ostateczny wynik treningowy i testowy oraz ploty obrazujące
    ewaluacje skuteczności działania klasyfikatora

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
input_file = 'fashion-dataset.csv'
data = pd.read_csv(input_file, header=None, skiprows=[0])

X, y = data.loc[0:, :783], data.loc[0:, 784:]
X = X / 255.0

'''
Return a flattened array
'''
y = y.astype(int).values
y = y.ravel()

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
    hidden_layer_sizes=(130,90,), # Ten parametr pozwala nam ustawić liczbę warstw i liczbę węzłów, które chcemy mieć w klasyfikatorze sieci neuronowych
    max_iter=3,  # maksymalna liczba iteracji
    alpha=1e-4,  # warunek regulujący
    solver="sgd",  # odnosi się do stochastycznego spadku gradientu.
    verbose=True,  # czy drukować komunikaty o postępie na standardowe wyjście.
    random_state=1,  # Określa generowanie liczb losowych
    learning_rate_init=0.1,  # Zastosowana początkowa szybkość uczenia się. Używane tylko, gdy solver=’sgd’ lub ‘adam’.
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

'''
Creating and visualisation the confusion matrix for predictions
'''
predictions = mlp.predict(X_test)
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
cm = confusion_matrix(y_test, predictions, labels=mlp.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
disp.plot()
plt.show()