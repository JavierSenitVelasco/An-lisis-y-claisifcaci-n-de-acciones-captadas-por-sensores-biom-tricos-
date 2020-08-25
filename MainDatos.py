
"""

@author: Javier Senit Velasco y Alberto Pérez Garrido
"""

import Clasificador

import EstrategiaParticionado
from Datos import Datos
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from IPython.display import HTML, display
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



dataset=Datos("Watch_accelerometer.csv")

estrategia = EstrategiaParticionado.ValidacionCruzada()




encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1], sparse=False)
X = encAtributos.fit_transform(dataset.datos[:, :-1])
Y = dataset.datos[:, -4]
Y = np.asarray(Y, dtype=np.float64)
clf = MultinomialNB(alpha=0)
score = cross_val_score(clf, X, Y, cv=10) #
error_media_sk_sin_t = 1 - score.mean()
print ("¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬El error medio sin laplace y con validación cruzada es : " + str(error_media_sk_sin_t))





encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1], sparse=False)
X = encAtributos.fit_transform(dataset.datos[:, :-1])
Y = dataset.datos[:, -4]
Y = np.asarray(Y, dtype=np.float64)
clf = KNeighborsClassifier(n_neighbors=2, p=2, metric='euclidean')
score = cross_val_score(clf, X, Y, cv=10,n_jobs=-1)
error_media_sk_sin_t = 1 - score.mean()
print("¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬Error medio con knn " + str(error_media_sk_sin_t))



encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1], sparse=False)
X = encAtributos.fit_transform(dataset.datos[:, :-1])
Y = dataset.datos[:, -1]
Y = np.asarray(Y, dtype=np.float64)
clf = LogisticRegression(max_iter=10)
score = cross_val_score(clf, X, Y, cv=50,n_jobs=-1)
error_media_sk = 1 - score.mean()
error_std_sk = score.std()
print("¬¬¬¬¬¬¬¬¬¬¬¬Error medio con RegLog: " +str(error_media_sk))





