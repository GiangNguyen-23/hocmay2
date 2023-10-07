from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

def data_encoder(X):
    for i, j in enumerate(X):
        for k in range(0, 7):
            if (j[k] == "vhigh"):
                j[k] = 0
            elif (j[k] == "high"):
                j[k] = 1
            elif (j[k] == "med"):
                j[k] = 2
            elif (j[k] == "low"):
                j[k] = 3
            elif (j[k] == "2"):
                j[k] = 4
            elif (j[k] == "3"):
                j[k] = 5
            elif (j[k] == "4"):
                j[k] = 6
            elif (j[k] == "5more"):
                j[k] = 7
            elif (j[k] == "more"):
                j[k] = 8
            elif (j[k] == "small"):
                j[k] = 9
            elif (j[k] == "big"):
                j[k] = 10
            elif (j[k] == "acc"):
                j[k] = 1
            elif (j[k] == "unacc"):
                j[k] = -1
    return X

df = pd.read_csv('cars.csv')

X_data = np.array(df[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']].values)
data=data_encoder(X_data)

dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = True)
# print(dt_Train)

X_train = dt_Train[:, :6]
y_train = dt_Train[:, 6]
X_test = dt_Test[:, :6]
y_test = dt_Test[:, 6]

# Chuyển đổi nhãn thành kiểu số nguyên
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# pla =Perceptron()
# pla.fit(X_train, y_train)
# y_predict = pla.predict(X_test)

pla = MLPClassifier(hidden_layer_sizes=(50,30,10), activation='tanh', max_iter=1000, random_state=42)
pla.fit(X_train, y_train)
y_predict = pla.predict(X_test)


accuracy_sklearn = accuracy_score(y_test, y_predict)
print(f'Độ chính xác bằng : {accuracy_sklearn}')