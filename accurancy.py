from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Hàm tính độ chính xác bằng tay
def custom_accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for i in range(total):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / total


df = pd.read_csv('D:/Opengl/diabetes.csv')

X_data = np.array(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']].values)
data=X_data
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = True)
print(dt_Train)

X_train = dt_Train[:, :8]
y_train = dt_Train[:, 8]
X_test = dt_Test[:, :8]
y_test = dt_Test[:, 8]

pla =Perceptron()
pla.fit(X_train, y_train)
y_predict = pla.predict(X_test)
count = 0

for i in range(0, len(y_predict)):
    if(y_test[i] == y_predict[i]):
        count= count +1

print('Ty le du doan dung: ', count/len(y_predict))

accuracy_sklearn = accuracy_score(y_test, y_predict)
print(f'Độ chính xác bằng scikit-learn: {accuracy_sklearn * 100:.2f}%')

# # Tính độ chính xác bằng hàm custom
accuracy_custom = custom_accuracy(y_test, y_predict)
print(f'Độ chính xác bằng hàm custom: {accuracy_custom * 100:.2f}%')