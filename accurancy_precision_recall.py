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

def calculate_precision(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Độ dài của y_true và y_pred phải giống nhau")

    # Số lượng true positives (TP) và false positives (FP)
    tp = 0
    fp = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1

    # Tính precision
    # print(tp," ||", fp)
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    return precision

def calculate_recall(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Độ dài của y_true và y_pred phải giống nhau")

    # Số lượng true positives (TP) và false negatives (FN)
    tp = 0
    fn = 0
    for i in range(len(y_true)):
        if (y_true[i] == 1 and y_pred[i] == 1):
            tp += 1
        elif (y_true[i] == 1 and y_pred[i] == 0) :
            fn += 1

    # Tính recall
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    return recall

df = pd.read_csv('diabetes.csv')

X_data = np.array(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']].values)
data=X_data
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = False)
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


# Tính độ chính xác bằng hàm custom
accuracy_custom = custom_accuracy(y_test, y_predict)
print(f'Độ chính xác bằng hàm custom: {accuracy_custom :.20f}')

precision = calculate_precision(y_test, y_predict)
print(f"Precision: {precision:.20f}")

recall = calculate_recall(y_test, y_predict)
print(f"Recall: {recall:.20f}")