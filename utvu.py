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

df = pd.read_csv('D:/code hoc may/Breast_cancer_data.csv') #đọc tập csv và lưu trữ dạng dataframe
X_data = np.array(df[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness','diagnosis']].values) # lấy giá trị của các cột được chọn từ df và chuyển đổi thành mảng numpy
data=X_data #gán cho data
dt_Train, dt_Test = train_test_split(data, test_size=0.2, shuffle = True) #tách dữ liệu thành các tập huấn luyện và kiểm tra. 70% huấn luyện, 30% để kiểm tra. true là dữ liệu đc xáo trộn khi phân tách

X_train = dt_Train[:, :5]
y_train = dt_Train[:, 5]
X_test = dt_Test[:, :5]
y_test = dt_Test[:, 5]
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
pla = MLPClassifier(hidden_layer_sizes=(200,100,50,20), activation='tanh', max_iter=5000, random_state=42)
pla.fit(X_train, y_train)
y_predict = pla.predict(X_test)


accuracy_sklearn = accuracy_score(y_test, y_predict)
print(f'Độ chính xác bằng : {accuracy_sklearn}')