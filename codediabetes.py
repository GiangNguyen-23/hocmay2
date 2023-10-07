import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #chia dữ liệu thành các tập huấn luyện và kiểm tra
from sklearn.linear_model import Perceptron #mô hình preceptron_1 loại phân loại tuyến tính

df = pd.read_csv('diabetes.csv') #đọc tập csv và lưu trữ dạng dataframe

X_data = np.array(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']].values) # lấy giá trị của các cột được chọn từ df và chuyển đổi thành mảng numpy
data=X_data #gán cho data
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = False) #tách dữ liệu thành các tập huấn luyện và kiểm tra. 70% huấn luyện, 30% để kiểm tra. true là dữ liệu đc xáo trộn khi phân tách

X_train = dt_Train[:, :8]
y_train = dt_Train[:, 8]
X_test = dt_Test[:, :8]
y_test = dt_Test[:, 8]

#khởi tạo mô hình
pla =Perceptron()
pla.fit(X_train, y_train) #fit thực hiện huấn luyện mô hình

y_predict = pla.predict(X_test)
count = 0
#tính toán thủ công: ss các nhãn dự đoán với nhãn thực tế
for i in range(0, len(y_predict)):
    if(y_test[i] == y_predict[i]):
        count= count +1

print('Ty le du doan dung: ', count/len(y_predict))

