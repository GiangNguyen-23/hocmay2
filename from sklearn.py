from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing 

df = pd.read_csv('D:/code hoc may/Breast_cancer_data.csv') #đọc tập csv và lưu trữ dạng dataframe
X_data = np.array(df[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness','diagnosis']].values) # lấy giá trị của các cột được chọn từ df và chuyển đổi thành mảng numpy
data=X_data #gán cho data
dt_Train, dt_Test = train_test_split(data, test_size=0.2, shuffle = True) #tách dữ liệu thành các tập huấn luyện và kiểm tra. 70% huấn luyện, 30% để kiểm tra. true là dữ liệu đc xáo trộn khi phân tách

X_train = dt_Train[:, :5]
y_train = dt_Train[:, 5]
X_test = dt_Test[:, :5]
y_test = dt_Test[:, 5]
clf = MLPClassifier(hidden_layer_sizes=(200,100,50,20), activation='tanh', max_iter=5000, random_state=42)
clf.fit(X_train, y_train) 
y_predict = clf.predict(X_test) 
count = 0 
for i in range(0,len(y_predict)) : 
    if(y_test[i] == y_predict[i]) : 
        count = count + 1 
print('Tỷ lệ dự đoán đúng : ', count/len(y_predict))
# print('Accuracy: ',accuracy_score(y_test, y_predict)) #Accuracy hay độ chính xác
# print('Precision: ',precision_score(y_test, y_predict,average='micro')) # Precision (độ chuẩn xác) tỉ lệ số điểm Positive (dự đoán đúng) / tổng số điểm mô hình dự đoán là Positive
# #càng cao càng tốt tức là tất cả số điểm đều đúng 
# print('Recall: ',recall_score(y_test, y_predict,average='micro')) #Recall tỉ lệ số điểm Positive mô hình dự đoán đúng trên tổng số điểm được gán nhãn là Positive ban đầu
# #Recall càng cao, tức là số điểm là positive bị bỏ sót càng ít
# #micro tbc của precision và recall theo các lớp.