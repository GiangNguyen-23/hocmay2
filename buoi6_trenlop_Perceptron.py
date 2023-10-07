import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Perceptron 
df = pd.read_csv('cars.csv') 
X_data = np.array(df[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']].values)

from sklearn import preprocessing 
data = pd.read_csv('cars.csv') 
le=preprocessing.LabelEncoder() 
data=data.apply(le.fit_transform) 
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = True) 
X_train = dt_Train.drop(['stt','acceptability'], axis = 1) 
y_train = dt_Train['acceptability'] 
X_test= dt_Test.drop(['stt','acceptability'], axis = 1)
y_test= dt_Test['acceptability']
y_test = np.array(y_test)
print(X_train)
print(y_train)
print(X_test)

pla = Perceptron() 
pla.fit(X_train, y_train) 
y_predict = pla.predict(X_test) 
count = 0 
for i in range(0,len(y_predict)) : 
    if(y_test[i] == y_predict[i]) : 
        count = count +1 
print('Ty le du doan dung : ', count/len(y_predict))

