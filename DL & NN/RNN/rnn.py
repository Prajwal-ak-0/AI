import pandas as pd
import numpy as np
import matplotlib as plt

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []

for i in range(60, 1257):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Input


regressor = Sequential()

regressor.add(Input(shape=(x_train.shape[1],1)))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(x_train, y_train,batch_size=32 , epochs=100)

dataset_test = pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price = dataset_train.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train['open'],dataset_test['open']), axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60 :].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []

for i in range(60, 80):
    x_test.append(inputs[i-60:i,0])
    
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)










