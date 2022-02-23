## load data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

price_data = pd.read_csv('./StockPrice/AAPL.csv')
price_data.head(5)
price_data.describe()

plt.show()

plt.plot(price_data['Date'], price_data['Close'])
plt.title('1980-2021 Apple Stock Price')
plt.xlabel(price_data['Date'].to_datetime('%Y'))
plt.ylim([-5, 200])

import datetime

new_dataset = pd.DataFrame(index=range(0, len(price_data)), columns=['Date', 'Close'])
for i in range(0, len(price_data)):
    new_dataset["Date"][i] = price_data["Date"][i]
    new_dataset['Close'][i] = price_data['Close'][i]

# new_data.columns = [''] * len(new_data.columns)
final_data = new_dataset.values

train_data = final_data[0:9888, :]
valid_data = final_data[9888:, :]

new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(new_dataset)

x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i - 60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

lstm_model = Sequential()

lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))

lstm_model.add(LSTM(units=50))

lstm_model.add(Dense(1))

inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_closing_price = lstm_model.predict(X_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

train_data = new_dataset[:9888]
valid_data = new_dataset[9888:]
valid_data['Predictions'] = predicted_closing_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close', "Predictions"]])

lstm_model.evaluate(X_test, predicted_closing_price, batch_size=1, verbose=2)
