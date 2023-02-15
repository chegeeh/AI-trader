from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import ta

# no keys required for crypto data
client = CryptoHistoricalDataClient()

trading_client = TradingClient('PK57XDIHU6ZMJ45B8DT1', '8CzDpnwTChn4s1DB7yzvg6HqutVIuDGaE8OqdiDP')

request_params = CryptoBarsRequest(
                        symbol_or_symbols=["BTC/USD"],
                        timeframe=TimeFrame.Day,
                        start=datetime.strptime("2020-01-01", '%Y-%m-%d')
                        )

bars = client.get_crypto_bars(request_params)

# convert to dataframe
x=bars.df

print(x)

# Prepare the data for use in the ANN model
data = x.drop(columns=['open', 'high', 'low'])
data['price_change'] = data['close'].pct_change()
data = data.dropna()

# Split the data into training and testing sets
X = data.drop(columns=['close', 'price_change'])
y = np.where(data['price_change'] > 0, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def algorithm1(X, y):
    print("************ ALGORITHM 1 RUNNING *************")
# Decision Tree Model
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    global y_pred_dt
    y_pred_dt = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    print('Decision Tree Accuracy:', dt_accuracy)




def algorithm2(X, y):
    print("************ ALGORITHM 2 RUNNING *************")
# Support Vector Machine Model
    svm = SVC()
    svm.fit(X_train, y_train)
    global y_pred_svm
    y_pred_svm = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    print('Support Vector Machine Accuracy:', svm_accuracy)




def algorithm3(X, y):
    print("************ ALGORITHM 3 RUNNING *************")
# Define the ANN model
    model = Sequential()
    model.add(Dense(120, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(83, activation='relu'))
    model.add(Dense(53, activation='sigmoid'))
    model.add(Dense(27, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the ANN model
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    global y_pred
# Print the accuracy of the model on the testing set
    y_pred = model.predict(X_test)
    y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
    annSCORE=accuracy_score(y_test, y_pred)
    print("ANN ALGORITHM SCORE ACCURACY: ", annSCORE)





def algorithm4(X, y):
    print("************ ALGORITHM 4 RUNNING *************")
# Deep Neural Network Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=128, activation='relu', input_dim=X_train.shape[1]))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test))
    global y_pred_dnn
    y_pred_dnn = model.predict(X_test)
    y_pred_dnn = [1 if y >= 0.5 else 0 for y in y_pred_dnn]
    dnn_accuracy = accuracy_score(y_test, y_pred_dnn)
    print('Deep Neural Network Accuracy:', dnn_accuracy)


def algorithm5(x):
    print("************ ALGORITHM 5 RUNNING *************")
    # Preprocess the data
    data = x.drop(columns=['open', 'high', 'low'])
    data['price_change'] = data['close'].pct_change()
    data = data.dropna()

# Calculate harmonic patterns using the ta library
#    data = x.reset_index()
#    data = ta.add_all_ta_features(data, "open", "high", "low", "close", "volume")


# Split the data into training and testing sets
    X = data.drop(columns=['close', 'price_change'])
    y = np.where(data['price_change'] > 0, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshape the input data to match the expected shape of the model
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

# Define the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((1, X_train.shape[1]), input_shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model on the training data
    history = model.fit(X_train, y_train, epochs=53, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test data
    results = model.evaluate(X_test, y_test)
    print('Test loss:', results[0])
    print('Test accuracy:', results[1])

# Predict the class labels of the test data
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)

# Calculate the prediction accuracy
    accuracy = np.mean(y_pred == y_test)
    print('Prediction accuracy:', accuracy)

def algorithm6(x):
    print("************ ALGORITHM 6 RUNNING *************")
    def preprocess_data(x):
        data = x.drop(columns=['open', 'high', 'low'])
        data['price_change'] = data['close'].pct_change()
        data = data.dropna()
        return data

    def build_model(input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def graph_nn(x):
        data = preprocess_data(x)
        X = data.drop(columns=['close', 'price_change'])
        y = np.where(data['price_change'] > 0, 1, 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)
        input_shape = (X_train.shape[1],)
        model = build_model(input_shape)
        model.fit(X_train, y_train, epochs=100, batch_size=22, validation_data=(X_test, y_test))
        results = model.evaluate(X_test, y_test)
        print('Test loss:', results[0])
        print('Test accuracy:', results[1])
        y_pred = model.predict(X_test)
        y_pred = np.round(y_pred)
        accuracy = np.mean(y_pred == y_test)
        print('Prediction accuracy:', accuracy)

    graph_nn(x)


'''    
# Preprocess the data
    data = x.drop(columns=['open', 'high', 'low'])
    data['price_change'] = data['close'].pct_change()
    data = data.dropna()
    
    # Split the data into training and testing sets
    X = data.drop(columns=['close', 'price_change'])
    y = np.where(data['price_change'] > 0, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.GraphConv(units=16, activation='relu'))
    model.add(tf.keras.layers.GraphConv(units=32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Fit the model on the training data
    history = model.fit(X_train, y_train, epochs=53, batch_size=32, validation_data=(X_test, y_test))
    
    # Evaluate the model on the test data
    results = model.evaluate(X_test, y_test)
    print('Test loss:', results[0])
    print('Test accuracy:', results[1])
    
    # Predict the class labels of the test data
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)
    
    # Calculate the prediction accuracy
    accuracy = np.mean(y_pred == y_test)
    print('Prediction accuracy:', accuracy)
'''
'''
def algorithm6(x):
    # Preprocess the data for training the LSTM model
    data = x.copy()
    data.drop(['open', 'high', 'low', 'volume'], axis=1, inplace=True)
    data['close_price'] = data['close'].astype(float)
    data['timestamp'] = data.index
    data = data.reindex(index=data.index[::-1])

    # Split the data into training and testing datasets
    split = int(len(data) * 0.8)
    training_data = data[:split]
    testing_data = data[split:]

    # Scale the data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    training_data['close_price'] = scaler.fit_transform(training_data[['close_price']])
    testing_data['close_price'] = scaler.transform(testing_data[['close_price']])

    # Convert the data into 3D arrays for training the LSTM model
    def data_to_3d_array(data, timestamp):
        X = []
        y = []
        for i in range(timestamp, len(data)):
            X.append(data[i-timestamp:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = data_to_3d_array(training_data[['close_price']].values, 60)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test, y_test = data_to_3d_array(testing_data[['close_price']].values, 60)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(33, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(tf.keras.layers.LSTM(units=49, activation='relu', return_sequences=False))
    #model.add(tf.keras.layers.LSTM(units=66,activation="sigmoid"))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the LSTM model to the training data
    model.fit(X_train, y_train, epochs=20, batch_size=5, verbose=1)


    # Evaluate the model on the test data
    results = model.evaluate(X_test, y_test)

    #predictions = scaler.inverse_transform(results)
    #test_acc = model.evaluate(X_test, y_test)
    print('Test Loss: ', results[0])
    print("Test Accuracy: ", results[1])
'''    

'''
    # Preprocess the data for training the LSTM model
    data = x.copy()
    data.drop(['open', 'high', 'low', 'volume'], axis=1, inplace=True)
    data['close_price'] = data['close'].astype(float)
    data['timestamp'] = data.index
    data = data.reindex(index=data.index[::-1])

    # Split the data into training and testing datasets
    split = int(len(data) * 0.8)
    training_data = data[:split]
    testing_data = data[split:]

     # Scale the data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    training_data['close_price'] = scaler.fit_transform(training_data[['close_price']])
    testing_data['close_price'] = scaler.transform(testing_data[['close_price']])

    # Convert the data into 3D arrays for training the LSTM model
    def data_to_3d_array(data, timestamp):
        X = []
        y = []
        for i in range(timestamp, len(data)):
            X.append(data[i-timestamp:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = data_to_3d_array(training_data[['close_price']].values, 60)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test, y_test = data_to_3d_array(testing_data[['close_price']].values, 60)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(tf.keras.layers.LSTM(units=50, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=30))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))


    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # Fit the LSTM model to the training data
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

# Evaluate the model on the test data
    test_loss = model.evaluate(X_test, y_test)
    test_acc = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss} Test Accuracy: {test_acc}')

# Make predictions on the test data
    predictions = model.predict(X_test)

# Inverse transform the predictions to get back the original price values
    predictions = scaler.inverse_transform(predictions)


'''   


'''

    data = x.drop(columns=['open', 'high', 'low'])
    data['price_change'] = data['close'].pct_change()
    data = data.dropna()

# Split the data into training and testing sets
    X = data.drop(columns=['close', 'price_change'])
    y = np.where(data['price_change'] > 0, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshape the input data to match the expected shape of the model
    # Reshape the input data for the LSTM
    X_train_timestamp = np.array(X_train.index.values).reshape(-1, 1)
    X_train_features = np.array(X_train).reshape(-1, X_train.shape[1], 1)
    X_train = np.concatenate((X_train_timestamp, X_train_features), axis=2)
    
    X_test_timestamp = np.array(X_test.index.values).reshape(-1, 1)
    X_test_features = np.array(X_test).reshape(-1, X_test.shape[1], 1)
    X_test = np.concatenate((X_test_timestamp, X_test_features), axis=2)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(tf.keras.layers.LSTM(units=50, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=30))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model on training data
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# Make predictions
    predictions = model.predict(X_test)
 #  predictions = np.where(predictions > 0.5, 1, 0)

'''







#algorithm1(X, y)
#algorithm2(X, y)
#algorithm3(X, y)
#algorithm4(X, y)
#algorithm5(x)
algorithm6(x)
'''
combined_predictions =(y_pred_dnn + y_pred + y_pred_svm + y_pred_dt) / 4

for prediction in combined_predictions:
    if prediction == 1:
        market_order_data = MarketOrderRequest(
            symbol="BTC/USD",
            qty=0.1000,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
            )
        market_order = trading_client.submit_order(
            order_data=market_order_data
            )
    else:
        market_order_data = MarketOrderRequest(
            symbol="BTC/USD",
            qty=0.1000,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY)
        market_order = trading_client.submit_order(
            order_data=market_order_data
            )
'''



