from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.dates import DateFormatter
import pandas
import keras
import numpy as np
import matplotlib.pyplot as plt
import mock
import datetime
import matplotlib.dates as mdates
import tensorflow as tf
import h5py
from keras.models import load_model

# Define time_step
time_step = 6

#Function for reading data
def read_dataset(filename, columns, scaler):
    dataframe = pandas.read_csv(filename, usecols=columns, engine='python')
    dataset = dataframe.valuesq
    #dataset = scaler.fit_transform(dataset)
    return dataset

#Function for formatting the data into proper arrays
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)

#Function for saving the model
# Save entire model (architecture + weights + config)
def save_model(filename, model):
    fn_json = filename + '.json'
    fn_h5 = filename + '.h5'
    model_json = model.to_json()
    with open(fn_json, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(fn_h5)



#hard tan function
def hard_tanh (x):
    return tf.minimum(tf.maximum(x, -1.), 1.)



#Function for adding noise to input
def noisy_input(x, c=0.05):
    noise = np.random.normal(loc=0.0, scale=1.0, size=np.shape(x))
    output = x + c*noise
    return output

#Function for training LSTM model
def model_lstm_train(units, trainX, trainY, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(LSTM(units, input_shape=(time_step, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/LSTM_model', model)

    return model

#Function for training GRU model
def model_gru_train(units, trainX, trainY, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(GRU(units, input_shape=(time_step, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/GRU_model',model)

    return model

#Function for training NAN LSTM model
def model_nan_lstm_train(units, trainX, trainY, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(LSTM(units, input_shape=(time_step, 1), activation=hard_tanh))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/NAN_LSTM_model', model)

    return model

#Function for training NAN GRU model
def model_nan_gru_train(units, trainX, trainY, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(GRU(units, input_shape=(time_step, 1), activation=hard_tanh))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/NAN_GRU_model',model)

    return model

#Function for training NANI LSTM model
def model_nani_lstm_train(units, trainX, trainY, epochs, batch_size, validation_split):
    trainX = noisy_input(trainX)
    model = Sequential()
    model.add(LSTM(units, input_shape=(time_step, 1), activation=hard_tanh))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/NANI_LSTM_model', model)

    return model

#Function for training NANI GRU model
def model_nani_gru_train(units, trainX, trainY, epochs, batch_size, validation_split):
    trainX=noisy_input(trainX)
    model = Sequential()
    model.add(GRU(units, input_shape=(time_step, 1), activation=hard_tanh))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/NANI_GRU_model',model)

    return model



#Set so results don't change every time it is run
np.random.seed(7)



#Scaler for data
scaler = MinMaxScaler(feature_range=(-1, 1))

#Read dataset from file. Dataset already seperated into two files for ease
train = read_dataset('data/train_dataset.csv', [1], scaler)
test = read_dataset('data/test_dataset.csv', [1], scaler)

#General variables
time_step = 6
len_day = 288 #len_day in terms of 5 minutes
# epochs = 500
batch_size = 32
units = 32
validation_split = 0.2

#Create datasets
trainX, trainY = create_dataset(train, time_step)
testX, testY = create_dataset(test, time_step)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

#For optimal training, search for the right number of iterations.
start = 20
end = 320
step = 20

n = (int)((end - start)/step)




for epochs in range(start, end, step):

     i = (int)((epochs - start)/step)

     print('Number of epochs: %d' %(epochs))

     modelLstm = model_lstm_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
     modelGru = model_gru_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

     modelNanLstm= model_nan_lstm_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
     modelNanGru = model_nan_gru_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

     modelNaniLstm= model_nani_lstm_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
     modelNaniGru = model_nani_gru_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)





