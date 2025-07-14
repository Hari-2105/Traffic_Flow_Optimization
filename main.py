from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import tensorflow as tf
from keras.models import load_model

# -----------------------------
# Seed and Custom Activation
# -----------------------------
np.random.seed(7)

def hard_tanh(x):
    return tf.minimum(tf.maximum(x, -1.), 1.)

# -----------------------------
# Data Preparation
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
train = pd.read_csv('data/train_dataset.csv', usecols=[1]).values
test = pd.read_csv('data/test_dataset.csv', usecols=[1]).values

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 6
trainX, trainY = create_dataset(train, time_step)
testX, testY = create_dataset(test, time_step)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# -----------------------------
# Model Builder
# -----------------------------
def build_model(model_type, activation='tanh'):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(32, input_shape=(time_step, 1), activation=activation))
    elif model_type == 'GRU':
        model.add(GRU(32, input_shape=(time_step, 1), activation=activation, reset_after=False))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

# -----------------------------
# Models Configuration
# -----------------------------
models_info = {
    'LSTM_model': ('LSTM', 'tanh', False),
    'GRU_model': ('GRU', 'tanh', False),
    'NAN_LSTM_model': ('LSTM', hard_tanh, False),
    'NAN_GRU_model': ('GRU', hard_tanh, False),
    'NANI_LSTM_model': ('LSTM', hard_tanh, True),
    'NANI_GRU_model': ('GRU', hard_tanh, True)
}

# -----------------------------
# Evaluation and Predictions
# -----------------------------
results = []
predictions = []

for filename, (cell_type, activation, noisy) in models_info.items():
    print(f"Evaluating: {filename}")
    model = build_model(cell_type, activation=activation)
    model.load_weights(f"models/{filename}.h5")

    inputX = testX.copy().astype(np.float32)
    if noisy:
        inputX += 0.05 * np.random.normal(loc=0.0, scale=1.0, size=inputX.shape).astype(np.float32)

    train_pred = model.predict(trainX)
    test_pred = model.predict(inputX)

    train_mse = mean_squared_error(trainY, train_pred)
    train_mae = mean_absolute_error(trainY, train_pred)
    test_mse = mean_squared_error(testY, test_pred)
    test_mae = mean_absolute_error(testY, test_pred)

    results.append([
        filename,
        round(train_mse, 4), round(train_mae, 4),
        round(test_mse, 4), round(test_mae, 4)
    ])

    predictions.append((filename, test_pred))

# -----------------------------
# Save metrics to CSV
# -----------------------------
results_df = pd.DataFrame(results, columns=['Model', 'Train MSE', 'Train MAE', 'Test MSE', 'Test MAE'])
print(results_df)
results_df.to_csv('models/evaluation_metrics.csv', index=False)

# -----------------------------
# Plot True vs All Model Predictions
# -----------------------------
len_day = 288
start = len_day - 6
end = 2 * len_day

timeAxis = [datetime.datetime(2016, 3, 2, 0, 0) + datetime.timedelta(minutes=5 * i) for i in range(end)]
true = testY[start:start + end]

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(timeAxis, true, label='True Plot', color='black', linewidth=2)

for (filename, pred) in predictions:
    pred_cut = pred[start:start + end]
    ax.plot(timeAxis, pred_cut, label=filename.replace('_model', '').replace('_', ' '))

fig.autofmt_xdate()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('Time')
plt.ylabel('Vehicles / 5min')
plt.title('Traffic Flow: Prediction vs True')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Load current year data
import pandas as pd
from io import StringIO

# Provided string-based CSV input
time_step=6
current_data = """
5 Minutes,Lane 1 Flow (Veh/5 Minutes),# Lane Points,% Observed
01/26/2024 12:00,14,1,100
01/26/2024 12:05,12,1,100
01/26/2024 12:10,15,1,100
01/26/2024 12:15,14,1,100
01/26/2024 12:20,13,1,100
01/26/2024 12:25,17,1,100
01/26/2024 12:30,16,1,100
01/26/2024 12:35,15,1,100
01/26/2024 12:40,14,1,100
"""

# Convert string to DataFrame using StringIO
df_current = pd.read_csv(StringIO(current_data))

# Extract the required column (vehicle count)
current_values = df_current['Lane 1 Flow (Veh/5 Minutes)'].values.reshape(-1, 1)

# Normalize using the same scaler
# -----------------------------
# Fit Scaler on Training Data
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train)  # <-- Fix: This fits the scaler to the training data

current_scaled = scaler.transform(current_values)

# Create dataset using the same time step
currentX, currentY = create_dataset(current_scaled, time_step)
if currentX.size == 0:
    print("Not enough data to create input sequence for the model.")
    exit()
else:
    currentX = currentX.reshape(currentX.shape[0], currentX.shape[1], 1)


# Predict with one model for demo, e.g., LSTM
modelLstm = build_model('LSTM', activation='tanh')
modelLstm.load_weights("models/LSTM_model.h5")
current_pred = modelLstm.predict(currentX)

# Inverse transform to get actual scale
current_pred_inv = scaler.inverse_transform(current_pred)
current_true_inv = scaler.inverse_transform(currentY.reshape(-1, 1))

# Plot true vs predicted
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

timeAxis = [datetime.datetime(2024, 1, 26, 12, 0) + datetime.timedelta(minutes=5 * (i + time_step)) for i in range(len(current_pred_inv))]

plt.figure(figsize=(10, 5))
plt.plot(timeAxis, current_true_inv, label="True Flow", marker='o')
plt.plot(timeAxis, current_pred_inv, label="Predicted Flow", linestyle='--')
plt.title("LSTM Model - Traffic Flow Prediction (2024)")
plt.xlabel("Time")
plt.ylabel("Veh/5min")
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.grid(True)
plt.tight_layout()
plt.show()
