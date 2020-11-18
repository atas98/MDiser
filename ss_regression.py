# %%
# Imports
import control.matlab as ctrl
import ipywidgets as wg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display
from scipy.integrate import odeint
from tensorflow import keras
from tensorflow.keras import layers

def plot_history(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  # plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [OP]')
  plt.legend()
  plt.grid(True)

# %%
# Make ss process model
# W1 = ctrl.rss(2, 1, 1)

A = [[ 1.25995527, -8.43046078],
 [ 2.74453578, -2.50931661]]

B = [[ 0.69248978],
 [-1.18452031]]

C = [[-0.50462121, -0.        ]]

D = [[-0.]]

T = np.linspace(0, 10, 50)
Y = np.zeros_like(T)
W1 = ctrl.StateSpace(A, B, C, D)

Y1, T1, X1 = ctrl.lsim(W1, U=np.ones_like(T), T=T)
plt.subplot(121)
plt.plot(T1, Y1)
plt.subplot(122)
plt.plot(T1, X1)
plt.show()
# %%
past = 5
time = 10
sampling = 0.1
# %%
def form_data(input_seq, output_seq, past):
    data_len = np.max(input_seq.shape)
    X = np.zeros(shape=(data_len-past, 2*past))
    Y = np.zeros(shape=(data_len-past,))
    for i in range(0, data_len-past):
        X[i, 0:past] = input_seq[i:i+past, 0]
        X[i, past:] = output_seq[i:i+past, 0]
        Y[i] = output_seq[i+past, 0]
    return X, Y

# %%
# define an input sequence for the simulation
Y_data = np.array([])
U_data = np.array([])

# random signal
for _ in range(10):
    x0_train = np.random.rand(2, 1)
    input_seq_train = np.repeat(np.random.rand(time, 1), 1/sampling)
    Y1, _, _ = ctrl.lsim(W1, U=input_seq_train, X0=x0_train, T=np.linspace(0, time, int(time/sampling)))
    
    Y_data = np.append(Y_data, [Y1])
    U_data = np.append(U_data, input_seq_train)

# random signal (to fill 4000 dots)
x0_train = np.random.rand(2, 1)
input_seq_train = np.random.rand(506, 1)

Y1, _, _ = ctrl.lsim(W1, U=input_seq_train, X0=x0_train, T=np.linspace(0, int(506*0.1), 506))
    
Y_data = np.append(Y_data, [Y1])
U_data = np.append(U_data, input_seq_train)

# random steped signal
x0_train = np.random.rand(2, 1)
input_seq_train = np.repeat(np.random.rand(time, 1), 1/sampling*20)
Y1, _, _ = ctrl.lsim(W1, U=input_seq_train, X0=x0_train, T=np.linspace(0, time*20, int(time*20/sampling)))

Y_data = np.append(Y_data, [Y1])
U_data = np.append(U_data, input_seq_train)

# random linespace
x0_train = np.random.rand(2, 1)
input_seq_train = np.array([])
for _ in range(5):
    input_seq_train = np.append(input_seq_train, [np.linspace(np.random.randint(0, 20), np.random.randint(20, 50), 50)])
Y1, _, _ = ctrl.lsim(W1, U=input_seq_train, X0=x0_train, T=np.linspace(0, 25*5, 50*5))

Y_data = np.append(Y_data, [Y1])
U_data = np.append(U_data, input_seq_train)

Y1, _, _ = ctrl.lsim(W1, U=-input_seq_train, X0=x0_train, T=np.linspace(0, 25*5, 50*5))

Y_data = np.append(Y_data, [Y1])
U_data = np.append(U_data, -input_seq_train)



plt.plot(Y_data)
plt.show()

# %% 
Y_data = Y_data.T
Y_data = Y_data[0:-1].reshape((-1, 1))

U_data = U_data[1:].reshape((-1, 1))

X_train, Y_train = form_data(U_data, Y_data, past)

X_train = X_train.reshape(-1, 10) 
X_train.shape, Y_train.shape

# %%
Y_data = np.array([])
U_data = np.array([])

for _ in range(10):
    x0_train = np.random.rand(2, 1)
    input_seq_train = np.repeat(np.random.rand(time, 1), 1/sampling)
    Y1, _, _ = ctrl.lsim(W1, U=input_seq_train, X0=x0_train, T=np.linspace(0, time, int(time/sampling)))
    
    Y_data = np.append(Y_data, [Y1])
    U_data = np.append(U_data, input_seq_train)


# random signal (to fill 4000 dots)
x0_train = np.random.rand(2, 1)
input_seq_train = np.random.rand(506, 1)

Y1, _, _ = ctrl.lsim(W1, U=input_seq_train, X0=x0_train, T=np.linspace(0, int(506*0.1), 506))
    
Y_data = np.append(Y_data, [Y1])
U_data = np.append(U_data, input_seq_train)

#  step
x0_train = np.random.rand(2, 1)
input_seq_train = np.repeat(np.random.rand(time, 1), 1/sampling*20)
Y1, _, _ = ctrl.lsim(W1, U=input_seq_train, X0=x0_train, T=np.linspace(0, time*20, int(time*20/sampling)))

Y_data = np.append(Y_data, [Y1])
U_data = np.append(U_data, input_seq_train)

# random linespace
x0_train = np.random.rand(2, 1)
input_seq_train = np.array([])
for _ in range(5):
    input_seq_train = np.append(input_seq_train, [np.linspace(np.random.randint(0, 20), np.random.randint(20, 50), 50)])
Y1, _, _ = ctrl.lsim(W1, U=input_seq_train, X0=x0_train, T=np.linspace(0, 25*5, 50*5))

Y_data = np.append(Y_data, [Y1])
U_data = np.append(U_data, input_seq_train)

Y1, _, _ = ctrl.lsim(W1, U=-input_seq_train, X0=x0_train, T=np.linspace(0, 25*5, 50*5))

Y_data = np.append(Y_data, [Y1])
U_data = np.append(U_data, -input_seq_train)

plt.plot(Y_data)
plt.show()

# %% 
Y_data = Y_data.T
Y_data = Y_data[0:-1].reshape((-1, 1))

U_data = U_data[1:].reshape((-1, 1))

X_val, Y_val = form_data(U_data, Y_data, past)

X_val = X_val.reshape(-1, 10) 
X_val.shape, Y_val.shape

# %%
model = keras.models.Sequential()
model.add(layers.Dense(2, activation='linear', use_bias=False, input_dim=2*past))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
history = model.fit(X_train, Y_train, epochs=1000, batch_size=200,
                    validation_data=(X_val, Y_val), verbose=1)
                    # callbacks=[early_stop])

# %%
model = keras.models.load_model("Models/process_rnn")

# %%
plot_history(history)
# %%
# for the time instants from 0 to past-1, we use the on-line data

predict_time = 100

Y_out = np.empty(predict_time)
input = np.zeros(shape=(1, 10))

for i in range(0, predict_time):
    y_predict_tmp = model.predict(input)

    input[0, 0:4] = input[0, 1:5]
    input[0, 4] = 1

    input[0, 4:9] = input[0, 5:10]
    input[0, 9] = y_predict_tmp

    Y_out[i] = y_predict_tmp

plt.plot(Y_out)
# plt.xlabel('Discrete time steps')
# plt.ylabel('Output')
# plt.legend()
# # plt.savefig('prediction_offline.png')
plt.show()
# %%
Y_data = np.array([])
U_data = np.array([])

# x0_train = np.random.rand(2, 1)
# input_seq_train = np.repeat(np.random.rand(time, 1), 1/sampling*20)
# Y1, _, _ = ctrl.lsim(W1, U=input_seq_train, X0=x0_train, T=np.linspace(0, time*20, int(time*20/sampling)))

# Y_data = np.append(Y_data, [Y1])
# U_data = np.append(U_data, input_seq_train)

# Y_data = Y_data.T
# Y_data = Y_data[0:-1].reshape((-1, 1))

# U_data = U_data[1:].reshape((-1, 1))

# X_test, Y_test = form_data(U_data, Y_data, past)

# X_test = X_test.reshape(-1, 10, 1) 
# X_test.shape, Y_test.shape

Y_pred = model.predict(X_val)



plt.plot(Y_pred[:100], label="Y_pred")
plt.plot(Y_val[:100], label="Y_test")
plt.legend()
plt.show()

# %%
Y_data = np.array([])
U_data = np.array([])

x0_train = np.zeros((2, 1))
T=np.linspace(0, 50, 500)
input_seq_train = np.ones_like(T)
Y1, _, _ = ctrl.lsim(W1, U=input_seq_train, T=T)

Y_data = np.append(Y_data, [Y1])
U_data = np.append(U_data, input_seq_train)

Y_data = Y_data.T
Y_data = Y_data[0:-1].reshape((-1, 1))
U_data = U_data[1:].reshape((-1, 1))

X_test, Y_test = form_data(U_data, Y_data, past)
X_test = X_test.reshape(-1, 10)

Y_pred = model.predict(X_test)

plt.plot(T[:-5], np.concatenate([np.zeros(shape=(1,1)), Y_pred], axis=0), label="Y_pred")
plt.plot(T[:-5], Y_data[:-4], label="Y_real")
plt.legend()
plt.show()
# %%
# Off-line prediction

predict_time = X_test.shape[0]-2*past

Y_predicted_offline = np.zeros(shape=(predict_time, 1))
Y_past = Y_pred[0:past, :].T
X_predict_offline = np.zeros(shape=(1, 2*past))

for i in range(len(X_test)-2*past):
    X_predict_offline[:, 0:past] = X_test[i+2*past, 0:past].T
    X_predict_offline[:, past:2*past] = Y_past
    y_predict_tmp = model.predict(X_predict_offline.reshape(1, 10))
    Y_predicted_offline[i] = y_predict_tmp
    Y_past[:, 0:past-1] = Y_past[:, 1:]
    Y_past[:, -1] = y_predict_tmp

# %%
plt.plot(Y_predicted_offline, label="Y_predicted_offline")
# plt.plot(Y_test, label="Y_test")
# plt.plot(Y_pred, label="Y_pred")
plt.legend()
plt.show()
# %%
model.save("Models/process_rnn_dense2")
# %%

# %%
cells = [2, 4, 8, 16, 32, 64, 128]

for cell in cells:
    print(f"No-bias; Cells: {cell}")

    model = keras.models.Sequential()
    model.add(layers.Dense(cell, activation='linear', use_bias=False, input_dim=2*past))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    history = model.fit(X_train, Y_train, epochs=1000, batch_size=200,
                        validation_data=(X_val, Y_val), verbose=0)

    model.save(f"Models/process_rnn_dense{cell}")
    with open("ss_regression_training.log", 'a') as log:
        log.write(f"{cell}: {model.evaluate(X_val, Y_val)} \n")

for cell in cells:
    print(f"No-bias; Cells: {cell}")

    model = keras.models.Sequential()
    model.add(layers.Dense(cell, activation='linear', use_bias=True, input_dim=2*past))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    history = model.fit(X_train, Y_train, epochs=1000, batch_size=200,
                        validation_data=(X_val, Y_val), verbose=0)

    model.save(f"Models/process_rnn_dense{cell}_bias")
    with open("ss_regression_training.log", 'a') as log:
        log.write(f"{cell}_bias: {model.evaluate(X_val, Y_val)} \n")

# %%
with open("y_train_f.npy", "wb") as f:
    np.save(f, Y_train)

with open("x_train_f.npy", "wb") as f:
    np.save(f, X_train)

with open("y_val_f.npy", "wb") as f:
    np.save(f, Y_val)

with open("x_val_f.npy", "wb") as f:
    np.save(f, X_val)
# %%
with open("y_train_f.npy", "rb") as f:
    Y_train = np.load(f)
# %%
Y_train.shape
# %%
