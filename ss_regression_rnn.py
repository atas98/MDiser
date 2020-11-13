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

from plot_history import plot_history

# %%
# Make ss process model
A = [[-6.06563565,  -91.44136501,   -5.18056403],
     [0.96510955,   17.99936409,    1.11912107],
     [-15.41658097, -244.30234385,  -14.14321333]]
B = [[0.40439781],
     [0.70342777],
     [0.88882234]]
C = [[-0.87086982, -0.48585514,  0.]]
D = [[0.]]

W1 = ctrl.rss(6, 1, 1)

T = np.linspace(0, 25, 1000)
Y = np.zeros_like(T)
# W1 = ctrl.StateSpace(A, B, C, D)

Y1, T1 = ctrl.impulse(W1, T=T)
plt.plot(T1, Y1)
plt.show()

# %%
# Data preparation

np.random.seed(1)
T = np.linspace(0, 12, 120)
Y_data = np.array([])
U_data = np.array([])
X_data = np.array([])

# # step
# for _ in range(5):
#     U1 = np.ones_like(T)
#     Y1, T1, X1 = ctrl.lsim(W1, U=U1, T=T, X0=0.)
#     Y_data = np.append(Y_data, Y1)
#     U_data = np.append(U_data, U1)
#     X_data = np.append(X_data, X1)


# # impulse
# for _ in range(5):
#     U1 = np.zeros_like(T)
#     U1[0] = 1.
#     Y1, T1, X1 = ctrl.lsim(W1, U=U1, T=T, X0=X1[-1])
#     Y_data = np.append(Y_data, Y1)
#     U_data = np.append(U_data, U1)
#     X_data = np.append(X_data, X1)

# random reapeted signal
for i in range(100):
    U1 = np.ones_like(T)*np.random.randint(-25, 25)
    Y1, T1, X1 = ctrl.lsim(W1, U=U1, T=T, X0=X1[-1])
    Y_data = np.append(Y_data, Y1)
    U_data = np.append(U_data, U1)
    X_data = np.append(X_data, X1)

# sin signal
# for i in range(10):
#     U1 = np.sin(T*0.01)*50.
#     Y1, T1, X1 = ctrl.lsim(W1, U=U1, T=T, X0=X1[-1])
#     Y_data = np.append(Y_data, Y1)
#     U_data = np.append(U_data, U1)
#     X_data = np.append(X_data, X1)

# random noise signal
# for i in range(5):
#     U1 = np.random.randint(-5, 5, size=T.shape)
#     Y1, T1, X1 = ctrl.lsim(W1, U=U1, T=T, X0=X1[-1])
#     Y_data = np.append(Y_data, Y1)
#     U_data = np.append(U_data, U1)
#     X_data = np.append(X_data, X1)


print(len(Y_data))
plt.plot(Y_data)
plt.show()

# %%
# Dataframe creation
data = pd.DataFrame(data=np.array(
    [U_data, Y_data]).transpose(), columns=["U_data", "Y_data"])
data['X_data_0'] = X_data.reshape(-1, 3)[:, 0]
data['X_data_1'] = X_data.reshape(-1, 3)[:, 1]
data['X_data_2'] = X_data.reshape(-1, 3)[:, 2]

data['X_data_t-1_0'] = np.array([0, *X_data.reshape(-1, 3)[:, 0][:-1]])
data['X_data_t-1_1'] = np.array([0, *X_data.reshape(-1, 3)[:, 1][:-1]])
data['X_data_t-1_2'] = np.array([0, *X_data.reshape(-1, 3)[:, 2][:-1]])

data['X_data_t-2_0'] = data['X_data_t-1_0'].shift(periods=1, fill_value=0.0)
data['X_data_t-2_1'] = data['X_data_t-1_1'].shift(periods=1, fill_value=0.0)
data['X_data_t-2_2'] = data['X_data_t-1_2'].shift(periods=1, fill_value=0.0)

data['X_data_t-3_0'] = data['X_data_t-2_0'].shift(periods=1, fill_value=0.0)
data['X_data_t-3_1'] = data['X_data_t-2_1'].shift(periods=1, fill_value=0.0)
data['X_data_t-3_2'] = data['X_data_t-2_2'].shift(periods=1, fill_value=0.0)

data['X_data_t-4_0'] = data['X_data_t-3_0'].shift(periods=1, fill_value=0.0)
data['X_data_t-4_1'] = data['X_data_t-3_1'].shift(periods=1, fill_value=0.0)
data['X_data_t-4_2'] = data['X_data_t-3_2'].shift(periods=1, fill_value=0.0)

data['X_data_t-5_0'] = data['X_data_t-4_0'].shift(periods=1, fill_value=0.0)
data['X_data_t-5_1'] = data['X_data_t-4_1'].shift(periods=1, fill_value=0.0)
data['X_data_t-5_2'] = data['X_data_t-4_2'].shift(periods=1, fill_value=0.0)

data['U_data_t-1'] = data['U_data'].shift(periods=1, fill_value=0.0)
data['U_data_t-2'] = data['U_data'].shift(periods=2, fill_value=0.0)
data['U_data_t-3'] = data['U_data'].shift(periods=3, fill_value=0.0)
data['U_data_t-4'] = data['U_data'].shift(periods=4, fill_value=0.0)

data['Y_data_t-1'] = data['Y_data'].shift(periods=1, fill_value=0.0)
data['Y_data_t-2'] = data['Y_data'].shift(periods=2, fill_value=0.0)
data['Y_data_t-3'] = data['Y_data'].shift(periods=3, fill_value=0.0)
data['Y_data_t-4'] = data['Y_data'].shift(periods=4, fill_value=0.0)
data['Y_data_t-5'] = data['Y_data'].shift(periods=5, fill_value=0.0)

print(data.shape)

data.info(verbose=True)

# %%
# Train/Test dataframes preparation
train_dataset = data.sample(frac=0.9, random_state=0)
test_dataset = data.drop(train_dataset.index)

train_labels = pd.DataFrame()
train_labels['Y_data'] = train_dataset.pop('Y_data')
train_labels['X_data_0'] = train_dataset.pop('X_data_0')
train_labels['X_data_1'] = train_dataset.pop('X_data_1')
train_labels['X_data_2'] = train_dataset.pop('X_data_2')

test_labels = pd.DataFrame()
test_labels['Y_data'] = test_dataset.pop('Y_data')
test_labels['X_data_0'] = test_dataset.pop('X_data_0')
test_labels['X_data_1'] = test_dataset.pop('X_data_1')
test_labels['X_data_2'] = test_dataset.pop('X_data_2')

train_labels.head()

# %%
# Building model for process regression


def build_rnn_process_model():
    input_x = layers.Input(shape=(5, 3), name="input_x")
    input_u = layers.Input(shape=(5, 1), name="input_u")
    input_y = layers.Input(shape=(5, 1), name="input_y")

    x_layer = layers.LSTM(5, return_sequences=True, activation="relu")(input_x)
    u_layer = layers.LSTM(5, return_sequences=True, activation="relu")(input_u)
    y_layer = layers.LSTM(5, return_sequences=True, activation="relu")(input_y)

    combined = layers.concatenate([u_layer, y_layer])

    # output_y = layers.LSTM(64, return_sequences=True, activation="relu")(combined)
    # output_y = layers.LSTM(64, return_sequences=True, activation="relu")(combined)
    output_y = layers.LSTM(15, return_sequences=False,
                           activation="relu")(combined)
    output_y = layers.Dense(32, activation="relu")(output_y)
    output_y = layers.Dense(32, activation="relu")(output_y)
    output_y = layers.Dense(32, activation="relu")(output_y)
    output_y = layers.Dense(32, activation="linear")(output_y)
    output_y = layers.Dense(32, activation="linear")(output_y)
    output_y = layers.Dense(4, activation="linear")(output_y)

    # Dividing output
    y_out = layers.Lambda(lambda x: x[:, 0], name="y_out")(output_y)
    x_out = layers.Lambda(lambda x: x[:, 1:], name="x_out")(output_y)

    model = keras.models.Model(inputs={
        "input_x": input_x,
        "input_u": input_u,
        "input_y": input_y},
        outputs={"y": y_out,
                 "x": x_out})

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer)
    return model


rnn_process_model = build_rnn_process_model()
rnn_process_model.summary()

# %%
rnn_process_model({"input_x": np.zeros(shape=(1, 5, 3)),
                   "input_u": np.zeros(shape=(1, 5, 1)),
                   "input_y": np.zeros(shape=(1, 5, 1))})

# %%
# Reshaping inputs for lstm
train_dataset_x = np.stack([
    train_dataset[['X_data_t-1_0', 'X_data_t-1_1', 'X_data_t-1_2']].values,
    train_dataset[['X_data_t-2_0', 'X_data_t-2_1', 'X_data_t-2_2']].values,
    train_dataset[['X_data_t-3_0', 'X_data_t-3_1', 'X_data_t-3_2']].values,
    train_dataset[['X_data_t-4_0', 'X_data_t-4_1', 'X_data_t-4_2']].values,
    train_dataset[['X_data_t-5_0', 'X_data_t-5_1', 'X_data_t-4_2']].values
], axis=1)

train_dataset_y = np.stack([
    train_dataset[['Y_data_t-1']].values,
    train_dataset[['Y_data_t-2']].values,
    train_dataset[['Y_data_t-3']].values,
    train_dataset[['Y_data_t-4']].values,
    train_dataset[['Y_data_t-5']].values
], axis=1)

train_dataset_u = np.stack([
    train_dataset[['U_data']].values,
    train_dataset[['U_data_t-1']].values,
    train_dataset[['U_data_t-2']].values,
    train_dataset[['U_data_t-3']].values,
    train_dataset[['U_data_t-4']].values
], axis=1)

train_labels_x = np.stack([
    train_labels[['X_data_0']].values,
    train_labels[['X_data_1']].values,
    train_labels[['X_data_2']].values
], axis=1)
# %%
# Training process model
lr = 0.025
keras.backend.set_value(rnn_process_model.optimizer.learning_rate, lr)
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
history = rnn_process_model.fit(
    {
        "input_x": train_dataset_x,
        "input_y": train_dataset_y,
        "input_u": train_dataset_u
    },
    {
        "y": train_labels['Y_data'],
        "x": train_labels_x
    },
    epochs=100, validation_split=0.1,
    verbose=1,
    callbacks=[early_stop])
# %%
# Plotting history and saving model
plot_history(history)
rnn_process_model.save('Models/rnn_process')

# %%
# Testing rnn model
T = np.linspace(0, 20, 200)
Y1, T1, X1 = ctrl.lsim(W1, T=T, U=np.ones_like(T))

sp = np.zeros(shape=(1, 5, 1))
sp[0, 0, 0] = 1
y0 = np.zeros_like(sp)

Ys = np.empty_like(T)

for i, _ in enumerate(T):
    output = rnn_process_model.predict(
        {"input_x": x0, "input_y": y0, "input_u": sp})
    y0[0, 1:, 0] = y0[0, -1, 0]
    y0[0, 0, 0] = output["y"]

    Ys[i] = output['y']

    sp[0, 1:, 0] = sp[0, :-1, 0]
    sp[0, 0, 0] = 1

# plt.subplot(121)
# plt.plot(T, Y1)
plt.plot(T, Ys)
plt.title("Ys")
plt.show()
