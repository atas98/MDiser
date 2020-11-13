# Imports
import numpy as np
import control.matlab as ctrl
import matplotlib.pyplot as plt
import pandas as pd

from scipy.integrate import odeint
import ipywidgets as wg
from IPython.display import display

from plot_history import plot_history

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# %%
# Generate ss system 
W1 = ctrl.rss(3, 1, 1)

Y1, T1 = ctrl.step(W1)
plt.plot(Y1)
plt.show()
# %%
# Load predefined ss process model
A = [[-0.36228716, -0.12335425,  0.18812419],
     [-0.08761607, -0.42593229,  0.11877621],
     [-0.01100789,  0.00731988, -0.2019818 ]]
B = [[ 1.3176858 ], [-0.89495859], [ 0.6163043 ]]
C = [[ 0.93902826, -1.61913233, -0. ]]
D = [[0.54580745]]

# W1 = ctrl.rss(3, 1, 1)

T = np.linspace(0, 25, 1000) 
Y = np.zeros_like(T)
W1 = ctrl.StateSpace(A, B, C, D)

Y1, T1 = ctrl.step(W1, T=T)
plt.plot(T1, Y1)
plt.show()

# %%
# Data preparation
np.random.seed(1)
T = np.linspace(0, 100, 1000)
Y_data = np.array([])
U_data = np.array([])
X_data = np.array([])

# step
for _ in range(5):
    U1 = np.ones_like(T)
    Y1, T1, X1 = ctrl.lsim(W1, U=U1, T=T, X0=0.)
    Y_data = np.append(Y_data, Y1)
    U_data = np.append(U_data, U1)
    X_data = np.append(X_data, X1)


# impulse
for _ in range(5):
    U1 = np.zeros_like(T)
    U1[0] = 1.
    Y1, T1, X1 = ctrl.lsim(W1, U=U1, T=T, X0=X1[-1])
    Y_data = np.append(Y_data, Y1)
    U_data = np.append(U_data, U1)
    X_data = np.append(X_data, X1)

# random reapeted signal
for i in range(50):
    U1 = np.ones_like(T)*np.random.randint(-25, 25)
    Y1, T1, X1 = ctrl.lsim(W1, U=U1, T=T, X0=X1[-1])
    Y_data = np.append(Y_data, Y1)
    U_data = np.append(U_data, U1)
    X_data = np.append(X_data, X1)

# sin signal
# for i in range(10):
#     U1 = np.sin(T*0.01)*25.
#     Y1, T1, X1 = ctrl.lsim(W1, U=U1, T=T, X0=X1[-1])
#     Y_data = np.append(Y_data, Y1)
#     U_data = np.append(U_data, U1)
#     X_data = np.append(X_data, X1)

# random noise signal
# for i in range(5):
    # U1 = np.random.randint(-5, 5, size=T.shape)
    # Y1, T1, X1 = ctrl.lsim(W1, U=U1, T=T, X0=X1[-1])
    # Y_data = np.append(Y_data, Y1)
    # U_data = np.append(U_data, U1)
    # X_data = np.append(X_data, X1)


print(len(Y_data))
plt.plot(Y_data)
plt.show()

# %%
# Dataframe creation
data = pd.DataFrame(data=np.array([U_data, Y_data]).transpose(), columns=["U_data", "Y_data"])
data['X_data_0'] = X_data.reshape(-1, 3)[:, 0]
data['X_data_1'] = X_data.reshape(-1, 3)[:, 1]
data['X_data_2'] = X_data.reshape(-1, 3)[:, 2]

data['X_data_prev_0'] = np.array([0, *X_data.reshape(-1, 3)[:, 0][:-1]])
data['X_data_prev_1'] = np.array([0, *X_data.reshape(-1, 3)[:, 1][:-1]])
data['X_data_prev_2'] = np.array([0, *X_data.reshape(-1, 3)[:, 2][:-1]])

print(data.shape)

data.head(10)

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
def build_process_model():
    model = keras.models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(4,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(4, activation="linear")
    ])

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
    return model
  
process_model = build_process_model()
process_model.summary()

# %%
# Load model
process_model = keras.models.load_model("Models/process")

# %%
# Training process model
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
history = process_model.fit(train_dataset, train_labels, 
                    epochs=100, validation_split = 0.2, 
                    verbose=0 , batch_size=100,
                    callbacks = [early_stop])
# %%
# Plotting history and saving model
plot_history(history)
process_model.save('Models/process')

# %%
# Test model step
T = np.linspace(0, 100, 1000)
Y1, _, X1 = ctrl.lsim(W1, np.ones_like(T), T=T)

u = 1
x_prev = np.zeros(3)

y = process_model.predict(np.array([u, *x_prev]).reshape(1, -1))

ys = np.empty_like(T)
xs = []
for i, t in enumerate(T):
    y, x0, x1, x2 = process_model.predict(np.array([u, *x_prev]).reshape(1, -1))[0]
    x_prev[0] = x0
    x_prev[1] = x1
    x_prev[2] = x2
    xs.append([x0, x1, x2])
    ys[i] = y

plt.plot(T, ys.squeeze())
plt.plot(T, Y1)
plt.show()