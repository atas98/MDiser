# %%
# Imports

import control.matlab as ctrl
import ipywidgets as wg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from scipy.integrate import odeint
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.python.training import optimizer

def plot_history(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [OP]')
    plt.legend()
    plt.grid(True)

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
# Defining process

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
# Loading rnn process model

process_rnn = keras.models.load_model("ss_regression/Models/process_rnn_gru128_bias")

# %%
# %%
past = 5
Y_data = np.array([])
U_data = np.array([])

x0_train = np.zeros((2, 1))
T=np.linspace(0, 100, 1000)
input_seq_train = np.concatenate([T[:500], -T[:500]])
Y1, _, _ = ctrl.lsim(W1, U=input_seq_train, T=T)

Y_data = np.append(Y_data, [Y1])
U_data = np.append(U_data, input_seq_train)

Y_data = Y_data.T
Y_data = Y_data[0:-1].reshape((-1, 1))
U_data = U_data[1:].reshape((-1, 1))

X_test, Y_test = form_data(U_data, Y_data, past)
X_test = X_test.reshape(-1, 10, 1)

Y_pred = process_rnn.predict(X_test)

plt.plot(T[:-5], np.concatenate([np.zeros(shape=(1,1)), Y_pred], axis=0), label="Y_pred")
plt.plot(T[:-5], Y_data[:-4], label="Y_real")
plt.plot(T[:-5], np.concatenate([np.zeros(shape=(1)), Y_out], axis=0), label="Y_real")
plt.xlim([47, 60])
plt.legend()
plt.show()

# %%
X_tmp = X_test[0].copy().reshape((1, 10, 1))
Y_out = np.empty(Y_pred.shape[0])

for i in range(Y_pred.shape[0]):
    Y_pred = process_rnn.predict(X_tmp).item()
    Y_out[i] = Y_pred
    X_tmp[0, 5:-1, 0] = X_tmp[0, 6:, 0]
    X_tmp[0, -1, 0] = Y_pred
    X_tmp[0, :5, 0] = X_test[i, :5, 0] 



plt.plot(Y_out)
plt.show()
# %%
X_test[0, :5, 0]
# %% [markdown]
## TODO:
# - [ ]  Move sp to criteria
# - [ ]  Try to start training with pid weights
# %%
# Defining rnn system model structure

def build_system_rnn(process):
    # Freezing process model layers
    for l in process.layers:
        l.trainable = False

    # Creating controller layers
    input = layers.Input(shape=(10, 1)) # 0:5 - u(t); 5:10 - y(t); 10:15 - sp(t);
    # u = layers.GRU(64, activation='linear', use_bias=False, stateful=True, return_sequences=True)(input)
    u = layers.LSTM(128, activation='linear', use_bias=True)(input)
    # u = layers.Dense(64, activation='linear')(input)
    # u = layers.Dense(64, activation='linear')(u)
    # u = layers.Dense(64, activation='linear')(u)
    u = layers.Dense(1)(u)

    y = tf.concat([input[0, 1:5, 0], u[0], input[0, 5:, 0]], axis=0)# u[0],
    y = tf.reshape(y, shape=(1, 10, 1))
    y = process(y)

    model = keras.models.Model(inputs=input,
                               outputs={"y":y, "u":u})

    # model.add_loss(custom_loss(input[-1, 0], y, u))

    optimizer=keras.optimizers.Adam(0.001)
    model.compile(optimizer=optimizer)
    
    return model

def custom_loss(sp, y_out, u_out):
    k, p = tf.constant(1, dtype=tf.float32), tf.constant(0.005, dtype=tf.float32)

    return tf.math.squared_difference(sp, y_out)+p*tf.math.square(u_out)

system_model = build_system_rnn(process_rnn)
system_model.summary()

# %% [markdown]

## Controller fitting
# 1. Simulate control-less process with constant U -> initial values
# 2. Initialize vars with values from previous step || initial values
# 3. Form data for model input 
# 4. Pass data to model/save grads/save outputs/losses * (sum/mean?) *
# 5. Apply grads with optimizer
# 6. Goto step one for N steps
# 7. Save results
# 8. Plot results depending on loss value * (transparency?) *
# %%
# Creating dataset from original process

past = 5
time = 20
sampling = 0.1

Y_data = np.array([])
U_data = np.array([])
X1 = [np.zeros((2, 1))]

input_seq_train = np.ones((200, 1))
Y1, _, X1 = ctrl.lsim(W1, U=input_seq_train, X0=X1[-1], T=np.linspace(0, time, int(time/sampling)))

Y_data = np.append(Y_data, [Y1])
U_data = np.append(U_data, input_seq_train)

plt.figure(figsize=(15, 15))
plt.subplot(221)
plt.plot(Y_data)
plt.title("Y")
plt.subplot(222)
plt.plot(U_data)
plt.title("U")
plt.subplot(223)
plt.plot(X1)
plt.title("X")
plt.show()
# %% 
# Forming initial dataset

past = 5

Y_data = Y_data.T
Y_data = Y_data[0:-1].reshape((-1, 1))

U_data = U_data[1:].reshape((-1, 1))

X_train, Y_train = form_data(U_data, Y_data, past)

X_train = X_train.reshape(-1, 10, 1)

# sp = (np.repeat(np.ones(2, dtype=np.float32), 100)[:-6].reshape((-1, 1, 1))*-10).astype(np.float32)
# X_train = np.concatenate([X_train, sp], axis=1)

X_train = X_train.astype(np.float32)

X_train.shape, Y_train.shape

# %%
plt.plot(Y_train)
# %%
# Simulate process with rnn model
X_train[0, :, 0].shape
Y_pred = system_model.predict(X_train[:, :, :])

plt.plot(np.concatenate([np.zeros((1)), Y_train[:100]], axis=0))
plt.plot(np.concatenate([np.zeros((1, 1)), Y_pred[:100]], axis=0))

plt.show()
# %%
# Training loop
keras.backend.set_value(system_model.optimizer.learning_rate, 0.001)

@tf.function
def training_step(x_batch):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:

        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        
        output = system_model(x_batch_train.reshape(1, -1, 1), training=True)  # {"y", "u"}
        # Compute the loss value for this minibatch.
        loss_value = custom_loss(10.0, output["y"], output["u"])

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, system_model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    
    system_model.optimizer.apply_gradients(zip(grads, system_model.trainable_weights))

    return loss_value, output

epochs = 1
Y_out = []
U_out = []
Losses = []
X_train_size = X_train.shape[0]

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    Y_out.append([])
    U_out.append([])

    tmp_x_train = X_train.copy()

    Losses.append([])

    # Iterate over the batches of the dataset.
    for i, x_batch_train in enumerate(tmp_x_train):
        # for _ in range(int(i*0.05)):
        loss, output = training_step(tmp_x_train[i])
            

        # Save outputs
        Y_out[-1].append(output["y"].numpy().item())
        U_out[-1].append(output["u"].numpy().item())
        Losses[-1].append(loss.numpy().item())

        if i < X_train_size-1:
            # Updating X_train with new u signal
            tmp_x_train[i+1, :4, :] = tmp_x_train[i, 1:5, :]
            tmp_x_train[i+1, 4, :] = U_out[-1][i]

            # Updating X_train with new y output
            tmp_x_train[i+1, 5:9, :] = tmp_x_train[i, 6:10, :]
            tmp_x_train[i+1, 9, :] = Y_out[-1][i]
    
    # if epoch%2 == 0:
    #     display.clear_output()
        
    #     fig = plt.figure(figsize=(13, 5))
    #     ax1 = fig.add_subplot(121)
    #     ax1.set_ylim([0, 20])
    #     ax1.plot(np.concatenate([np.zeros(1), np.array(Y_out)[-1, :].T], axis=0))
    #     ax2 = fig.add_subplot(122)
    #     ax2.set_ylim([0, 20])
    #     ax2.plot(np.concatenate([np.zeros(1), np.array(U_out)[-1, :].T], axis=0))

    #     display.display(fig)

        
    # Save loss value
    # Losses.append(loss_value)
    print(f"Loss value: {np.mean(Losses[-1])}")
    print(f"Last y: {Y_out[-1][-1]}")

plt.figure(figsize=(13, 5))
plt.subplot(121)
plt.title("Y_out")
plt.plot(np.array(Y_out)[-1, :].T)
plt.subplot(122)
plt.title("U_out")
plt.plot(np.array(U_out)[-1, :].T)
plt.show()

# %%
plt.figure(figsize=(13, 5))
plt.subplot(121)
plt.title("Y_out")
plt.plot(np.array(Y_out)[-1, :].T)
# plt.plot(Y_train)
plt.ylim([0, 15])
plt.subplot(122)
# plt.ylim([-150, 0])
plt.title("U_out")
plt.plot(np.array(U_out)[-1, :].T)
# # plt.ylim([10, 30])
# plt.subplot(223)
# plt.title("SP")
# plt.plot(sp.squeeze())
# plt.subplot(224)
# plt.title("delta")
# plt.plot(Y_out-np.array([sp.squeeze()]*len(Y_out)).T)
plt.show()

# %%
Y1, T1, X1 = ctrl.lsim(W1, U=np.array(U_out)[-1, :].T, T=np.linspace(0, time, int(time/sampling))[6:])
plt.subplot(121)
plt.plot(T1, Y1)
plt.subplot(122)
plt.plot(T1, np.array(U_out)[-1, :].T)

# %%
