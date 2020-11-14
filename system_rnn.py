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
from tensorflow.python.ops.gen_array_ops import shape

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
process_rnn = keras.models.load_model("Models/process_rnn")

# %%
input = layers.Input(shape=(10, 1))

gate = keras.models.Model(inputs=input, outputs=tf.concat([input[1:5, 0], tf.reshape(input[4, 0], shape=(1, 1)), input[5:10, 0]], axis=0))

gate(np.linspace(1, 10, 10).reshape(10, 1))


# %%
def build_system_rnn(process):
    # Freezing process model layers
    for l in process.layers:
        l.trainable = False

    # Creating controller layers
    input = layers.Input(shape=(11, 1)) # 0:5 - u(t); 5:10 - y(t); 10:15 - sp(t);
    u = layers.GRU(4, activation='linear', use_bias=False)(input)
    u = layers.Dense(1)(u)

    y = tf.concat([input[0, 1:5, 0], u[0], input[0, 5:10, 0]], axis=0)
    y = tf.reshape(y, shape=(1, 10, 1))
    y = process(y)

    def custom_loss(sp, y_out, u_out):
        # k, p = tf.constant(0.8, dtype=tf.float32), tf.constant(0.2, dtype=tf.float32)

        return tf.math.squared_difference(sp, y_out)+tf.math.square(u_out)

    model = keras.models.Model(inputs=input,
                               outputs={"y":y, "u":u})

    model.add_loss(custom_loss(input[-1, 0], y, u))

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(optimizer=optimizer)
    
    return model

system_model = build_system_rnn(process_rnn)
system_model.summary()
# %%
print(np.arange(11).reshape((1, 11, 1)))
system_model(np.arange(11).reshape((1, 11, 1)))

# %%
past = 5
time = 10
sampling = 0.1

Y_data = np.array([])
U_data = np.array([])

for _ in range(10):
    x0_train = np.random.rand(2, 1)
    input_seq_train = np.repeat(np.random.rand(time, 1), 1/sampling)
    Y1, _, _ = ctrl.lsim(W1, U=input_seq_train, X0=x0_train, T=np.linspace(0, time, int(time/sampling)))
    
    Y_data = np.append(Y_data, [Y1])
    U_data = np.append(U_data, input_seq_train)


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
past = 5

Y_data = Y_data.T
Y_data = Y_data[0:-1].reshape((-1, 1))

U_data = U_data[1:].reshape((-1, 1))

X_train, Y_train = form_data(U_data, Y_data, past)

X_train = X_train.reshape(-1, 10, 1) 
X_train.shape, Y_train.shape
# %%

X_train = np.concatenate([X_train, np.ones((3494, 1, 1), dtype=np.float32)], axis=1)
X_train.shape

# %%
history = system_model.fit(x=X_train, epochs=100,
                    validation_split=0.2, verbose=2)

# %%
X_test = np.array([[15.58102578,  3.75740879,  3.75740879,  3.75740879,  3.75740879,
                    -5.05053568, -3.52573204, -2.44433999, -2.49559331, -2.36226845]])

# %%
Y_predicted_offline = np.zeros(shape=(predict_time, 1))
Y_past = Y_pred[0:past, :].T
X_predict_offline = np.zeros(shape=(1, 2*past))

for i in range(0, predict_time):
    X_predict_offline[:, 0:past] = X_test[i+2*past, 0:past].T
    X_predict_offline[:, past:2*past] = Y_past
    y_predict_tmp = model.predict(X_predict_offline.reshape(1, 10, 1))
    Y_predicted_offline[i] = y_predict_tmp
    Y_past[:, 0:past-1] = Y_past[:, 1:]
    Y_past[:, -1] = y_predict_tmp

# %%
plt.plot(Y_predicted_offline, label="Y")
plt.legend()
plt.show()