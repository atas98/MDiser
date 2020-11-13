# %%
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
# Load process model
process_model = keras.models.load_model("Models/process")

# %%
def build_system_model(proc_model):
    # Creating controller layers
    input_controller = layers.Input(shape=(2,))
    sp = input_controller[0]
    input_prev_x = layers.Input(shape=(3,))
    u = layers.Dense(64, activation='relu')(input_controller)
    u = layers.Dense(32, activation='relu')(u)
    u = layers.Dense(32, activation='relu')(u)
    u = layers.Dense(16, activation="relu")(u)
    u = layers.Dense(16, activation="relu")(u)
    u = layers.Dense(1, activation="linear", name="u_out")(u)
    merged_proc_input = layers.Concatenate(axis=1)([u, input_prev_x])

    # Freezing process model layers
    for l in proc_model.layers:
        l.trainable = False

    # Adding process model
    y = proc_model(merged_proc_input)

    # Dividing output
    y_out = layers.Lambda(lambda x : x[:,0])(y)
    x_out = layers.Lambda(lambda x : x[:,1:])(y)

    def custom_loss(sp, y_out, u):
        k, p = tf.constant(0.8, dtype=tf.float32), tf.constant(0.2, dtype=tf.float32)

        return k*tf.math.squared_difference(sp, 
                                            y_out, tf.float32)+\
                                            tf.math.square(u, tf.float32)

    model = keras.models.Model(inputs={"input_controller":input_controller, "input_prev_x":input_prev_x}, 
                               outputs={"y":y_out, "x":x_out, "u":u})

    model.add_loss(custom_loss(sp, y_out, u))

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(optimizer=optimizer)
    
    return model
  
system_model = build_system_model(process_model)
system_model.summary()
# %%
system_model = keras.models.load_model("Models/results/system_v2")
# %%
# Plotting model
keras.utils.plot_model(system_model, show_layer_names=False, show_shapes=True, expand_nested=True)
# %%
# Train model
# init start params
epochs = 1
T = np.linspace(0, 100, 1000)

for ep in range(epochs):
    print("Epoch№", ep)
    sp = np.random.randint(-5, 5)
    y0 = 0.0
    x0 = np.zeros(3).reshape(1, -1)
    prev_x = x0

    inp_ctrl = np.array([sp, y0]).reshape(1, -1)

    # process_model.fit(train_dataset, train_labels, 
                        # epochs=100, validation_split = 0.2, 
                        # verbose=0 , batch_size=100,
                        # callbacks = [early_stop])

    for t in T:
        print("\r", int(t), end="", flush=True)
        system_model.fit(x={"input_controller":inp_ctrl, "input_prev_x":prev_x}, y=np.array([sp]), epochs=1, verbose=0)
        output = system_model.predict({"input_controller":inp_ctrl, "input_prev_x":prev_x})
        inp_ctrl[0, 1] = output["y"]
        prev_x = output["x"]

# %%
# Testing
sp = 30.0
y0 = 0.0
x0 = np.zeros(3).reshape(1, -1)
prev_x = x0

inp_ctrl = np.array([sp, y0]).reshape(1, -1)

T = np.linspace(0, 20, 200)

Ys = np.empty_like(T)
Us = np.empty_like(T)
Xs = np.empty(shape=(200, 3))

prev_x = np.zeros(3).reshape(1, -1)
for i, _ in enumerate(T):
    output = system_model.predict({"input_controller":inp_ctrl, "input_prev_x":prev_x})
    inp_ctrl[0, 1] = output["y"]
    prev_x = output["x"]

    Ys[i] = output['y']
    Us[i] = output['u']
    Xs[i, 0] = prev_x[0, 0]
    Xs[i, 1] = prev_x[0, 1]
    Xs[i, 2] = prev_x[0, 2]

plt.subplot(221)
plt.plot(T, Ys)
plt.title("Ys")
plt.subplot(222)
plt.plot(T, Us)
plt.title("Us")
plt.subplot(223)
plt.plot(T, Xs)
plt.title("Xs")
plt.show()

# %%
def pretraining(epochs, sp):
    loss_fn = tf.math.squared_difference
    for _ in range(epochs):
        y0 = 0.0
        x0 = np.zeros(3).reshape(1, -1)
        prev_x = x0
        inp_ctrl = np.array([sp, y0]).reshape(1, -1)
        for _ in T:     
            with tf.GradientTape() as tape:
                output = system_model({"input_controller":inp_ctrl, "input_prev_x":prev_x}, training=True)  # Logits for this minibatch
                
                loss_value = loss_fn(tf.constant(sp, dtype=tf.float32), output["y"])

            grads = tape.gradient(loss_value, system_model.trainable_weights)
            system_model.optimizer.apply_gradients(zip(grads, system_model.trainable_weights))
            
            inp_ctrl[0, 1] = output["y"]
            prev_x = output["x"]
# %%
# Long Testing

y0 = 0.0
prev_x = np.zeros(3).reshape(1, -1) 

Ys_long = []
Xs_long = []
Us_long = []

for ep_sp in range(10):
    sp = np.random.randint(0, 50)
    print(ep_sp, sp)
    pretraining(2, sp)

    inp_ctrl = np.array([sp, y0]).reshape(1, -1)

    T = np.linspace(0, 20, 200)

    Ys = np.empty_like(T)
    Us = np.empty_like(T)
    Xs = np.empty(shape=(200, 3))

    for i, _ in enumerate(T):
        output = system_model.predict({"input_controller":inp_ctrl, "input_prev_x":prev_x})
        inp_ctrl[0, 1] = output["y"]
        prev_x = output["x"]

        Ys[i] = output['y']
        Us[i] = output['u']
        Xs[i, 0] = prev_x[0, 0]
        Xs[i, 1] = prev_x[0, 1]
        Xs[i, 2] = prev_x[0, 2]

    y0 = Ys[-1]
    print(y0)

    Ys_long.append(Ys.copy())
    Us_long.append(Us.copy())
    Xs_long.append(Xs.copy())