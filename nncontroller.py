# %% 
# Imports
import math
import re
import numpy as np
from numpy.core.fromnumeric import shape
from scipy import signal
import control.matlab as ctrl
import matplotlib.pyplot as plt
import pandas as pd
from simple_pid import PID

from scipy.integrate import odeint
import ipywidgets as wg
from IPython.display import display

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.merge import Concatenate

print(tf.__version__)
# %%
# OP - Controller output [u(t)]
# PV - Process output [y(t)]
# SP - Setpoint [U]
# e - error [SP-PV]

n = 100 # time points to plot
T = 20.0 # final time
SP_start = 2.0 # time of set point change

def process(y,t,u):
    Kp = 4.0
    taup = 3.0
    dydt = (1.0/taup) * (-y + Kp * u)
    return dydt

def pidPlot(Kc,tauI,tauD):
    t = np.linspace(0,T,n) # create time vector
    P= np.zeros(n)          # initialize proportional term
    I = np.zeros(n)         # initialize integral term
    D = np.zeros(n)         # initialize derivative term
    e = np.zeros(n)         # initialize error
    OP = np.zeros(n)        # initialize controller output
    PV = np.zeros(n)        # initialize process variable
    SP = np.zeros(n)        # initialize setpoint
    SP_step = int(SP_start/(T/(n-1))+1) # setpoint start
    SP[0:SP_step] = 0.0     # define setpoint
    SP[SP_step:n] = 4.0     # step up
    y0 = 0.0                # initial condition
    # loop through all time steps
    for i in range(1,n):
        # simulate process for one time step
        ts = [t[i-1],t[i]]         # time interval
        y = odeint(process,y0,ts,args=(OP[i-1],))  # compute next step
        y0 = y[1]                  # record new initial condition
        # calculate new OP with PID
        PV[i] = y[1]               # record PV
        e[i] = SP[i] - PV[i]       # calculate error = SP - PV
        dt = t[i] - t[i-1]         # calculate time step
        P[i] = Kc * e[i]           # calculate proportional term
        I[i] = I[i-1] + (Kc/tauI) * e[i] * dt  # calculate integral term
        D[i] = -Kc * tauD * (PV[i]-PV[i-1])/dt # calculate derivative term
        OP[i] = P[i] + I[i] + D[i] # calculate new controller output
        
    # plot PID response
    plt.figure(1,figsize=(15,7))
    plt.subplot(2,2,1)
    plt.plot(t,SP,'k-',linewidth=2,label='Setpoint (SP)')
    plt.plot(t,PV,'r:',linewidth=2,label='Process Variable (PV)')
    plt.legend(loc='best')
    plt.subplot(2,2,2)
    plt.plot(t,P,'g.-',linewidth=2,label=r'Proportional = $K_c \; e(t)$')
    plt.plot(t,I,'b-',linewidth=2,label=r'Integral = $\frac{K_c}{\tau_I} \int_{i=0}^{n_t} e(t) \; dt $')
    plt.plot(t,D,'r--',linewidth=2,label=r'Derivative = $-K_c \tau_D \frac{d(PV)}{dt}$')    
    plt.legend(loc='best')
    plt.subplot(2,2,3)
    plt.plot(t,e,'m--',linewidth=2,label='Error (e=SP-PV)')
    plt.legend(loc='best')
    plt.subplot(2,2,4)
    plt.plot(t,OP,'b--',linewidth=2,label='Controller Output (OP)')
    plt.legend(loc='best')
    plt.xlabel('time')
    
Kc_slide = wg.FloatSlider(value=0.1,min=-0.2,max=1.0,step=0.05)
tauI_slide = wg.FloatSlider(value=4.0,min=0.01,max=5.0,step=0.1)
tauD_slide = wg.FloatSlider(value=0.0,min=0.0,max=1.0,step=0.1)
wg.interact(pidPlot, Kc=Kc_slide, tauI=tauI_slide, tauD=tauD_slide)
# %% 
# Data generation
import random
random.seed(1)

n = 10000 # time points to plot
T = 1000.0 # final time
SP_start = 0.0 # time of set point change

def pidPlot_RndSP(Kc,tauI,tauD):
    t = np.linspace(0,T,n) # create time vector
    P= np.zeros(n)          # initialize proportional term
    I = np.zeros(n)         # initialize integral term
    D = np.zeros(n)         # initialize derivative term
    e = np.zeros(n)         # initialize error
    OP = np.zeros(n)        # initialize controller output
    PV = np.zeros(n)        # initialize process variable
    SP = np.zeros(n)        # initialize setpoint
    SP_step = int(SP_start/(T/(n-1))+1) # setpoint start

    for i in np.arange(n/200, dtype=int):
        SP[i*25*10: (i*25*10)+25*10] = np.random.randint(-5, 5)     # define setpoint

    y0 = 0.0                # initial condition

    # loop through all time steps
    for i in range(1,n):
        # simulate process for one time step
        ts = [t[i-1],t[i]]         # time interval
        y = odeint(process,y0,ts,args=(OP[i-1],))  # compute next step
        y0 = y[1]                  # record new initial condition

        # calculate new OP with PID
        PV[i] = y[1]               # record PV
        e[i] = SP[i] - PV[i]       # calculate error = SP - PV
        dt = t[i] - t[i-1]         # calculate time step
        P[i] = Kc * e[i]           # calculate proportional term
        I[i] = I[i-1] + (Kc/tauI) * e[i] * dt  # calculate integral term
        D[i] = -Kc * tauD * (PV[i]-PV[i-1])/dt # calculate derivative term
        OP[i] = P[i] + I[i] + D[i] # calculate new controller output
        
    
    # plot PID response
    plt.figure(1,figsize=(15,7))
    plt.subplot(2,2,1)
    plt.plot(t,SP,'k-',linewidth=2,label='Setpoint (SP)')
    plt.plot(t,PV,'r:',linewidth=2,label='Process Variable (PV)')
    plt.legend(loc='best')
    plt.subplot(2,2,2)
    plt.plot(t,P,'g.-',linewidth=2,label=r'Proportional = $K_c \; e(t)$')
    plt.plot(t,I,'b-',linewidth=2,label=r'Integral = $\frac{K_c}{\tau_I} \int_{i=0}^{n_t} e(t) \; dt $')
    plt.plot(t,D,'r--',linewidth=2,label=r'Derivative = $-K_c \tau_D \frac{d(PV)}{dt}$')    
    plt.legend(loc='best')
    plt.subplot(2,2,3)
    plt.plot(t,e,'m--',linewidth=2,label='Error (e=SP-PV)')
    plt.legend(loc='best')
    plt.subplot(2,2,4)
    plt.plot(t,OP,'b--',linewidth=2,label='Controller Output (OP)')
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.show()

    return SP, PV, OP, e
        
Kp_opt = 0.2
Ki_opt = 3.11
Kd_opt = 0.6

SP_data, PV_data, OP_data, e_data = pidPlot_RndSP(Kp_opt, Ki_opt, Kd_opt)
# %%
# Dataframe creation
data = pd.DataFrame(data=np.array([SP_data, OP_data, PV_data]).transpose(), columns=["SP", "OP", "PV"])

# Data preparation
train_dataset = data.sample(frac=0.9, random_state=0)
test_dataset = data.drop(train_dataset.index)

train_labels = train_dataset.pop('OP')
test_labels = test_dataset.pop('OP')

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

# Data Normalization
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# %%
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    # layers.Dropout(.2),
    layers.Dense(64, activation='relu'),
    # layers.Dropout(.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adam(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
  
pid_model = build_model()

# Параметр patience определяет количество эпох, проверяемых на улучшение
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)callbacks=[early_stop]

history = pid_model.fit(train_dataset, train_labels, 
                    epochs=100, validation_split = 0.2, 
                    verbose=1 )

pid_model.save('./Models/pid_nn')

# %%
# pid_model = keras.models.load_model('./Models/pid_nn')

def plot_history(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
#   plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [OP]')
  plt.legend()
  plt.grid(True)

# plot_history(history)

# %%
# Eval test dataset
pid_model.evaluate(test_dataset, test_labels)

# %%
# Plotting process with nn_pid controller
n = 100 # time points to plot
T = 20.0 # final time
SP_start = 2.0 # time of set point change

def NNControled_Plot(nn_model):
    t = np.linspace(0,T,n) # create time vector
    e = np.zeros(n)         # initialize error
    OP = np.zeros(n)        # initialize controller output
    PV = np.zeros(n)        # initialize process variable
    SP = np.zeros(n)        # initialize setpoint
    SP_step = int(SP_start/(T/(n-1))+1) # setpoint start

    SP[:] = 100

    y0 = 0.0                # initial condition

    # loop through all time steps
    for i in range(1,n):
        # simulate process for one time step
        ts = [t[i-1],t[i]]         # time interval
        y = odeint(process,y0,ts,args=(OP[i-1],))  # compute next step
        y0 = y[1]                  # record new initial condition

        # generate new OP
        PV[i] = y[1]               # record PV
        e[i] = SP[i] - PV[i]       # calculate error = SP - PV

        OP[i] = nn_model.predict(np.array([SP[i], PV[i]]).reshape(1, -1))[0][0]

        # OP[i] = P[i] + I[i] + D[i] # calculate new controller output
    
    return SP, PV, OP, e

SP1, PV1, OP1, e1 = NNControled_Plot(pid_model)

t = np.linspace(0,T,n) # create time vector
plt.figure(1,figsize=(15,7))
plt.subplot(2,2,1)
plt.plot(t,SP1,'k-',linewidth=2,label='Setpoint (SP)')
plt.plot(t,PV1,'r:',linewidth=2,label='Process Variable (PV)')
plt.legend(loc='best')
plt.subplot(2,2,2)
plt.plot(t,e1,'m--',linewidth=2,label='Error (e=SP-PV)')
plt.legend(loc='best')
plt.subplot(2,2,3)
plt.plot(t,OP1,'b--',linewidth=2,label='Controller Output (OP)')
plt.legend(loc='best')
plt.xlabel('time')
plt.show()

# %%
W1 = ctrl.rss(3, 1, 1)

Y1, T1 = ctrl.step(W1)
plt.plot(Y1)
plt.show()
# %%
# Make ss process model
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
W1

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
# Normalization 

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
# process_model.predict(np.zeros(shape=(1, 4)))
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
train_labels.head()
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
# %%
xs = np.array(xs)
plt.subplot(131)
plt.plot(T, X1[:, 0])
plt.plot(T, xs[:, 0])
plt.subplot(132)
plt.plot(T, X1[:, 1])
plt.plot(T, xs[:, 1])
plt.subplot(133)
plt.plot(T, X1[:, 2])
plt.plot(T, xs[:, 2])
plt.show()

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
system_model.output
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

# %%
# Ys_long = np.concatenate(Ys_long)
# Us_long = np.concatenate(Us_long)
# Xs_long = np.concatenate(Xs_long)

U = np.array([32, 28, 11, 3, 33, 17, 26, 37, 39, 44]).repeat(200)

plt.figure(figsize=(20,10))
plt.subplot(221)
plt.plot(U)
plt.plot(Ys_long)
plt.title("Ys")
plt.subplot(222)
plt.plot(Us_long)
plt.title("Us")
plt.subplot(223)
plt.plot(U)
plt.subplot(224)
plt.plot(Ys_long-U)
plt.title("err")
# plt.subplot(224)
# plt.plot(Xs_long)
# plt.title("Xs")
plt.show()

# np.concatenate(Ys_long)

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

# %% [markdown]
# ## Use RNNs to make process model

# %%
# Make ss process model
# A = [[  -6.06563565  -91.44136501   -5.18056403]
#  [   0.96510955   17.99936409    1.11912107]
#  [ -15.41658097 -244.30234385  -14.14321333]]
# B = [[0.40439781]
#  [0.70342777]
#  [0.88882234]]
# C = [[-0.87086982 -0.48585514  0.        ]]
# D = [[0.]]

# W1 = ctrl.rss(3, 1, 1)

T = np.linspace(0, 25, 1000) 
Y = np.zeros_like(T)
# W1 = ctrl.StateSpace(A, B, C, D)

Y1, T1 = ctrl.step(W1, T=T)
plt.plot(T1, Y1)
plt.show()

# %%
W1

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
data = pd.DataFrame(data=np.array([U_data, Y_data]).transpose(), columns=["U_data", "Y_data"])
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
    output_y = layers.LSTM(15, return_sequences=False, activation="relu")(combined)
    output_y = layers.Dense(32, activation="relu")(output_y)
    output_y = layers.Dense(32, activation="relu")(output_y)
    output_y = layers.Dense(32, activation="relu")(output_y)
    output_y = layers.Dense(32, activation="linear")(output_y)
    output_y = layers.Dense(32, activation="linear")(output_y)
    output_y = layers.Dense(4, activation="linear")(output_y)

        # Dividing output
    y_out = layers.Lambda(lambda x : x[:,0], name="y_out")(output_y)
    x_out = layers.Lambda(lambda x : x[:,1:], name="x_out")(output_y)

    model = keras.models.Model(inputs={
                                       "input_x":input_x, 
                                       "input_u":input_u, 
                                       "input_y":input_y}, 
                               outputs={"y":y_out,
                                        "x":x_out})

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='mse',
                    optimizer=optimizer)
    return model
  
rnn_process_model = build_rnn_process_model()
rnn_process_model.summary()

# %%
rnn_process_model({"input_x":np.zeros(shape=(1, 5, 3)), 
                   "input_u":np.zeros(shape=(1, 5, 1)), 
                   "input_y":np.zeros(shape=(1, 5, 1))})

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
                        "input_x":train_dataset_x,
                        "input_y":train_dataset_y,
                        "input_u":train_dataset_u
                    }, 
                    {
                        "y":train_labels['Y_data'],
                        "x":train_labels_x
                    }, 
                    epochs=100, validation_split = 0.1, 
                    verbose=1, 
                    callbacks = [early_stop])
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
    output = rnn_process_model.predict({"input_x":x0, "input_y":y0, "input_u":sp})
    y0[0, 1:, 0] = y0[0, -1, 0]
    y0[0, 0, 0] = output["y"]

    Ys[i] = output['y']

    sp[0,1:,0] = sp[0,:-1,0]
    sp[0,0,0] = 1

# plt.subplot(121)
# plt.plot(T, Y1)
plt.plot(T, Ys)
plt.title("Ys")
plt.show()
# %%
