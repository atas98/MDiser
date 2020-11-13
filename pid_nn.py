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
# Process with pid controller
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

# %%
# Plot training history
plot_history(history)

# %%
# Saving model
pid_model.save('./Models/pid_nn')

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