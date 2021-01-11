# %%
# Plotting process with nn controller
n = 100 # time points to plot
T = 10 # final time
SP_start = 2.0 # time of set point change

env.reset()

def NNControled_Plot():
    t = np.linspace(0,T,n) # create time vector
    e = np.zeros(n)         # initialize error
    OP = np.zeros(n)        # initialize controller output
    PV = np.zeros(n)        # initialize process variable
    SP = np.ones(n)        # initialize setpoint
    SP_step = int(SP_start/(T/(n-1))+1) # setpoint start

    SP = np.repeat(np.random.uniform(0, 10, 10), 50)

    y0 = 0.0                # initial condition

    # loop through all time steps
    for i in range(1,n):
        # simulate process for one time step
        ts = [t[i-1],t[i]]         # time interval
        y = env(np.array(OP[i]).reshape(1), 0)         # compute next step
        y0 = y[0]                  # record new initial condition

        PV[i] = y[0]               # record PV
        e[i] = SP[i] - PV[i]       # calculate error = SP - PV

        state = tf.expand_dims(tf.convert_to_tensor([PV[i], SP[i]]), 0)
        OP[i] = tf.squeeze(actor_model(state)).numpy()

        # OP[i] = P[i] + I[i] + D[i] # calculate new controller output
    
    return SP, PV, OP, e

SP1, PV1, OP1, e1 = NNControled_Plot()

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
# Target value changes over-time

Ys = []
Us = []
SPs = []

x0 = None

x0 = env.reset(x0=x0)
sp = [100, 100]
for t in range(len(T)-2):
    # if t%50==0:
    #     sp = np.array(np.random.randint(1, 20, size=2)) 
    #     SPs.append(sp)

    state, _ = env.ret_state(sp)
    action = target_actor(state.reshape(1, -1)).numpy()
    Ys.append(env(action))
    Us.append(action)

Us = np.array(Us)
Ys = np.array(Ys)
SPs = np.array(SPs)

plt.figure(figsize=(15, 7))

ax1 = plt.subplot(121)
ax1.set_title("Виходи процесу")
ax1.set_ylabel("Y")
ax1.set_xlabel("T")
ax1.plot(np.linspace(0, 50, 248), Ys)
ax1.plot(np.linspace(0, 50, 248), np.ones(248)*10, "--")
ax1.legend(["Вихід процесу", "Завдання"])

ax2 = plt.subplot(122)
ax2.set_xlabel("T")
ax2.set_ylabel("U")
ax2.set_title("Сигнали керування")
ax2.plot(np.linspace(0, 50, 248), Us[:, 0, :])
ax2.legend(["Керування"])

plt.show()

# %%
# Target value is curved
Ys = []
Us = []
SPs = []

x0 = None

x0 = env.reset(x0=x0)
for t in range(len(T)-2):
    sp = np.sin(t/25)*8+10 
    SPs.append(sp)

    state, _ = env.ret_state(sp)
    action = target_actor(state.reshape(1, -1)).numpy()
    Ys.append(env(action))
    Us.append(action)

Us = np.array(Us)
Ys = np.array(Ys)
SPs = np.array(SPs)

plt.figure(figsize=(15, 7))

ax1 = plt.subplot(121)
ax1.set_title("Виходи процесу")
ax1.set_ylabel("Y")
ax1.set_xlabel("T")
ax1.plot(Ys)
# ax1.plot(np.linspace(0, 50, 249), SPs, "--")
ax1.legend(["Вихід 1", "Завдання"])

ax2 = plt.subplot(122)
ax2.set_xlabel("T")
ax2.set_ylabel("U")
ax2.set_title("Сигнали керування")
# ax2.plot(np.linspace(0, 50, 249), Us[:, 0, :])
ax2.legend(["Керування"])

plt.show()

# %%
# Plotting systems form different learning episodes

Ys = []
Us = []
SPs = []

x0 = None

x0 = env.reset()
for t in range(len(T)-1):
    sp = [10] 
    SPs.append(sp)

    state, _ = env.ret_state(sp)
    action = target_actor(state.reshape(1, -1)).numpy()
    Ys.append(env(action))
    Us.append(action)

Us = np.array(Us)
Ys = np.array(Ys)
SPs = np.array(SPs)

plt.figure(figsize=(15, 7))
# plt.ylim(bottom=-5, top=18)
# plt.xlim(left=0, right=20)
plt.title("Виходи процесу")
plt.ylabel("Y")
plt.xlabel("T")
plt.plot(Ys, 'b', linewidth=1.5, label="Кінець навчання")
plt.plot(SPs, "r--", label="Завдання")
# np.linspace(0, 50, 249)[:100],
# np.linspace(0, 50, 249)[:100],

ep_models_ys = np.zeros(shape=(11, 249))


directory = r'Models/q_learning/SISO/'
for i, model_dir in enumerate(os.scandir(directory)):
    if model_dir.is_dir() and "target" not in model_dir.name and "actor" in model_dir.name:
        model = tf.keras.models.load_model(model_dir.path, compile=False)
        ep = int(model_dir.name[8:])
        linewidth = 1 # ep*2.5/2500+0.5
        alpha = 1 # ep/2500*0.89+0.1


        x0 = env.reset()
        sp = [10]

        for t in range(len(T)-1):
            state, _ = env.ret_state(sp)
            action = model(state.reshape(1, -1)).numpy()
            ep_models_ys[i, t] = env(action)

        # np.linspace(0, 50, 1000), 
        if not ep == 2500:
            plt.plot(ep_models_ys[i], linewidth=linewidth, alpha=alpha, label=f"Епізод {ep}")
 

plt.legend(loc=4, prop={'size': 15})
plt.show()
# %%
# Smoothing ep_models plots
import scipy.interpolate

for i, row in enumerate(ep_models_ys):
    f = interpolate.interp1d(np.linspace(0, 50, 249), row)
    newT = np.linspace(0, 50, 1000)

    newY = f(newT)
    newY = np.array(newY)

    ep_models_ys_smooth[i, :] = newY

# %%
# Plot learning history 

from matplotlib.patches import Polygon

plt.figure(figsize=(15, 8))
fig, ax = plt.subplots()

ax.plot(np.linspace(0, 10000, 10000), avg_reward_list, linewidth=2)

area_iy = np.concatenate([avg_reward_list[:1000], np.repeat(None, 1500)])
area_ix = np.linspace(0, 10000, 10000)

bbox = dict(boxstyle="round", fc="0.8")
arrowprops = dict(
    arrowstyle = "->")
disp = ax.annotate('Заповнення\n буфера досвіду',
            (500, -6000), xytext=(0.5*150, -35),
            textcoords='offset points',
            bbox=bbox, arrowprops=arrowprops)

verts = [(1000, -10500.0), (0, -10500.0), *zip(area_ix, area_iy), (1000, -10500.0), (0, -10500.0)]
poly = Polygon(verts, facecolor='0.9')
ax.add_patch(poly)



ax.set_xlabel("Episode")
ax.set_ylabel("Avg. Epsiodic Reward")
plt.show()
# %%
# Plot static characteristics with deltas
T = np.linspace(0, 50, 249)
sps1 = np.linspace(0.1, 20, 100)
sps2 = 0

lastys = np.zeros_like(sps)

for i, sp in enumerate(sps):
    
    x0 = env.reset()

    for t in range(len(T)-1):
        state, _ = env.ret_state(sp)
        action = target_actor(state.reshape(1, -1)).numpy()

        if t == len(T)-2:
            lastys[i] = env(action)
        else:
            env(action)

    print(i)

plt.figure()
plt.plot(sps-lastys)

# plt.legend(loc=4, prop={'size': 15})
plt.show()

# %%
# Plotting OU process
ou_noise = OUActionNoise(np.zeros(1), std_deviation=.4)

noise = [ou_noise() for _ in range(10000)]
plt.figure(figsize=(10, 5))
plt.plot(noise)
plt.show()

# %%
from celluloid import Camera

fig = plt.figure()
camera = Camera(fig)

# plt.axis('off')
plt.ylim(bottom=-0, top=10)
plt.yticks([])
 
for row in history_smoothed:
    plt.plot(row, c="b")
    plt.plot(np.ones(249)*6.32, c="r", ls="--")
    camera.snap()
 
animation = camera.animate()
animation.save('celluloid_minimal.gif', writer = 'Pillow')
