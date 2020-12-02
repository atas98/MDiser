# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# %%
# Step response
X = np.linspace(0, 70, 200)

env.reset()
Ys10 = [env([1.0, 0.0]) for i in range(200)]
Ys10 = np.array(Ys10)
env.reset()
Ys01 = [env([0.0, 1.0]) for i in range(200)]
Ys01 = np.array(Ys01)
env.reset()
Ys11 = [env([1.0, 1.0]) for i in range(200)]
Ys11 = np.array(Ys11)

fig = plt.figure(tight_layout=True, figsize=(12, 8), dpi=250)
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[0, :])
plt.plot(X, Ys11)
ax.set_ylabel('yout')
ax.set_xlabel('T')
ax.set_title("Вхід[1]=1; Вхід[2]=1")
ax.legend(["Вихід 1", "Вихід 2"])

ax = fig.add_subplot(gs[1, 0])
ax.plot(X, Ys10)
ax.set_ylabel("yout")
ax.set_xlabel("T")
ax.set_title("Вхід[1]=1; Вхід[2]=0")
ax.legend(["Вихід 1", "Вихід 2"])

ax = fig.add_subplot(gs[1, 1])
ax.plot(X, Ys01)
ax.set_ylabel("yout")
ax.set_xlabel("T")
ax.set_title("Вхід[1]=0; Вхід[2]=1")
ax.legend(["Вихід 1", "Вихід 2"])

fig.align_labels()

plt.show()

# %%
# Impulse response
X = np.linspace(0, 70, 200)

Ys10, Ys01, Ys11 = [], [], []
env.reset()
for t in range(200):
    if t == 0:
        Ys10.append(env([1.0, 0.0]))
    else:
        Ys10.append(env([0.0, 0.0]))
Ys10 = np.array(Ys10)

env.reset()
for t in range(200):
    if t == 0:
        Ys01.append(env([0.0, 1.0]))
    else:
        Ys01.append(env([0.0, 0.0]))
Ys01 = np.array(Ys01)

env.reset()
for t in range(200):
    if t == 0:
        Ys11.append(env([1.0, 1.0]))
    else:
        Ys11.append(env([0.0, 0.0]))
Ys11 = np.array(Ys11)

fig = plt.figure(tight_layout=True, figsize=(12, 8), dpi=250)
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[0, :])
plt.plot(X, Ys11)
ax.set_ylabel('yout')
ax.set_xlabel('T')
ax.set_title("Вхід[1]=1, 0..; Вхід[2]=1, 0..")
ax.legend(["Вихід 1", "Вихід 2"])

ax = fig.add_subplot(gs[1, 0])
ax.plot(X, Ys10)
ax.set_ylabel("yout")
ax.set_xlabel("T")
ax.set_title("Вхід[1]=1, 0..; Вхід[2]=1, 0..")
ax.legend(["Вихід 1", "Вихід 2"])

ax = fig.add_subplot(gs[1, 1])
ax.plot(X, Ys01)
ax.set_ylabel("yout")
ax.set_xlabel("T")
ax.set_title("Вхід[1]=1, 0..; Вхід[2]=1, 0..")
ax.legend(["Вихід 1", "Вихід 2"])

fig.align_labels()

plt.show()
# %%
# Harmonic response
X = np.linspace(0, 10, 200)

env.reset()
Ys10 = [env([np.sin(0.1*i), 0.0]) for i in range(200)]
Ys10 = np.array(Ys10)
env.reset()
Ys01 = [env([0.0, np.sin(0.1*i)]) for i in range(200)]
Ys01 = np.array(Ys01)
env.reset()
Ys11 = [env([np.sin(0.1*i), np.sin(0.1*i)]) for i in range(200)]
Ys11 = np.array(Ys11)

fig = plt.figure(tight_layout=True, figsize=(12, 8), dpi=250)
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[0, :])
plt.plot(X, Ys11)
ax.set_ylabel('yout')
ax.set_xlabel('T')
ax.set_title("Вхід[1]=1; Вхід[2]=1")
ax.legend(["Вихід 1", "Вихід 2"])

ax = fig.add_subplot(gs[1, 0])
ax.plot(X, Ys10)
ax.set_ylabel("yout")
ax.set_xlabel("T")
ax.set_title("Вхід[1]=1; Вхід[2]=0")
ax.legend(["Вихід 1", "Вихід 2"])

ax = fig.add_subplot(gs[1, 1])
ax.plot(X, Ys01)
ax.set_ylabel("yout")
ax.set_xlabel("T")
ax.set_title("Вхід[1]=0; Вхід[2]=1")
ax.legend(["Вихід 1", "Вихід 2"])

fig.align_labels()

plt.show()
# %%
# History plotting

plt.figure(figsize=(15, 5))

plt.subplot(121)
plt.plot(avg_reward_list)
plt.title("Середня винагорода за весь час")
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")

plt.subplot(122)
plt.title("Середня винагорода за останні 15 тисяч епізодів")
plt.plot(np.arange(5000, 20000), avg_reward_list[5000:])
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

# %%
# Controlled response

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


fig = plt.figure(tight_layout=True, figsize=(12, 8), dpi=250)
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[0, :])
ax.plot(np.linspace(0, 50, 248), Ys)
ax.plot(np.linspace(0, 50, 248), np.ones(248)*100, "r--")
ax.set_title("Виходи процесу")
ax.set_ylabel("Y")
ax.set_xlabel("T")
ax.legend(["Вихід 1", "Вихід 2", "Завдання"])

ax = fig.add_subplot(gs[1, 0])
ax.plot(np.linspace(0, 50, 249), np.concatenate([np.zeros(1), Us[:, 0, 0]]))
ax.set_ylabel("u")
ax.set_xlabel("T")
ax.legend(["Вхід 1"])

ax = fig.add_subplot(gs[1, 1])
ax.plot(np.linspace(0, 50, 249), np.concatenate([np.zeros(1), Us[:, 0, 1]]))
ax.set_ylabel("u")
ax.set_xlabel("T")
ax.legend(["Вхід 2"])

fig.align_labels()

plt.show()

# %%
# Target value changes over-time

Ys = []
Us = []
SPs = []

x0 = None

x0 = env.reset(x0=x0)
for t in range(len(T)-2):
    if t%50==0:
        sp = np.array(np.random.randint(10, 100, size=2)) 
        SPs.append(sp)

    state, _ = env.ret_state(sp)
    action = target_actor(state.reshape(1, -1)).numpy()
    Ys.append(env(action))
    Us.append(action)

Us = np.array(Us)
Ys = np.array(Ys)
SPs = np.array(SPs)


fig = plt.figure(tight_layout=True, figsize=(12, 8), dpi=250)
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[0, 0])
ax.plot(np.linspace(0, 50, 248), np.repeat(SPs[:, 0], 50)[:-2], "r--")
ax.plot(np.linspace(0, 50, 248), Ys[:, 0])
ax.set_ylabel("Y1")
ax.set_xlabel("T")
ax.legend(["Завдання", "Вихід 1"])

ax = fig.add_subplot(gs[0, 1])
ax.plot(np.linspace(0, 50, 248), np.repeat(SPs[:, 1], 50)[:-2], "r--")
ax.plot(np.linspace(0, 50, 248), Ys[:, 1])
ax.set_ylabel("Y2")
ax.set_xlabel("T")
ax.legend(["Завдання", "Вихід 2"])

ax = fig.add_subplot(gs[1, 0])
ax.plot(np.linspace(0, 50, 249), np.concatenate([np.zeros(1), Us[:, 0, 0]]))
ax.set_ylabel("u1")
ax.set_xlabel("T")
ax.legend(["Вхід 1"])

ax = fig.add_subplot(gs[1, 1])
ax.plot(np.linspace(0, 50, 249), np.concatenate([np.zeros(1), Us[:, 0, 1]]))
ax.set_ylabel("u2")
ax.set_xlabel("T")
ax.legend(["Вхід 2"])

fig.align_labels()

plt.show()
# %%
# Target value is curved

Ys = []
Us = []
SPs = []

x0 = None

x0 = env.reset(x0=x0)
for t in range(len(T)-2):
    sp = np.array([np.sin(t*0.1)*3+10, np.cos(t*0.1)*3+10]) 
    SPs.append(sp)

    state, _ = env.ret_state(sp)
    action = target_actor(state.reshape(1, -1)).numpy()
    Ys.append(env(action))
    Us.append(action)

Us = np.array(Us)
Ys = np.array(Ys)
SPs = np.array(SPs)


fig = plt.figure(tight_layout=True, figsize=(12, 8), dpi=250)
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[0, 0])
ax.plot(np.linspace(0, 50, 248), SPs[:, 0], "r--")
ax.plot(np.linspace(0, 50, 248), Ys[:, 0])
ax.set_ylabel("Y1")
ax.set_xlabel("T")
ax.legend(["Завдання", "Вихід 1"])

ax = fig.add_subplot(gs[0, 1])
ax.plot(np.linspace(0, 50, 248), SPs[:, 1], "r--")
ax.plot(np.linspace(0, 50, 248), Ys[:, 1])
ax.set_ylabel("Y2")
ax.set_xlabel("T")
ax.legend(["Завдання", "Вихід 2"])

ax = fig.add_subplot(gs[1, 0])
ax.plot(np.linspace(0, 50, 249), np.concatenate([np.zeros(1), Us[:, 0, 0]]))
ax.set_ylabel("u1")
ax.set_xlabel("T")
ax.legend(["Вхід 1"])

ax = fig.add_subplot(gs[1, 1])
ax.plot(np.linspace(0, 50, 249), np.concatenate([np.zeros(1), Us[:, 0, 1]]))
ax.set_ylabel("u2")
ax.set_xlabel("T")
ax.legend(["Вхід 2"])

fig.align_labels()

plt.show()
# %%
# Plotting systems form different learning episodes

Ys = []
Us = []
SPs = []

x0 = None

x0 = env.reset()
for t in range(len(T)-1):
    sp = [10, 10]
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
plt.subplot(211)
plt.title("Вихід 1")
plt.ylabel("Y")
plt.xlabel("T")
plt.plot(np.linspace(0, 50, 249)[:100], Ys[:100, 0], 'b', linewidth=1.5, label="Кінець навчання")
plt.plot(np.linspace(0, 50, 249)[:100], SPs[:100, 0], "r--", label="Завдання")

ep_models_ys = np.zeros(shape=(4, 249, 2))


directory = r'Models/q_learning/MIMO/v1'
for i, model_dir in enumerate(os.scandir(directory)):
    if model_dir.is_dir() and "target" not in model_dir.name and "actor" in model_dir.name:
        model = tf.keras.models.load_model(model_dir.path, compile=False)
        ep = int(model_dir.name[8:])
        linewidth = 1
        alpha = 1


        x0 = env.reset()
        sp = [10, 10]

        for t in range(len(T)-1):
            state, _ = env.ret_state(sp)
            action = model(state.reshape(1, -1)).numpy()
            ep_models_ys[i, t] = env(action)
        
        plt.plot(np.linspace(0, 50, 249)[:100], ep_models_ys[i, :100, 0], 
                 label=f"Епізод {int(ep/2500*5000)}")

plt.legend(loc=4, prop={'size': 15})
 

plt.subplot(212)
plt.title("Вихід 2")
plt.ylabel("Y")
plt.xlabel("T")
plt.plot(np.linspace(0, 50, 249)[:100], Ys[:100, 1], 'b', linewidth=1.5, label="Кінець навчання")
plt.plot(np.linspace(0, 50, 249)[:100], SPs[:100, 1], "r--", label="Завдання")
for t, ys in enumerate(ep_models_ys):
    plt.plot(np.linspace(0, 50, 249)[:100], ep_models_ys[t, :100, 1], linewidth=linewidth, alpha=alpha, label=f"Епізод {t*5000}")
 

plt.legend(loc=4, prop={'size': 15})
plt.show()
# %%
