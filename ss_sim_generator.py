# %%
import matplotlib.pyplot as plt
import numpy as np
import control.matlab as ctrl
import scipy.signal as sig

# %%

def simulate(A, B, C, initial_state, input_sequence, time_steps, sampling_period):
    I = np.identity(A.shape[0])  # this is an identity matrix
    Ad = np.linalg.inv(I-np.dot(sampling_period, A))
    Bd = np.dot(np.dot(Ad, sampling_period), B)
    Xd = np.zeros(shape=(A.shape[0], B.shape[1], time_steps+1)) # 4x2
    Yd = np.zeros(shape=(C.shape[0], time_steps+1)) # 2

    for i in range(0, time_steps):
        if i == 0:
            Xd[:, [i]] = initial_state #.reshape(A.shape[0], B.shape[1], 1)
            Yd[:, [i]] = np.dot(C, initial_state)
            x = np.dot(Ad, initial_state)+np.dot(Bd, input_sequence[i])
        else:
            Xd[:, [i]] = x
            Yd[:, [i]] = np.dot(C, x)
            x = np.dot(Ad, x)+np.dot(Bd, input_sequence[i])
    Xd[:, [-1]] = x
    Yd[:, [-1]] = np.dot(C, x) # 2x4 . 
    return Xd, Yd

# %%
A = np.matrix([[-8.19602524e-01, -5.30764289e+00,  9.60996874e+00, -3.73225374e+00],
               [ 5.93765570e-03, -5.13378511e+00,  8.16602053e+00, -3.23781101e+00],
               [-1.13225282e-01, -1.16381820e+00,  1.74885985e+00, -8.34686690e-01],
               [ 2.20462904e-02, -2.32325063e-01,  3.93052884e-01, -4.71165499e-01]])
B = np.matrix([[ 0.86651717,  0.24843595],
               [ 0.58334683,  0.        ],
               [ 0.35637561, -0.8226274 ],
               [ 1.28363613,  0.7931689 ]])
C = [[-0.         , 0.64754102,  0.        , -0.03895053],
     [ 0.         , 0.        , -0.        , -1.17455932]]
D = np.matrix([[0., 0.],
               [0., 0.]])

W1 = ctrl.ss(A, B, C, D)
# W1 = ctrl.rss(4, 2, 2)

U = np.array([np.concatenate([np.ones(1), np.zeros(499)]).reshape(-1, 1), np.concatenate([np.zeros(1), np.zeros(499)]).reshape(-1, 1)]).reshape(500, 2)

Y1, T1, X1 = ctrl.lsim(W1, T=np.linspace(0, 50, 500), U=U)
# Y1, T1 , X1 = ctrl.lsim(W1, T=np.linspace(0, 100, 200), U=np.random.uniform(0, 50, size=(200, 2)))(500, 1)
plt.figure(figsize=(15, 9))
# plt.subplot(211)
plt.ylabel("yout")
plt.plot(T1, Y1)
plt.legend(["Вихід 1", "Вихід 2"], loc=4)
# plt.subplot(212)
# plt.ylabel("xout")
# plt.xlabel("T")
# plt.plot(T1, X1)
# plt.legend(["X1", "X2", "X3", "X4"], loc=4)
plt.show()

# %%

X1, Y1 = simulate(np.matrix(A), np.matrix(B), np.matrix(C), 
                 np.zeros(shape=(4, 1)), 
                 np.ones(shape=(100,)),
                 100, 
                 0.1
                 )

plt.plot(Y1.T)

plt.show()
# %%
A, B, C = W1.A, W1.B, W1.C
Ys = []
X_tmp = np.zeros(shape=(4, 2))
for i in range(200):
    X_tmp, Y_tmp = simulate(
                    np.matrix(A), np.matrix(B), np.matrix(C), 
                    X_tmp, 
                    np.ones(2),
                    1, 
                    0.1
                )
    Y_tmp = Y_tmp[-1][0]
    X_tmp = X_tmp[:, -1].reshape((4, 2))
    Ys.append(Y_tmp)

Ys = np.array(Ys)
Ys.shape
plt.plot(Ys)
plt.show()

# %%

"""Test PID algorithm."""
import scipy.signal as sig
# transfer function in s-space describes sys
tf = sig.tf2ss([10], [100, 10])
times = np.arange(1, 200, 5.0)
#step = 1 * np.ones(len(times))
# initialize PID
sysout = [0.0]
pidout = [0.0]
real_time = [0.0]
for time in times:
    real_time.append(time)
    t, sysout, xout = sig.lsim(tf, np.ones_like(real_time), real_time)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(real_time, sysout, 'r')
plt.show()

# %%
# u{ndarray}, sp -> generator -> y{ndarray} -> rating -> y_out, sp

# Environment
class Environment:
    @staticmethod
    def simulate(A, B, C, initial_state, input_sequence, time_steps, sampling_period):
        from numpy.linalg import inv
        I = np.identity(A.shape[0])  # this is an identity matrix
        Ad = inv(I-sampling_period*A)
        Bd = Ad*sampling_period*B
        Xd = np.zeros(shape=(A.shape[0], time_steps+1))
        Yd = np.zeros(shape=(C.shape[0], time_steps+1))

        for i in range(0, time_steps):
            if i == 0:
                Xd[:, [i]] = initial_state
                Yd[:, [i]] = C*initial_state
                x = Ad*initial_state+Bd*input_sequence[i]
            else:
                Xd[:, [i]] = x
                Yd[:, [i]] = C*x
                x = Ad*x+Bd*input_sequence[i]
        Xd[:, [-1]] = x
        Yd[:, [-1]] = C*x
        return Xd, Yd
        
    @staticmethod
    def reward(yt, sp, epsilon, c):
        # if abs(yt-sp) < epsilon:
        #     return c
        # else:
        return -abs(yt-sp)

    def __init__(self, A, B, C,
                  initial_state,
                  T=200, sample=0.1):
        self.A = A
        self.B = B
        self.C = C
        self.T = T
        self.sample = sample
        self.initial_state = initial_state
        self.reset()

    def __call__(self, u, sp):
        self.x_curr, self.y_curr = simulate(
            self.A, self.B, self.C, 
            self.x_curr, 
            u,
            1, 
            self.sample
        )
        self.y_curr = self.y_curr[-1][0]
        self.x_curr = self.x_curr[:, -1].reshape(4, 1)
        return [self.y_curr, sp]

    def reset(self):
        self.x_curr, self.y_curr = simulate(
            self.A, self.B, self.C, 
            self.initial_state, 
            np.ones(1),
            1, 
            self.sample
        )
        self.x_curr = self.x_curr[:, -1].reshape(4, 1)
        self.y_curr = self.y_curr[-1][0]
        return self.y_curr, 0.0


env = Environment(np.matrix(A), np.matrix(B), np.matrix(C), 
                   initial_state=np.zeros(shape=(4, 1)))
print(env.reset())
Ys = [env(np.ones(1)*500, 1) for i in range(200)]
Ys = np.array(Ys)

plt.plot(Ys)
plt.show()
# %%
# env = Environment(np.matrix(A), np.matrix(B), np.matrix(C), 
#                    initial_state=np.zeros(shape=(4, 1)))

env.reset()
Ys = []
Us = []
SPs = []
u = np.array(0.0).reshape(1, 1)

# for _ in range(10):
SP = np.random.uniform(1, 20)
for _ in range(200):
    state = env(u, SP)
    Ys.append(state[0])
    state = tf.expand_dims(tf.convert_to_tensor([Ys[-1], SP]), 0)
    u = tf.squeeze(actor_model(state)).numpy().reshape(1, 1)
    Us.append(u.item())
    SPs.append(SP)

plt.figure(figsize=(15,8))
plt.subplot(121)
plt.plot(Ys, label="y_out")
plt.plot(SPs, label="u_out")
plt.subplot(122)
plt.plot(Us)
plt.show()

# %%
# u{ndarray}, sp -> generator -> y{ndarray} -> rating -> y_out, sp

# Environment
class SlowEnvironment:       
    
    def __init__(self, A, B, C, D, T, p=5, delta=0.5, trust_time=5):
        self.sys = sig.StateSpace(A, B, C, D)
        
        self.p = p
        self.T = T

        self.input_size = self.sys.B.shape[1]
        self.output_size = self.sys.C.shape[0]

        self.delta = delta
        self.trust_time = trust_time

        self.x0 = None
        self.xout = None

        self.reset()

    def __call__(self, u):
        self.idx = self.idx+1
        tmp_i = self.idx+self.p

        self.U[tmp_i] = u
        _, yout, xout = sig.lsim(self.sys, U=self.U[self.p:tmp_i], T=self.T[:self.idx], X0=self.x0)
        try:
            self.Y[tmp_i] = yout[-1]
        except:
            self.Y[tmp_i] = yout.item()
        self.xout = xout[-1]
        return self.Y[tmp_i]


    def reset(self, x0=None):
        self.idx = 0
        self.ontarget_time = 0
        
        self.Y = np.zeros(shape=(self.T.shape[0]+self.p, self.sys.C.shape[0]), dtype=np.float32)
        self.U = np.zeros(shape=(self.T.shape[0]+self.p, self.sys.B.shape[1]), dtype=np.float32)
        
        self.x0 = x0
        return self.xout

    def ret_state(self, sp):
        # return <prev_ys, prev_actions, sps>, done
        es = np.array([sp], dtype=np.float32).reshape(-1, order='F')-self.Y[self.idx+1, :].reshape(-1)
        if np.all(es < self.delta):
            self.ontarget_time = self.ontarget_time+1 
        elif self.ontarget_time > 0:
            self.ontarget_time = 0
        return np.concatenate([self.U[self.idx:self.idx+self.p+1, :].reshape(-1, order='F'),
                               self.Y[self.idx:self.idx+self.p+1, :].reshape(-1, order='F'),
                               np.array([sp], dtype=np.float32).reshape(-1, order='F')]),\
                               self.ontarget_time > self.trust_time
    
    def ret_state_e(self, sp):
        # return <prev_ys, prev_actions, es>
        es = np.array([sp], dtype=np.float32).reshape(-1, order='F')-self.Y[self.idx+1, :].reshape(-1)
        return np.concatenate([self.U[self.idx:self.idx+self.p+1].reshape(-1, order='F'),
                               self.Y[self.idx-self.p:self.idx+1].reshape(-1, order='F'),
                               es])

    @staticmethod
    def reward_mae(state):
        # some realy ugly code here
        state = np.array(state)
        yt = state[env.input_size*(env.p+1):env.input_size*(env.p+1)+env.output_size*(env.p+1)].reshape(env.output_size, -1)[:, -1]
        sp = state[env.input_size*(env.p+1)+env.output_size*(env.p+1):]
        return -np.sum(np.abs(yt-sp))

    @staticmethod
    def reward_polar(state, prev_state):
        yt, sp = state
        prev_yt, prev_sp = prev_state
        try:
            return np.sum([0 if np.abs(prev_yt_i-sp_i) > np.sum(np.abs(yt_i-sp_i))\
                                else -1\
                                    for yt_i, prev_yt_i, sp_i in zip(yt, prev_yt, sp)]) 
        except:
            return np.sum([0 if np.abs(prev_yt_i-sp) > np.sum(np.abs(yt_i-sp))\
                                else -1\
                                    for yt_i, prev_yt_i in zip(yt, prev_yt)]) 



env = SlowEnvironment(W1.A, W1.B, W1.C, W1.D, np.linspace(0, 70, 201))
x0 = env.reset()
Ys = [env([1.0]) for i in range(200)]
Ys = np.array(Ys)
# Ys = np.stack(Ys[1:, 0])

plt.plot(np.linspace(0, 70, 200), Ys)
plt.show()
print(env.reset())
# %%
W1 = sig.StateSpace(W1.A, W1.B, W1.C, W1.D)
_, yout = sig.step(W1, T=np.linspace(0, 100, 200))
plt.plot(yout)
# %%
print(SlowEnvironment.reward_mae([Ys[10, 0], 5], [Ys[11, 0], 5]))
# %%
# Index last ys: state[env.input_size*(env.p+1):env.input_size*(env.p+1)+env.output_size*(env.p+1)].reshape(env.output_size, -1)[:, -1]
# Index sps: state[env.input_size*(env.p+1)+env.output_size*(env.p+1):]
# env.reset()
env.ret_state(sp=np.array([5]))
# %%
