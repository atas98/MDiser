import matplotlib.pyplot as plt
import numpy as np

# %%

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

# %%
A = np.array(
    [[-3.73845774,  0.91887332, -0.98887796, -1.93489807],
     [-2.99520481, -0.06474333, -0.41704245, -3.11724574],
     [ 2.62958444, -0.31955758, -0.64564190,  3.26552020],
     [ 0.04782102, -0.67961865, -0.69458023, -2.54359985]])
B = np.array(
    [[-0.20367168],
     [ 0.        ],
     [-0.33708403],
     [-1.12758918]])

C = np.array([[ 0.25124848, -1.03273692,  0.        , -1.4901019 ]])

W1 = ctrl.ss(A, B, C, [[0]])

Y1, T1 , X1 = ctrl.lsim(W1, T=np.linspace(0, 100, 200), U=np.ones_like(np.linspace(0, 100, 200)))
plt.plot(T1, Y1)
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
Ys = []
X_tmp = np.zeros(shape=(4, 1))
for i in range(200):
    X_tmp, Y_tmp = simulate(
                    np.matrix(A), np.matrix(B), np.matrix(C), 
                    X_tmp, 
                    np.ones(1),
                    1, 
                    0.1
                )
    Y_tmp = Y_tmp[-1][0]
    X_tmp = X_tmp[:, -1].reshape((4, 1))
    Ys.append(Y_tmp)

Ys = np.array(Ys)
Ys.shape
plt.plot(Ys)
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
A = W1.A
B = W1.B
C = W1.C
# %%
A
# %%
