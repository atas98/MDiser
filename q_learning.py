# %%
# Imports
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# %%
# Ornstein-Uhlenbeck process

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

# %%
# The Buffer class implements Experience Replay Memory.

num_states = 26
print("Size of State Space ->  {}".format(num_states))
num_actions = 2
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = 200.0
lower_bound = -200.0

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)

        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

# %%
# Models initialization

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return np.squeeze(legal_action)

# %%
# Initializing hyperparameters

ou_noise = OUActionNoise(np.zeros(1), std_deviation=.9)

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 10**-4
actor_lr = 10**-5

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 5000
# Discount factor for future rewards
gamma = 0.98
# Used to update target networks
tau = 10**-4


buffer = Buffer(50000, 128)

T = np.linspace(0, 50, 250)
env = SlowEnvironment(W1.A, W1.B, W1.C, W1.D, T, p=5, delta=0.1, trust_time=6)
# %%
# Trainig loop
# [x] TODO: Checkpoints
# [ ] TODO: Prioritized replay memory
# [x] TODO: Early episode break (5 timesteps within specified delta)

start_time = time.time()

# To store reward history of each episode
# ep_reward_list = []
# To store average reward history of last few episodes
# avg_reward_list = []

# Main training loop
for ep in range(total_episodes):

    env.reset()
    prev_state = np.zeros(num_states)
    episodic_reward = 0

    # Generate new sp
    sp = np.random.uniform(0.10, 10.0, size=env.input_size)

    for _ in range(len(T)-2):

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = np.reshape(policy(tf_prev_state, ou_noise), (2))
        # Recieve state and reward from environment.
        # ret_state -> Ys[:p]
        env(action)
        state, done = env.ret_state(sp)
        reward = env.reward_mae(state)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    avg_reward_list.append(avg_reward)

    if ep %100 == 0:
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        # Save target model if its reward is more than reward of prev saved model
    if ep %1000 == 0:
        target_actor.save(f'Models/q_learning/MIMO/actor_ep{ep}')
        target_critic.save(f'Models/q_learning/MIMO/critic_ep{ep}')


# Plotting graph
# Episodes versus Avg. Rewards
end_time = time.time() - start_time
print("Training time, s: ", end_time)
print("Training time for episode, s: ", end_time/total_episodes)

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

# %%
# target_actor.save(f'Models/q_learning/MIMO/target_actor')
# target_critic.save(f'Models/q_learning/MIMO/target_critic')

target_actor = tf.keras.models.load_model(f'Models/q_learning/SISO/target_actor', compile=False)
target_critic = tf.keras.models.load_model(f'Models/q_learning/MIMO/v2/target_critic', compile=False)
# %%
# saving training data
# avg_reward_list = np.array(avg_reward_list)
# with open('Models/q_learning/SISO/history.npy', 'wb') as f:
#     np.save(f, avg_reward_list)

# with open('Models/q_learning/SISO/system.pickle', 'wb') as f:
#     pickle.dump(W1, f)
# %%
# Load systems data 
with open('Models/q_learning/SISO/system.pickle', 'rb') as f:
    W1 = pickle.load(f)
# with open('Models/q_learning/MIMO/v2/history.npy', 'rb') as f:
#     avg_reward_list = np.load(f)
