# Implementation of REINFORCE policy gradient for LunarLander environment

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import gym
from collections import deque
import os
from scipy.io import savemat

env = gym.make('LunarLander-v2')
print("The state size is: ", env.observation_space)
print("The action size is : ", env.action_space.n)
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
print("Possible actions: ", possible_actions)

# hyper-parameters
act_space = 4                 # number of possible actions
obs_space = 8                # number of observation
total_iterations = 20000           # Total iterations
max_steps = 1000              # Max possible steps in an episode
n_experiences = 32           # number of episodes to calculate policy gradient
learning_rate = 3e-3            # learning rate for policy network
gamma = 0.99                   # Discounting rate
rew_buffer = deque(maxlen=100)  # buffer for saving recent 100 rewards

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# training or test
training = False  # if false, test the saved network


# %% function definitions
def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


input_lay = tf.placeholder(tf.float32, [None, obs_space])
act_picks = tf.placeholder(tf.float32, [None, act_space])
advantages = tf.placeholder(tf.float32, [None, ])
soft_max = mlp(input_lay, hidden_sizes=(128, 64, act_space), activation=tf.nn.relu, output_activation=tf.nn.softmax)
log_prob = tf.reduce_sum(tf.multiply(act_picks, tf.log(soft_max)), [1])
loss = -tf.reduce_mean(tf.multiply(log_prob, advantages))
train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# class definition
class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def clear_mem(self):
        self.buffer.clear()

    def length(self):
        return len(self.buffer)


memory = Memory(max_size=1000000)


# function definition
def predict_action(state_, actions_):
    act_dist = sess.run(soft_max, feed_dict={input_lay: state_.reshape(1, obs_space)})
    choice_ = np.random.choice(range(act_dist.shape[1]), p=act_dist.reshape(act_space, ))  # select one action
    action_ = actions_[choice_]  # one-hot encoding actions
    return action_, choice_


def discount_rewards(rewards_, gamma_):
    discounted_episode_rewards = np.zeros_like(rewards_)
    cumulative = 0.0
    for i in reversed(range(len(rewards_))):
        cumulative = cumulative * gamma_ + rewards_[i]
        discounted_episode_rewards[i] = cumulative
    return discounted_episode_rewards


saver = tf.train.Saver(max_to_keep=1)

reward_list = np.array([])  # save as mat file

best_mean_reward = -10000.0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if training:

        for iteration in range(total_iterations):
            for exp in range(n_experiences):  # collect set of trajectories (multiple episodes)

                state = env.reset()
                step = 0

                episode_rewards = []  # save all rewards for one episode

                while step < max_steps:  # steps

                    step += 1

                    choice, action = predict_action(state, possible_actions)
                    next_state, reward, done, _ = env.step(action)

                    if step == 1:
                        state_list = np.array(state, dtype=float).reshape((1, obs_space))
                        choice_list = np.array(choice, dtype=float).reshape((1, act_space))
                    else:
                        state_list = np.vstack((state_list, np.array(state, dtype=float).reshape((1, obs_space))))
                        choice_list = np.vstack((choice_list, np.array(choice, dtype=float).reshape((1, act_space))))
                    episode_rewards.append(reward)

                    if done:
                        next_state = np.zeros((obs_space,), dtype=np.int)
                        step = max_steps
                        total_reward = np.sum(episode_rewards)
                        reward_list = np.append(reward_list, total_reward)
                        rew_buffer.append(total_reward)
                        break
                    else:
                        state = next_state

                # end of one trajectory
                returns = discount_rewards(episode_rewards, gamma)
                for s in range(state_list.shape[0]):
                    memory.add((state_list[s, :], choice_list[s, :], returns[s]))

            # batch training from collected trajectories
            batch = memory.buffer
            states_mb = np.array([each[0] for each in batch], ndmin=2)
            choices_mb = np.array([each[1] for each in batch], ndmin=2)
            returns_mb = np.array([each[2] for each in batch])

            mean_reward = sum(rew_buffer)/len(rew_buffer)

            adv_mb = returns_mb.flatten()
            adv_mb = (adv_mb - np.mean(adv_mb)) / np.std(adv_mb)
            adv_mb = np.reshape(adv_mb, np.shape(returns_mb))

            _ = sess.run(train_opt, feed_dict={input_lay: states_mb, act_picks: choices_mb, advantages: adv_mb})
            memory.clear_mem()

            print('Episode: {}'.format(iteration * n_experiences),
                  'mean reward: {}'.format(mean_reward),
                  'best mean reward: {}'.format(best_mean_reward))

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                saver.save(sess, './check/my_model')
                savemat('./check/data.mat', {'reward_list': reward_list,
                                             'best_mean_reward': best_mean_reward})
                print("Model Saved")

    else:
        saver.restore(sess, tf.train.latest_checkpoint('./check/'))
        print("Model Loaded")

        step = 0
        state = env.reset()
        rew_sum = 0

        while step < max_steps:
            step += 1
            env.render()
            act_dist_ = sess.run(soft_max, feed_dict={input_lay: state.reshape(1, obs_space)})
            choice__ = np.random.choice(range(act_dist_.shape[1]), p=act_dist_.reshape(act_space, ))
            next_state, reward, done, _ = env.step(choice__)
            state = next_state
            rew_sum += reward

            if done:
                print('sum of rewards: ', rew_sum)
                break
