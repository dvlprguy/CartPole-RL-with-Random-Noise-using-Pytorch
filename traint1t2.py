# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:54:32 2020

@author: Anupreet Singh
"""

# Importing Libraries
from collections import deque
import numpy as np
import gym
import random
import os

# Using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense
from keras.models import Model, load_model
# np.random.seed(7)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

# list to store training logs
train_log=[]

# was getting some warning tried to solve using this
gym._gym_disable_underscore_compat = True

# defining neural network
# keras is used by default
# using a weight initializer helps the model train a little bit faster
# he_uniform worked really well for me
def NeuralNet(input_shape, action_space):
    # input layer
    X_input = Input(input_shape)
    # first fc layer
    X = Dense(50, input_shape=input_shape, activation="relu",
              kernel_initializer='he_uniform')(X_input)
    # second fc layer
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    # output fc layer
    X = Dense(action_space, activation="linear",
              kernel_initializer='he_uniform')(X)
    model = Model(inputs=X_input, outputs=X, name='CartPole DQN model')
    # RMSprop works far better than Adam as an optimizer in my case
    model.compile(loss="mse", optimizer=RMSprop(
        lr=0.0025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.summary()
    return model

# Agent Class for DQN
class AgentDQN:
    # initializer
    def __init__(self):
        # creating enviornment
        self.env = gym.make('CartPole-v1')
        self.env._gym_disable_underscore_compat = True
        # getting state space and action space
        self.statespace = self.env.observation_space.shape[0]
        self.actionspace = self.env.action_space.n
        # max episodes for which training will go on
        self.EPISODES = 1000
        # declaring memory of size 2000
        self.memory_dq = deque(maxlen=2000)

        # Hyperparams
        self.gamma = 0.95    
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.batch_size = 256
        self.train_start = 1000

        # Initializing Model
        self.model = NeuralNet(input_shape=(self.statespace,),
                           action_space=self.actionspace)
    
    # Function to store current state and observations in memory
    # Also updates the exploration rate
    def storeinmemory(self, state, action, reward, next_state, done):
        self.memory_dq.append((state, action, reward, next_state, done))
        if len(self.memory_dq) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    # Taking an action (random or using the nn)
    def take_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.actionspace)
        else:
            return np.argmax(self.model.predict(state))

    # training the nn using memory batches and targets
    def train_nn(self):
        if len(self.memory_dq) < self.train_start:
            return
        minibatch = random.sample(self.memory_dq, min(
            len(self.memory_dq), self.batch_size))
        state = np.zeros((self.batch_size, self.statespace))
        next_state = np.zeros((self.batch_size, self.statespace))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + \
                    self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    # Function to load model
    def load(self, name):
        self.model = load_model(name)

    # Function to save model
    def save(self, name):
        self.model.save(name)

    # Main code for running the agent
    def run_agent(self):
        # stores count of consec 500's
        # training stops at 10 consec 500's
        count_500_consec = 0
        last_500 = 0
        # Main for loop to simulate environment
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.statespace])
            done = False
            i = 0
            while not done:
                # self.env.render()
                action = self.take_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.statespace])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.storeinmemory(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e,
                                                                       self.EPISODES, i, self.epsilon))
                    train_log.append(i)
                    if i == 500:
                        if(last_500 == 1):
                            count_500_consec += 1
                        last_500 = 1
                        if(count_500_consec == 10):
                            print("Saving trained model as cartpole-dqnt1short.h5")
                            self.save("cartpole-dqnt1short.h5")
                            return
                    else:
                        last_500 = 0
                self.train_nn()
        self.save("cartpole-dqnt1short.h5")

    # Function to test agent on 100 runs
    def test_agent(self):
        sum=0
        self.load("cartpole-dqnt1short.h5")
        for e in range(100):
            state = self.env.reset()
            state = np.reshape(state, [1, self.statespace])
            done = False
            i = 0
            while not done:
                # self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.statespace])
                i += 1
                if done:
                    sum+=i
                    print("episode: {}/{}, score: {} ,avg_score: {}".format(e, 100, i,sum/100))
                    break


# main
if __name__ == "__main__":
    agent = AgentDQN()
    agent.run_agent()
    plt.plot(train_log)
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.show()
    agent.test_agent()
