# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 01:07:06 2020

@author: Anupreet Singh
"""




from collections import deque
import numpy as np
import gym
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense
from keras.models import Model, load_model
# np.random.seed(7)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

train_log=[]

gym._gym_disable_underscore_compat = True

def NeuralNet(input_shape, action_space):
    X_input = Input(input_shape)
    X = Dense(512, input_shape=input_shape, activation="relu",
              kernel_initializer='he_uniform')(X_input)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(action_space, activation="linear",
              kernel_initializer='he_uniform')(X)
    model = Model(inputs=X_input, outputs=X, name='CartPole DQN model')
    model.compile(loss="mse", optimizer=RMSprop(
        lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.summary()
    return model


class AgentDQN:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.env._gym_disable_underscore_compat = True
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 256
        self.train_start = 1000

        self.model = NeuralNet(input_shape=(self.state_size,),
                           action_space=self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(
            len(self.memory), self.batch_size))
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
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

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def run(self):
        count_500_consec = 0
        last_500 = 0
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
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
                            print("Saving trained model as cartpole-dqn.h5")
                            self.save("cartpole-dqnt3short.h5")
                            return
                    else:
                        last_500 = 0
                self.replay()
        self.save("cartpole-dqnt3short.h5")

    def test(self):
        sum=0
        self.load("cartpole-dqnt3short.h5")
        for e in range(100):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    sum+=i
                    print("episode: {}/{}, score: {} ,avg_score: {}".format(e, 100, i,sum/100))
                    break


if __name__ == "__main__":
    agent = AgentDQN()
    agent.run()
    plt.plot(train_log)
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.show()
    agent.test()
