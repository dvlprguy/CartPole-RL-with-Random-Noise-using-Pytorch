# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:32:22 2020

@author: Anupreet Singh
"""

""" IMPORTANT: CHANGE CONTENT IN CARTPOLE.PY BEFORE RUNNING THIS
    This is final model loader for all 3 tasks.
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

gym._gym_disable_underscore_compat = True


class AgentDQN:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.env._gym_disable_underscore_compat = True
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
    def load(self, name):
        self.model = load_model(name)
        
    def test(self,model_name):
        sum=0
        self.load(model_name)
        self.model.summary()
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
                

if __name__=="__main__":
    agent=AgentDQN()
    model_name1="./models/cartpole-dqnt1short.h5"
    agent.test(model_name1)