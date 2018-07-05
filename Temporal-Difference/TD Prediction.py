#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 18:23:08 2018

@author: amajidsinar
#LEFT = 0
#DOWN = 1
#RIGHT = 2
#UP = 3
"""

import gym
from collections import defaultdict

env = gym.make('FrozenLake-v0')

episodes = 1000
timesteps = 100
alpha = 0.1
gamma = 0.9

cost = -1
diamond = 20
pit = -20

V = defaultdict(float)

for e in range(episodes):
    state = env.reset()
    for t in range(timesteps):
        action = env.action_space.sample()
        next_state,reward,done,_ = env.step(action)
        #set different reward based on the next state
        if next_state == 15:
            reward = diamond
        elif next_state==5 or next_state==7 or next_state==11 or next_state==12:
            reward = pit
        else:
            reward = cost
        #take the V in the next step as the prediction
        sample = gamma * V[next_state]
        V[state] += alpha * (reward + sample - V[state])
        if done:
            break
        state = next_state
