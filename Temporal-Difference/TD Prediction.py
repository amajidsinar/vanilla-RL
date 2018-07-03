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
    episode = []
    for t in range(timesteps):
        action = env.action_space.sample()
        next_state,reward,done,_ = env.step(action)
        if done:
            if next_state == 15:
                episode.append((state,next_state,diamond,done))
            else:
                episode.append((state,next_state,pit,done))
            break
        episode.append((state,next_state,cost,done))
        state = next_state
    for val in episode:
        V[val[0]] += alpha * (val[2] + (gamma*(V[val[1]])) - V[val[0]])