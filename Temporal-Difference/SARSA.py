#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 22:24:06 2018

@author: amajidsinar

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
"""

import gym
from collections import defaultdict
import numpy as np
import random

env = gym.make('FrozenLake-v0')

episodes = 100
timesteps = 100

diamond = 20
pit = -20
cost = -1

nA = env.action_space.n

Q = defaultdict(lambda:np.zeros(nA))

alpha = 0.9
gamma = 0.9


def soft_policy(Q,epsilon=0.3):
    q = np.ones(nA) * epsilon / nA
    q[np.argmax(Q)] += 1 - epsilon
    
    roll_the_dice = random.random()
    
    cumsum = []
    for i,val in enumerate(q):
        cumsum.append(sum([j for j in q[:i+1]]))
    
    for i,val in enumerate(cumsum):
        if roll_the_dice < val:
            return i

for e in range(episodes):
    state = env.reset()
    episode = []
    for t in range(timesteps):
        action = soft_policy(Q[state])
        next_state,reward,done,_ = env.step(action)
        if done:
            if next_state == 15:
                episode.append((state,action,next_state,diamond,done))
            else:
                episode.append((state,action,next_state,pit,done))
            break
        episode.append((state,action,next_state,cost,done))
        state = next_state
    for i,val in zip(range(len(episode)-1),episode):
        sample = gamma * Q[episode[i+1][0]][episode[i+1][1]]
        Q[val[0]][val[1]] += alpha * (val[3] + sample - Q[val[0]][val[1]]) 
        
