#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:29:37 2018

@author: amajidsinar
"""

import time
start_time = time.time()

import gym
from collections import defaultdict

env = gym.make('FrozenLake-v0')


episodes = 100
timesteps = 500
discount = 1.0
return_sum = defaultdict(float)
return_count = defaultdict(float)
V = defaultdict(float)
alpha = 0.99

cost = -1
diamond = 20
pit = -10



for e in range(episodes):
    state = env.reset()
    episode = []
    for t in range(timesteps):
        next_state,reward,done,_ = env.step(env.action_space.sample()) 
        if done:
            if next_state == 15:
                episode.append((state,next_state,diamond,done))
            else:
                episode.append((state,next_state,pit,done))
            print('episode {} done after {} timesteps'.format(e, t+1))
            break;
        episode.append((state,next_state,cost,done))
        state = next_state
    for i,val in enumerate(episode):
        G = sum([j[2]*discount**i for j in episode[i:]])
        V[val[0]] += alpha * (G - V[val[0]])
            
                
                
    