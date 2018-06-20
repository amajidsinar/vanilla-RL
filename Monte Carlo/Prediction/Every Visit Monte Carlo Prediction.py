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


episodes = 100000
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
        nextState,reward,done,_ = env.step(env.action_space.sample()) 
        if done:
            if nextState == 15:
                episode.append((state,nextState,diamond,done))
            else:
                episode.append((state,nextState,pit,done))
            print('episode {} done after {} timesteps'.format(e, t+1))
            break;
        episode.append((state,nextState,cost,done))
        state = nextState
    states = set(value[0] for i,value in enumerate(episode))
    for s in states:
        occurence = [i for i,value in enumerate(episode) if value[0]==s]
        for o in occurence:
            G = sum([value[2]*discount**i for i,value in enumerate(episode[o:])])
            V[s] = V[s] + (alpha*(G-V[s]))
    
            
                
                
    