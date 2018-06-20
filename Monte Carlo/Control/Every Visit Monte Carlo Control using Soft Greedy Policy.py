#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:34:43 2018

@author: amajidsinar


#LEFT = 0
#DOWN = 1
#RIGHT = 2
#UP = 3
"""
import time
start_time = time.time()

import gym
from collections import defaultdict
import numpy as np
import random

env = gym.make('FrozenLake-v0')

episodes = 100
timesteps = 500
discount = 1.0
nA = env.action_space.n
alpha = 1.0
gamma = 1.0

Q = defaultdict(lambda: np.zeros(nA))
return_count = defaultdict(float)
return_sum = defaultdict(float)

diamond = 20
pit = -10
punishment = -1

def soft_policy(epsilon=0.4):
    
    #1. Create the probability distribution
    p = np.ones(nA) * epsilon / nA 
    p[np.argmax(Q[state])] += 1 - epsilon
    #2. Pick random action based on the distribution
    arr = []
    cumsum = 0
    for i in p:
        cumsum += i
        arr.append(cumsum)
    
    roll_the_dice = random.random() * p.sum()
    for action, val in enumerate(arr):
        if roll_the_dice < val:
            return action

for e in range(episodes):
    state = env.reset()
    episode = []
    for t in range(timesteps):
        action = soft_policy()
        state,reward,done,_ = env.step(action)
        if done:
            if state == 15:
                episode.append((state,action,diamond,done))
            else:
                episode.append((state,action,pit,done))
            print('episode {} done after {} timesteps'.format(e,t+1))
            break;
        episode.append((state,action,punishment,done))
    state_action = set((value[0],value[1]) for i,value in enumerate(episode))
    for s,a in state_action:
        occurence = [i for i,value in enumerate(episode) if (value[0],value[1]) == (s,a)]
        for o in occurence:
            G = sum([val[2]*gamma**i for i,val in enumerate(episode[o:])])
            Q[s][a] = Q[s][a] + (alpha*(G - Q[s][a]))
                
q = [np.argwhere(np.max(Q[i])== Q[i]) for i in range(len(Q))]
best_action = dict(zip(range(len(q)),q))
    

running_time = time.time() - start_time
print("--- %s seconds ---" % (time.time() - start_time))