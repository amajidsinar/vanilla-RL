#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 00:35:12 2018

@author: amajidsinar

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

"""


import time
start_time = time.time()

import gym
from collections import defaultdict
import numpy as np
import random

env = gym.make('FrozenLake-v0')



episodes = 100000
timesteps = 500
discount = 1.0
return_sum = defaultdict(float)
return_count = defaultdict(float)
alpha = 1
nA = env.action_space.n

cost = -1
pit = -10
diamond = 20


Q = defaultdict(lambda: np.zeros(nA))


def soft_policy(epsilon=0.3):    
    """
    Soft policy is basically setting the maximum
    """
    #1. Create the probability distribution
    #set all index to have probability of epsilon / nA
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - epsilon)
    
    #2.Take random action based on probability distribution
    arr = []
    cumsum = 0
    for i in A:
        cumsum += i
        arr.append(cumsum)
    rnd = random.random()*cumsum
    for i,val in enumerate(arr):
        if rnd < val:
            return i
    
    
for e in range(episodes):
    episode = []
    state = env.reset()
    for t in range(timesteps):
        action = soft_policy()
        state,reward,done,_ = env.step(action)
        if done:
            if state == 15:
                episode.append((state,action,diamond,done))
            else:
                episode.append((state,action,pit,done))
            print('Episode {} ends after {} timesteps.'.format(e,t+1))
            break
        episode.append((state,action,cost,done))
        
    #look for unique sa pair
    unique_sa = set([(value[0],value[1]) for i,value in enumerate(episode)])
    #enumerate for each sa pair
    for s,a in unique_sa:
        #find the first index of s,a
        first_occurence = [i for i,val in enumerate(episode) if (val[0],val[1]) == (s,a)][0]
        #sum the reward from the first occurence
        G = sum([val[2] for i,val in enumerate(episode[first_occurence:])])
        Q[s][a] = Q[s][a] + (alpha * (G - Q[s][a]))
        
#q = [np.max(Q[i])== Q[i] for i in range(len(Q))]
q = [np.argwhere(np.max(Q[i])== Q[i]) for i in range(len(Q))]
best_action = dict(zip(range(len(q)),q))



#val = np.array(val).reshape(4,4)