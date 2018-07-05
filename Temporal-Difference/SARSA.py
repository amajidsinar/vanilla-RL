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

episodes = 100000
timesteps = 100

diamond = 100
pit = -15
cost = -1

nA = env.action_space.n

Q = defaultdict(lambda:np.zeros(nA))

alpha = 0.9
gamma = 0.9

#define the soft policy function
def soft_policy(Q,epsilon=0.3):
    #create the probability distribution
    q = np.ones(nA) * epsilon / nA
    q[np.argmax(Q)] += 1 - epsilon
    
    roll_the_dice = random.random()
    
    #self explanatory
    cumsum = []
    for i,val in enumerate(q):
        cumsum.append(sum([j for j in q[:i+1]]))
    
    for i,val in enumerate(cumsum):
        if roll_the_dice < val:
            return i

for e in range(episodes):
    state = env.reset()
    for t in range(timesteps):
        #take action a to land state s'
        action = soft_policy(Q[state],0.3)
        next_state,reward,done,_ = env.step(action)
        #set different reward based on the next state
        if next_state == 15:
            reward = diamond
        elif next_state==5 or next_state==7 or next_state==11 or next_state==12:
            reward = pit
        else:
            reward = cost
        #from state s' pick the a' but dont take the move yet
        next_action = soft_policy(Q[next_state],0.3)
        #estimate the current value by sampling the next step or how much we would get if we take action a'
        #in the next state
        sample = gamma * Q[next_state][next_action]
        Q[state][action] += alpha * (reward + sample - Q[state][action])
        if done:
            break;
        state = next_state
        
