#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 04:53:44 2018

@author: amajidsinar

#LEFT = 0
#DOWN = 1
#RIGHT = 2
#UP = 3
"""

import gym
from collections import defaultdict
import numpy as np

env = gym.make('FrozenLake-v0')
nS = env.observation_space.n
nA = env.action_space.n

v = defaultdict(lambda: np.zeros(nS))
policy = defaultdict(lambda: np.ones(nA) * 0.25)
vNext = defaultdict(lambda: np.zeros(nA))

pit = -20
diamond = 20
cost = -1

iterations = 100
delta = 0.0005
transitionProbability = 0.3333333333
gamma = 1 

#find the next possible states
a = np.arange(0,16).reshape(4,4)
up = np.vstack((a[0],a[:3]))
right = np.column_stack((a[:,1:],a[:,3:]))
down = np.vstack((a[1:],a[3:]))
left = np.column_stack((a[:,:1],a[:,:3]))
nextPossibleStates = np.stack((left,down,right,up)).reshape(4,1,16)


for i in range(1,iterations):
    #Policy evaluation (prediction)
    for s in range(nS):
        if s == 5 or s == 7 or s == 11 or s == 12 or s==15:
            transitionProbability = 0
#            break;
        else:
            transitionProbability = 0.33333333
        for a in range(nA):
            nextState = nextPossibleStates[:,:,s][a]
            if nextState == 5 or nextState == 7 or nextState == 11 or nextState == 12:
                reward = pit
            elif nextState == 15:
                reward = diamond
            else:
                reward = cost
            vNext[s][a] = v[i-1][nextState]  
            v[i][s] += policy[s][a] * transitionProbability * (reward + (gamma*vNext[s][a]))
            #Policy improvement (control)
            exps = np.exp(vNext[s]-np.max(vNext[s]))
            policy[s] = exps / np.sum(exps)
            
#policy extraction
bestAction = [np.argwhere(policy[i] == np.max(policy[i])) for i in range(len(policy))]
