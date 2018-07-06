#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 21:32:36 2018

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

diamond = 100
pit = -15
cost = -1

nA = env.action_space.n

Q1 = defaultdict(lambda:np.zeros(nA))
Q2 = defaultdict(lambda:np.zeros(nA))

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
    Q1[0] = np.zeros(nA)
    Q2[0] = np.zeros(nA)
    for t in range(timesteps):
        Q_combined = list(zip([val for val in Q1[state]],[val for val in (Q2[state])]))
        Q_combined = [i[0]+i[1] for i in Q_combined]
        #take action a to land state s'
        action = soft_policy((Q_combined),0)
        next_state,reward,done,_ = env.step(action)
        #set different reward based on the next state
        if next_state == 15:
            reward = diamond
        elif next_state==5 or next_state==7 or next_state==11 or next_state==12:
            reward = pit
        else:
            reward = cost
        #roll the dice
        roll_the_dice = random.random()
        if roll_the_dice < 0.5:
            #from state s' pick the a' but dont take the move yet
            next_action = Q1[next_state].argmax()
            #estimate the current value by sampling the next step or how much we would get if we take action a'
            #in the next state
            sample = gamma * Q2[next_state][next_action]
            Q1[state][action] += alpha * (reward + sample - Q1[state][action])
        else:
            #from state s' pick the a' but dont take the move yet
            next_action = Q2[next_state].argmax()
            #estimate the current value by sampling the next step or how much we would get if we take action a'
            #in the next state
            sample = gamma * Q1[next_state][next_action]
            Q2[state][action] += alpha * (reward + sample - Q2[state][action])
        if done:
            break;
        state = next_state
        