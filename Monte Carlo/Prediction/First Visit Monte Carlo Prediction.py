#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:11:24 2018

@author: amajidsinar
"""

import time
start_time = time.time()

import gym
from collections import defaultdict

env = gym.make('FrozenLake-v0')

n_episode = 10000
n_timestep = 1000
discount = 1.0
alpha = 0.99

cost = -1
pit = -20
diamond = 20

# The final value function
V = defaultdict(float)

for e in range(n_episode):
    print("Episode {}".format(e))
    episode = []
    state = env.reset()
    #generate an episode
    for t in range(n_timestep):
        action = env.action_space.sample()
        nextState, reward, done, _ = env.step(action)
        if done:
            if state == 15:
                episode.append((state, nextState, diamond, done))
            else:
                episode.append((state, nextState, pit, done))
            print("Episode finish after {} timesteps".format(t+1))
            break
        episode.append((state, nextState, cost, done))
        state = nextState
    #store the unique value of state (without repeating)
    states = set(e[0] for e in episode)
    #for each unique state
    for s in states:
        # store the first occurence index of an episode
        first_occurence = [i for i,value in enumerate (episode) if value[0] == s][0]
        G = sum([(value[2]*discount**i) for i,value in enumerate(episode[first_occurence:])])
        V[s] = V[s] + (alpha*(G - V[s]))

        
running_time = time.time() - start_time
print("--- %s seconds ---" % (time.time() - start_time))
    