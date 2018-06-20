#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 01:24:02 2018

@author: amajidsinar
"""

import gym
import random
import numpy as np
from collections import defaultdict

env = gym.make('FrozenLake-v0')
episodes = 1
timesteps = 100
nA = env.action_space.n

return_sum = defaultdict(float)
return_count = defaultdict(float)

Q = defaultdict(lambda: np.zeros(nA))

