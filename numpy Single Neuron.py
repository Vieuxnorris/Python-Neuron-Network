# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:40:55 2021

@author: vieuxnorris
"""

import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

outputs = np.dot(weights, inputs) + bias

print(outputs)