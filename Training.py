# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 22:40:39 2021

@author: vieuxnorris
"""

from nnfs.datasets import spiral_data

import numpy as np
import nnfs

class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


nnfs.init()

X, y = spiral_data(samples=20, classes=3)

dense1 = Layer_Dense(2, 3)

dense1.forward(X)

print(dense1.output[:5])