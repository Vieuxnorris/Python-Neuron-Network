# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 20:25:48 2021

@author: vieuxnorris
"""
import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)
print('exponentiated values:')
print(exp_values)
norm_values = exp_values / np.sum(exp_values)
print('Normalized exponentiated values :')
print(norm_values)
print('sum of normalized values:', sum(norm_values))