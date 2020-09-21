#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:11:56 2020

@author: ngayulo
"""

#sigmoid function, f(t)=1/(1+e^(-k(t-1/2)))
#let k= 12

import matplotlib.pyplot as plt
import numpy as np

n=int(1/0.01)
t = np.linspace(0, 1, n)
y = np.zeros(t.shape[0])
i=0
for k in t:
   y[i]=1/(1+np.exp(-20*(k-0.5)))
   i=i+1
   
plt.plot(t,y)
plt.show()
 
print(y)

