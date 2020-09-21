#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 19:59:17 2020

@author: ngayulo
"""


import matplotlib.pyplot as plt
import numpy as np

#dx=(a+u(t))xdt+ b*x*dW 

dt = .001  # Time step.
T = 1.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0, T, n)  # Vector of times.

def sigmoid(t):
    return 1/(1+np.exp(-20*(t-0.5)))
        
def wiener_process(dt,T):
    W0 = [0]
    n=int(T/dt)
    # simulate the increments by normal random variable generator
    increments = np.random.normal(0, 1*np.sqrt(dt),n)
    W = W0 + list(np.cumsum(increments))
    return W

W=wiener_process(dt,T)

def numerical(x_0, a, b,t,dt,W):
    x= np.zeros(t.shape[0])
    x[0]=x_0
    for i in range (0,t.shape[0]-1,1):
        x[i+1]=x[i]+[(a+sigmoid(t[i]))*x[i]*dt]+[b*x[i]*(W[i+1]-W[i])]
        #print(x[i+1])
    return x

x=numerical(0.1,0.5,0.5,t,dt,W)
print(x[1:5])

def integrated_sigmoid(t):
    k1= (-1/20)*np.log(1+np.exp(-20*(t-0.5)))
    k2= (-1/20)*np.log(1+np.exp(10))
    return k1-k2

def analytic(y_0,a,b,t,W):
    y=np.zeros(t.shape[0])
    y[0]=y_0
    for i in range (1,t.shape[0],1):
        y[i]=y_0*np.exp((a-0.5*np.power(b,2))*t[i]+ integrated_sigmoid(t[i])+b*W[i])      
    return y

y=analytic(0.1,0.5,0.5,t,W)

#plt.figure(1)
plt.plot(t,x, color='b', label='Numerical Solution', linewidth=0.5)
#plt.figure(2)
plt.plot(t,y, color='r', label='Exact Solution', linewidth=0.5)
plt.legend()
plt.show()
