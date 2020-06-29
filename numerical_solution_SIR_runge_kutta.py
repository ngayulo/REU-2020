#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:25:05 2020

@author: ngayulo
"""
import matplotlib.pyplot as plt
import numpy as np

#simple SIR
#d_S= -b*S*I
#d_I= b*S*I - c*I
#d_R= c*I

def d_S(b,S,I):
    return -b*S*I

def d_I(b,c,S,I):
    return b*S*I - c*I 

def d_R(c,I):
    return c*I

def runge_kutta(b,c, S_0, I_0, R_0, dt, t):
    t_arr=np.array(np.arange(0,t+1,dt))
    S=np.zeros(t_arr.shape[0])
    I=np.zeros(t_arr.shape[0])
    R=np.zeros(t_arr.shape[0])
    S[0]=S_0
    I[0]=I_0
    R[0]=R_0
   
    #implementing RK4
    for i in range (0,t_arr.shape[0]-1):
        k_1=d_S(b,S[i],I[i])
        k_2=d_S(b,S[i]+(dt/2)*k_1,I[i]+(dt/2)*k_1)
        k_3=d_S(b,S[i]+(dt/2)*k_2,I[i]+(dt/2)*k_2)
        k_4=d_S(b,S[i]+dt*k_3,I[i]+dt*k_3)
        
        S[i+1]=S[i]+(1/6)*dt*(k_1+2*k_2+2*k_3+k_4)
        
        k_1=d_I(b,c,S[i],I[i])
        k_2=d_I(b,c,S[i]+(dt/2)*k_1,I[i]+(dt/2)*k_1)
        k_3=d_I(b,c,S[i]+(dt/2)*k_2,I[i]+(dt/2)*k_2)
        k_4=d_I(b,c,S[i]+dt*k_3,I[i]+dt*k_3)
        
        I[i+1]=I[i]+(1/6)*dt*(k_1+2*k_2+2*k_3+k_4)
        
        k_1=d_R(c,I[i])
        k_2=d_R(c,I[i]+(dt/2)*k_1)
        k_3=d_R(c,I[i]+(dt/2)*k_2)
        k_4=d_R(c,I[i]+dt*k_3)
        
        R[i+1]=R[i]+(1/6)*dt*(k_1+2*k_2+2*k_3+k_4)
       
    return S,I,R, t_arr

    
S,I,R,t_arr=runge_kutta(0.001,0.1,499,1,0,0.0005,100)
plt.xlim(0,100)
plt.ylim(0,500)
plt.plot(t_arr,S, label='s')
plt.plot(t_arr,I, label='i')
plt.plot(t_arr,R, label='r')
plt.legend()
plt.show

print(S)
print(I)
print(R)
#import pandas as pd
#output_file={'Time':t_arr,'S(t)':S,'I(t)':I,'R(t)':R}
#df=pd.DataFrame(output_file,columns=['Time','S(t)','I(t)','R(t)'])
#df.to_csv('dataframe2.csv')
