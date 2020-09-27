#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:25:05 2020

@author: ngayulo
"""
import matplotlib.pyplot as plt
import numpy as np

#simple SIR
#d_S= -b*S*I - a*S
#d_I= b*S*I - c*I
#d_R= c*I +a*S

#define the differentials
def d_S(a,b,S,I):
    return -a*S-b*S*I

def d_I(b,c,S,I):
    return b*S*I - c*I 

def d_R(a,c,S,I):
    return c*I+a*S

#we define the function for the rugge-kutta with the inputs listed below:
#a,b,c, are the parameters of the differentials
#S_0,I_0,R_0 are the initial values for S,I,R functions; i.e. S(0),I(0),R(0)
#dt is the step size
#t is total length of time we are approximating for
def runge_kutta(a,b,c, S_0, I_0, R_0, dt, t):
    #create an array of values for the time. For example, supposed we want to look at 5 days with step size 1, the time array is [0,1,2,3,4,5]
    t_arr=np.array(np.arange(0,t+1,dt))
    #create the arrays for S, I, R values. They have the same dimension as the time array. All values are initialized with zero
    S=np.zeros(t_arr.shape[0])
    I=np.zeros(t_arr.shape[0])
    R=np.zeros(t_arr.shape[0])
    #initialize the first value of each array with the respective initial values given
    S[0]=S_0
    I[0]=I_0
    R[0]=R_0
    N=S_0+I_0+R_0
    #implementing RK4
    for i in range (0,t_arr.shape[0]-1):
        #Rugge kutta for S
        k_1=d_S(a,b,S[i],I[i])
        k_2=d_S(a,b,S[i]+(dt/2)*k_1,I[i]+(dt/2)*k_1)
        k_3=d_S(a,b,S[i]+(dt/2)*k_2,I[i]+(dt/2)*k_2)
        k_4=d_S(a,b,S[i]+dt*k_3,I[i]+dt*k_3)
        
        #update the next point of the array based on the previous point
        S[i+1]=S[i]+(1/6)*dt*(k_1+2*k_2+2*k_3+k_4)
        
        #rugge kutta for I
        k_1=d_I(b,c,S[i],I[i])
        k_2=d_I(b,c,S[i]+(dt/2)*k_1,I[i]+(dt/2)*k_1)
        k_3=d_I(b,c,S[i]+(dt/2)*k_2,I[i]+(dt/2)*k_2)
        k_4=d_I(b,c,S[i]+dt*k_3,I[i]+dt*k_3)
        
        I[i+1]=I[i]+(1/6)*dt*(k_1+2*k_2+2*k_3+k_4)
        
        R[i+1]=N-S[i+1]-I[i+1]
        
        #rugge kutta for R
        #k_1=d_R(c,I[i])
        #k_2=d_R(c,I[i]+(dt/2)*k_1)
        #k_3=d_R(c,I[i]+(dt/2)*k_2)
        #k_4=d_R(c,I[i]+dt*k_3)
        
        #R[i+1]=R[i]+(1/6)*dt*(k_1+2*k_2+2*k_3+k_4)
    
    return S,I,R, t_arr

#nondimensionalized SIR 
def d_s(a,b,s,i,N):
    return (-b/a)*s*i*N-s

def d_i(a,b,c,s,i,N):
    return (b/a)*s*i*N-(c/a)*i

def d_r(a,c,s,i,N):
    return s+(c/a)*i

def runge_kutta_nondim(a,b,c, s_0, i_0, r_0, N, dt, t):
    #create an array of values for the time. For example, supposed we want to look at 5 days with step size 1, the time array is [0,1,2,3,4,5]
    t_arr=np.array(np.arange(0,t+1,dt))
    #create the arrays for s, i, r values. They have the same dimension as the time array. All values are initialized with zero
    s=np.zeros(t_arr.shape[0])
    i=np.zeros(t_arr.shape[0])
    r=np.zeros(t_arr.shape[0])
    #initialize the first value of each array with the respective initial values given
    s[0]=s_0
    i[0]=i_0
    r[0]=r_0
    #implementing RK4
    for x in range (0,t_arr.shape[0]-1):
        #Rugge kutta for S
        k_1=d_s(a,b,s[x],i[x],N)
        k_2=d_s(a,b,s[x]+a*(dt/2)*k_1,i[x]+a*(dt/2)*k_1,N)
        k_3=d_s(a,b,s[x]+a*(dt/2)*k_2,i[x]+a*(dt/2)*k_2,N)
        k_4=d_s(a,b,s[x]+a*dt*k_3,i[x]+a*dt*k_3,N)
        
        #update the next point of the array based on the previous point
        s[x+1]=s[x]+(1/6)*a*dt*(k_1+2*k_2+2*k_3+k_4)
        
        #rugge kutta for I
        k_1=d_i(a,b,c,s[x],i[x],N)
        k_2=d_i(a,b,c,s[x]+a*(dt/2)*k_1,i[x]+a*(dt/2)*k_1,N)
        k_3=d_i(a,b,c,s[x]+a*(dt/2)*k_2,i[x]+a*(dt/2)*k_2,N)
        k_4=d_i(a,b,c,s[x]+a*dt*k_3,i[x]+a*dt*k_3,N)
        
        i[x+1]=i[x]+(1/6)*a*dt*(k_1+2*k_2+2*k_3+k_4)
        
        r[x+1]=1-s[x+1]-i[x+1]
    
    return s,i,r, t_arr

#here you can play around with the paramters, initial values, step size, length of time
s,i,r,t_arr=runge_kutta_nondim(0.01,0.005,0.01,99/100,1/100,0,100, 0.0005,100)


#adjusitng the scale of the plot. x-axis is going from 0 to 100, y-axis is scaled from 0 to 500
plt.xlim(0,50)
plt.ylim(0,1)
#ploting S, I, R on the y-axis with time on the x-axis
plt.plot(t_arr,s, label='Fraction of Susceptible')
plt.plot(t_arr,i, label='Fraction of Infected')
plt.plot(t_arr,r, label='Fraction of Recovered')
#labeling axis and creating legend
plt.xlabel('Time')
plt.ylabel('Fraction of People')
plt.legend()
#creates and shows the plot
plt.show

#you can print the arrays for S,I,R to see the values
print(s)
print(i)
print(r)

#the below allows you to save a file with all the arrays
#import pandas as pd
#output_file={'Time':t_arr,'s(t)':s,'i(t)':i,'r(t)':r}
#df=pd.DataFrame(output_file,columns=['Time','s(t)','i(t)','r(t)'])
#df.to_csv('output_nondim.csv')

