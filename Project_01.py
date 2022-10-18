# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:46:37 2022

@author: nimes
"""

#initial conditions
x1 = 0
x2 = 0  #take 0 as initial value at time 0
x3 = 0
x0 = [x1, x2, x3]

f13 = 38    #mi3/year
f31 = 20   # from research paper we get these constants
f21 = 18
f32 = 18
v1 = 2900   #mi3
v2 = 850
v3 = 1180
k = [f13,f21,f31,f32,v1,v2,v3]

import numpy as np   #importing library numpy
t_start = 0 #initial time
t_end = 101 #end time
intrvl = 1  # interval at which we get results
tspan = (t_start, t_end)  #time span
t = np.arange(t_start, t_end, intrvl)  #time interval 0 to 100 at interval of 1

#function that return dxdt
def dxdt(t, x, f13, f31, f21, f32, v1, v2, v3):  #define the function 
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]   # initial value
    f13 = k[0]    
    f31 = k[2]
    f21 = k[1]
    f32 = k[3]
    v1 = k[4]   
    v2 = k[5]
    v3 = k[6]
    poln = 1 + np.sin(t)  #source of pollution ex 1
    #2 poln = 100 * t   
    #3 poln = 200 * np.exp((-10)*t)
    #lake 1 differential equation (change in concentration of pollution in lake 1)
    dx1dt = (f13*x3)/v3 + poln - (f31*x1)/v1 - (f21*x1)/v1 
    #lake 2 
    dx2dt = (f21*x1)/v1 - (f32*x2)/v2
    #lake 3
    dx3dt = (f31*x1)/v1 + (f32*x2)/v2 - (f13*x3)/v3
    dx4dt = (dx1dt, dx2dt, dx3dt)   #return function
    return (dx4dt)

from scipy.integrate import solve_ivp   # importing library
#using initial value problem method determine concentration of pollutants at given period of time
Xx = solve_ivp(dxdt, tspan, x0, 'RK23', t, args=k)   #RK23
#Xx

Yy = solve_ivp(dxdt, tspan, x0, 'RK45', t, args=k)   #RK45
#Yy

Zz = solve_ivp(dxdt, tspan, x0, 'Radau', t, args=k)  #Radau
#Zz

#check results
print(Xx.message)
print("number of evaluation of function:",Xx.nfev)
print("success:",Xx.success)

print(Yy.message)
print("number of evaluation of function:",Yy.nfev)
print("success:",Yy.success)

print(Zz.message)
print("number of evaluation of function:",Zz.nfev)
print("success:",Zz.success)

#convert to list to plot graph
Xx1 = Xx['y'].tolist()
Xx1

Yy1 = Yy['y'].tolist()
Yy1

Zz1 = Zz['y'].tolist()
Zz1

t1 = Xx['t'].tolist()
#t1

from matplotlib import pyplot as plt  #import library
x1 = Xx1[0]   #lake1 value of pollutant using RK23
x2 = Xx1[1]   #lake2
x3 = Xx1[2]   #lake3

y1 = Yy1[0]    #lake1 RK45
y2 = Yy1[1]    #lake2
y3 = Yy1[2]    #lake3

z1 = Zz1[0]    #lake1 Radau
z2 = Zz1[1]    #lake2
z3 = Zz1[2]    #lake3

fig,ax = plt.subplots(3,sharex=True,figsize=(5,10))  #3 plot
fig.tight_layout()  

lake10, = ax[0].plot(t1,x1,'yx',label='Lake1')
lake11, = ax[0].plot(t1,x2,'r+',label='Lake2')
lake12, = ax[0].plot(t1,x3,'b*',label='Lake3')

lake13, = ax[1].plot(t1,y1,'yx',label='Lake1')
lake14, = ax[1].plot(t1,y2,'r+',label='Lake2')
lake15, = ax[1].plot(t1,y3,'b*',label='Lake3')

lake16, = ax[2].plot(t1,z1,'yx',label='Lake1')
lake17, = ax[2].plot(t1,z2,'r+',label='Lake2')
lake18, = ax[2].plot(t1,z3,'b*',label='Lake3')

ax[0].legend(handles=[lake10,lake11,lake12],loc='upper left')
ax[1].legend(handles=[lake13,lake14,lake15],loc='upper left')
ax[2].legend(handles=[lake16,lake17,lake18],loc='upper left')

ax[0].set_title('RK23 Method')
ax[1].set_title('RK45 Method')
ax[2].set_title('Radau Method')

plt.xlabel('Time(T)')
plt.ylabel('Pollutant')
#ax[0].set_ylabel('Pollutant(RK23)')
#ax[1].set_ylabel('Pollutant(RK45)')
#ax[2].set_ylabel('Pollutant(Radau)')

ax[0].grid()
ax[1].grid()
ax[2].grid()

plt.show()

from scipy.integrate import trapezoid   #trapezoid fun to count cumulative
#RK23
Lake1Xx1 = Xx1[0]   #RK23- For lake 1 
Lake1Xx1

Lake2Xx1 = Xx1[1]    #lake2
Lake2Xx1

Lake3Xx1 = Xx1[2]   #lake3
Lake3Xx1

cum_lake1Xx1 = trapezoid(Lake1Xx1)  #cumulative of pollutants for lake 1
cum_lake1Xx1

cum_lake2Xx1 = trapezoid(Lake2Xx1)   #lake2
cum_lake2Xx1

cum_lake3Xx1 = trapezoid(Lake3Xx1)   #lake3
cum_lake3Xx1

#RK45
Lake1Yy1 = Yy1[0]   #RK45 - Lake1
Lake1Yy1

Lake2Yy1 = Yy1[1]    #lake2
Lake2Yy1

Lake3Yy1 = Yy1[2]    #lake3
Lake3Yy1

cum_lake1Yy1 = trapezoid(Lake1Yy1)   #cumulative of lake 1 pollutant
cum_lake1Yy1

cum_lake2Yy1 = trapezoid(Lake2Yy1)   #lake2
cum_lake2Yy1

cum_lake3Yy1 = trapezoid(Lake3Yy1)   #lake3
cum_lake3Yy1

#Radau
Lake1Zz1 = Zz1[0]   #Radau - lake1
Lake1Zz1

Lake2Zz1 = Zz1[1]   #lake2
Lake2Zz1

Lake3Zz1 = Zz1[2]    #lake3
Lake3Zz1

cum_lake1Zz1 = trapezoid(Lake1Zz1)   #cumulative of pollutnts of lake1
cum_lake1Zz1

cum_lake2Zz1 = trapezoid(Lake2Zz1)   #lake2
cum_lake2Zz1

cum_lake3Zz1 = trapezoid(Lake3Zz1)  #lake3
cum_lake3Zz1


























