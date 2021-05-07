# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:59:41 2021

@author: To+
"""
from ass1.ass_1 import *
import numpy as np
import matplotlib.pyplot as plt
from ass_6 import *

#%%

def project(x, limits):
    x2=x.copy()
    for i in range(np.shape(limits)[0]):
        if(x2[i]<limits[i,0]):
            x2[i]=limits[i,0]
        elif(x2[i]>limits[i,1]):
            x2[i]=limits[i,1]
    return(x2)


def SGD(f,f_g, x0, limits, mode, T,  eta0, alpha):
    x=x0.copy();
    x1=[x0]; eta=eta0
    for i in range(T):
        g=f_g(x)
        f_g2= g + np.random.normal(0,1.,np.shape(x)[0]) 
        eta= eta0/(i+1)
        x=project(x-eta*f_g2,limits)
        x1.append(list(x))
        
    if(mode=='A'):
        x_f=np.mean(x1,axis=0).copy()
    elif(mode=='alpha'):
        if(alpha==None):
            print('alpha: ')
            alpha= float(input())
        
        x_f=np.mean(x1[int(alpha*T):],axis=0).copy()
        
    elif(mode=='L'):
        x_f=x.copy() 
    return x_f,x1

#limits=np.array([[-0.5,1.],[-0.5,1.]])

limits=np.array([[-1E-6,1E-6],[-1E-6,1E-6]])#,[-1.,1.],[-1.,1.],[-1.,1.]])

true_min=f4("0")([1E-7, 1E-7]); true_min=0.
def exp_1(Ts, mode, x0):
    conv=[]
    for T in Ts:
        x,_= SGD(f4("0"), f4("1"), x0, limits=limits, eta0=3E-5 ,mode=mode, alpha=0.5, T=T);
        conv.append(abs(f4("0")(x)-true_min))
    #print(conv)
    return conv




log_Ts=np.linspace(5, 16,12, dtype=int);
Ts=2**log_Ts
conv_modes=[]; u_conv_modes=[]; modes=['A', 'alpha', 'L']
for mode in modes:
    print(mode)
    conv=[]
    for i in range(10):
        x0=(1E-6*(2*np.random.rand(2)-1.)).copy()
        conv.append(exp_1(Ts, mode, x0))
    conv_modes.append(np.mean(conv,axis=0))
    u_conv_modes.append(np.std(conv,axis=0))
    

plt.figure()
c=['b', 'r', 'y']
for i in range(len(modes)):
    
    plt.plot(log_Ts, conv_modes[i]*Ts, c[i])    
    plt.errorbar(log_Ts, conv_modes[i]*Ts, u_conv_modes[i]*Ts, fmt = c[i]+'x', capsize=3, alpha=.6, label=modes[i])

#plt.xscale('log2')
plt.legend(loc='best')
plt.xlabel("$log_{2}(T) $")
plt.ylabel("$(F(w)-F(w^{*}))\cdot T$")
plt.savefig("Experiment_1")

#%%

#ir añadiendo mas y mas ruido y ver como cambia

A, b, x0, limits= problem_gen(2)
def f_SGD(x):
    return(f(x,A,b))
def f_g_SGD(sigma):
    def f_g_aux(x):
        g=f_g(x,A,b).copy()
    #print(g)
        return(g+ np.random.normal(0,sigma, np.shape(x)[0]))
    return f_g_aux


x_t,_ =  coordinate_descent(x0, A, b, limits)
#%%
sigmas_log=np.linspace(-3, 1, 5)
sigmas=10**(sigmas_log)
for mode in modes:
    diff2=[];
    for j in range(100):
        print(j)
        
        diff=[]
        for sigma in sigmas:
            x,_= SGD(f_SGD, f_g_SGD(sigma), x0, limits=limits, eta0=1. ,mode=mode, alpha=0.5, T=10000);
            #print(x,x_t)
            diff.append(np.linalg.norm(x_t-x))
        diff2.append(diff)
        
    plt.plot(sigmas_log, np.mean(diff2, axis=0), color=c[modes.index(mode)])
    plt.errorbar(sigmas_log, np.mean(diff2, axis=0), np.std(diff2, axis=0), fmt = c[modes.index(mode)]+'x', capsize=3, alpha=.6, label=mode)
plt.xlabel("$log(sigma)$")
plt.ylabel("|$x-x^{*}$|")
plt.legend(loc="best",fontsize=9)
plt.savefig("SGD_with_noise")

#%%

sigmas_log=np.linspace(-.2,1., 7)
sigmas=10**(sigmas_log)
plt.figure()
x0=[2.,2.]
limits=np.array([[-2.,2.],[-2.,2.]])
for mode in modes:
    fs=[]
    for j in range(200):
        print(j)
        f_aux=[]
        for sigma in sigmas:
            x,x1= SGD(f1("0", alpha=1.), f1("1", alpha=1.,noise=False, sigma=sigma), x0, limits=limits, eta0=1. ,mode=mode, alpha=0.5, T=10000)
            
            f_aux.append(f1("0", alpha=1)(x))
        fs.append(f_aux)
    plt.plot(sigmas,np.mean(fs,axis=0), c[modes.index(mode)])
    plt.errorbar(sigmas, np.mean(fs,axis=0), np.std(fs, axis=0), fmt = c[modes.index(mode)]+'x', capsize=3, alpha=.6, label=mode)

plt.legend(loc='best')
plt.xlabel('noise $\sigma$')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('|$x-x^{*}$|')
plt.savefig("f1_with_noise")

#%% same experiment but for f3


sigmas_log=np.linspace(-.2,1., 7)
sigmas=10**(sigmas_log)
plt.figure()
x0=[2.,2.]
limits=np.array([[-2.,2.],[-2.,2.]])
for mode in modes:
    fs=[]
    for j in range(50):
        print(j)
        f_aux=[]
        for sigma in sigmas:
            x,x1= SGD(f3("0"), f3("1",noise=True, sigma=sigma), x0, limits=limits, eta0=1. ,mode=mode, alpha=0.5, T=10000)
            
            f_aux.append(f3("0")(x)-f3("0")([0.,0.]))
        fs.append(f_aux)
    plt.plot(sigmas,np.mean(fs,axis=0), c[modes.index(mode)])
    plt.errorbar(sigmas, np.mean(fs,axis=0), np.std(fs, axis=0), fmt = c[modes.index(mode)]+'x', capsize=3, alpha=.6, label=mode)

plt.legend(loc='best')
plt.xlabel('noise $\sigma$')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('|$x-x^{*}$|')
plt.savefig("f3_with_noise")

