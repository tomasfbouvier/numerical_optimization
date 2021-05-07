# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:58:19 2021

@author: To+
"""

from ass_4 import *
from ass1.ass_1 import *
import numpy as np
import mpmath as mp

def update_B(x, f_g, B , p, it):
    y= (f_g(x+p)-f_g(x))

    """
    if(it>61603):
        y=mp.matrix(y.tolist()); mp.matrix(p.tolist()); B=mp.matrix(B.tolist())
        print("y",y, "p", p, "B", B)
        print("(y-B@p).T@p)", float(((y-B@p).T@p)[0,0]))
        print("np.outer((y-B@p).T,(y-B@p))", ((y-B@p)@(y-B@p).T)[:,:])
        print("aux", ((y-B@p)@(y-B@p).T)[:,:]/float(((y-B@p).T@p)[0,0]))
    """
    aux2=((y-B@p).T@p)
    if(np.linalg.norm(aux2)==0.0):
        aux=np.zeros(np.shape(B))
    else:
        aux1=np.outer((y-B@p).T,(y-B@p));
        aux= aux1/aux2; 
    B+= aux
    return B,y

def SR1(f, f_g, x0, eta=0.2, Delta_max=1E-1,Delta=10., tolerance=2E-3, itmax=1E5, printer=True):
    B=np.identity(len(x0))
    x=x0.copy()
    it=0
    convergence=[]
    p=1.
    while(it<itmax and np.linalg.norm(p)>tolerance):
        convergence.append(f(x))
        it+=1
        
        x, p, Delta, rho= trust_step(f, f_g, B, x, eta, Delta_max,Delta)   
        B,y = update_B(x, f_g, B , p, it)
        
        #B0 trial
        if(it==1):
            B= y.T@(x-x0)/(y.T@y)*np.identity(len(x0))
        if(it%2000==0 and printer==True):
            print(x, p, B, it)
    return x, convergence


#testing and stuff

"""
x0= np.array([5.,5., 5., 5., 5.,5.])
B0=f1("2")(x0)-20*np.identity(np.shape(f1("2")(x0))[0])

x, convergence=SR1(f1("0"),f1("1"), B0, x0)

plt.plot(convergence)


eta=0.2; Delta_max=1E-1;Delta=1E-3
x, p, Delta, rho=trust_step(f1("0"), f1("1"), B, x, eta, Delta_max,Delta)
update_B(x, f1("1"), B , p)

print(x,p)
"""

"""

y=np.array([2.00000000e-02, 1.61795345e-32])

p=np.array([1.00000000e-02, 8.08976727e-34])

B=np.array([[ 2.00000000e+00 ,-1.28169335e-31]
,[-1.28169335e-31,  2.15843390e+01]])
"""