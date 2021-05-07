# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 11:47:36 2021

@author: To+
"""

from ass1.ass_1 import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci



def cholesky_modified(A, beta=1E-3, itmax=2E5):
    if min(np.diag(A))>0:
        tau=0
    else:
        tau= - min(np.diag(A))+beta
    it=0
    while(it<itmax):
        it+=1
        try:
            L=np.linalg.cholesky(A+tau*np.identity(np.shape(A)[0]))
            break
        except:
            tau=max(2*tau, beta)
    return(L)
"""
def backtracking_linesearch2(x,f,f_g,p,alpha0=1,tau=0.99, c=0.5, maxit=2E9):
    alpha=alpha0; j=0
    m=f_g(x).T@p
    t=-c*m
    while(f(x)-f(x+alpha*p)<alpha*t and j<maxit):
        j+=1
        alpha*=tau
    print(alpha)
    return(alpha)
"""

def backtracking_linesearch(x,f,f_g,p,alpha0=2.,tau=0.99, c=0.5, maxit=2E9):
    alpha=alpha0; j=0
    m=f_g.T@p
    while(f(x+alpha*p)>= f(x)+c*alpha*m and alpha>1E-19):
        j+=1
        alpha*=tau
    return(alpha)

def Newton_step(x, f, grad, B):
    L=cholesky_modified(B)

    y=np.linalg.solve(L, grad); 
    p=np.linalg.solve(L.T, y)
    #grad=f_g(x); print("grad=", grad)
    alpha=backtracking_linesearch(x,f,-grad,p)
    x+= alpha*p
    return x, p, alpha


def Newton(f, f_g, f_h, x0, truemin, maxit=2E9):
    x=x0.copy();it=0; p=1.;  convergence=[]; res=[]
    while(it<maxit and np.linalg.norm(p)>1E-8 ):
        it+=1
        B=f_h(x)

        grad= - f_g(x)
        x, p,_= Newton_step(x, f, grad, B)
        convergence.append(np.linalg.norm(p))
        res.append(np.linalg.norm(x-truemin))
        #print(p)
    #TODO criterio de convergencia
    return(x, convergence, res)

def steepest_descent(f, f_g, x0, truemin, alpha0, maxit=1E2):
    x=x0.copy();it=0; p=10.;  convergence=[]; res=[]
    #TODO implmentar la busqueda de alpha.  
    while(it<maxit and np.linalg.norm(p)>1E-8):
        #plt.plot(x[0],x[1], 'r.')
        it+=1
        p=-f_g(x)
        step=backtracking_linesearch(x,f,f_g,p, alpha0)*p
        x+= step
        convergence.append(np.linalg.norm(p))
        res.append(np.linalg.norm(x-truemin))
        print(it)
    return(x,convergence, res)

"""
x1, convergence1, res1=steepest_descent(f4("0"), f4("1"),np.array([1E-6, 1E-6]), np.array([4.55E-8,4.55E-8]), alpha0=1)
print("x=", x1)

x2, convergence2, res2=Newton(f1("0"), f1("1"),f1("2"), np.array([2., 2.]), np.array([4.55E-8,4.55E-8]))
print("x=", x2)

plt.plot(res1, 'b', label='steepest descent')
plt.plot(res2, 'r', label='Newton')
plt.yscale('log')
plt.legend(loc='best',fontsize=9)
plt.xlabel("iterations")
plt.ylabel("||x-x*||")



"""