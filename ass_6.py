# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:48:01 2021

@author: To+
"""
import numpy as np
import matplotlib.pyplot as plt
from ass_3 import *

def f(x,A=np.array([[1.,0.999],[0.999,1.]]), b=np.array([0.999, 0.999])):
    return 1/2.*x.T@A@x+b.T@x

def f_g(x,A, b):
    return A@x+b.T

def problem_gen(n):
    B= np.random.randn(n,n)
    
    A=B@B.T
    b=b=2.*np.random.rand(n)-1.; x0=10*np.ones(n)
    limits=2*np.ones([n,2]); limits[:,0]*=-1
    return(A, b, x0, limits)

def problem_gen_2():
    beta=2.*np.random.rand()-1.
    A=np.array([[1,beta],[beta,1]])
    b=2.*np.random.rand(2)-1.
    return(A,b)

def check_if_in_box(x, limits):
    discr=[]
    for i in range(len(x)):
        if(limits[i, 0]<x[i] and limits[i,1]>x[i]):
            discr.append(False)
        else:
            discr.append(True)
    if any(discr):
        return(False)
    else:
        return(True)

def solution_norm(f, f_g , A, b, x, bounds):
    g=f_g(x, A, b)
    h=[]; c1=[]; c2=[]
    for i in range(len(x)):
        if(x[i]==bounds[i,0] and g[i]>0):
            h.append(0);
        elif(x[i]==bounds[i,1] and g[i]<0):
            h.append(0)
        else:
            h.append(g[i])
    return np.linalg.norm(np.array(h))




def subproblem(x, A, b, i, limits):
    alpha= -(x@A[i,:]+b[i])/A[i,i]
    x_min= x[i]+alpha
    if x_min<limits[i,0]:
        x_min= limits[i,0]
    elif x_min>limits[i,1]:
        x_min= limits[i,1]
    return x_min, alpha




#A=np.array([[1.,0.999],[0.999,1.]])



#coordinate descent

def coordinate_descent(x0, A, b, limits,  itmax=1E5, tol=1E-12):
    h_norm=1.; g=f_g(x0, A, b)
    it=0;
    x=x0.copy()
    convergence=[]
    while it<itmax and h_norm>tol:
        for i in range(len(x)):
            x[i], alpha = subproblem(x, A,b, i, limits)
            g+=alpha*A[i,:]
        it+=1
        h_norm= solution_norm(f, f_g , A, b, x, limits)
        convergence.append(x)
    return(x, convergence)




#exact solution

def f2(A,b):
    def f_aux(x):
        return f(x,A,b)
    return f_aux


def exact_solver(x0, A, b, limits):
    f_g0=f_g(x0,A,b)
    
    x, _,_= Newton_step(x0, f2(A,b), -f_g0, A)
    if(check_if_in_box(x,limits)):

        it=0; maxit=100; p=1; B=A.copy(); #grad=f_g0.copy()
        while(it<maxit and np.linalg.norm(p)>1E-8):
            it+=1
            
            grad=A@x+b.T
            x, p,_= Newton_step(x, f2(A,b), -grad, B)
            #print(p)
        x_min=x.copy()
    else:
        f_aux=1.
        minima=[]
        for i in range(np.shape(limits)[0]):
            for j in range(np.shape(limits)[1]):
                x=np.zeros(len(b)); x[i]=limits[i,j]; 
                for k in(range(len(x))):
                    if k!=i:
                        x[k],_=subproblem(x, A,b, k, limits)
                        if(f2(A,b)(x)<f_aux):
                            x_min=x.copy()
                            f_aux=f2(A,b)(x)
    return(x_min)


A, b, x0, limits= problem_gen(2)
A=np.array([[1.,0.999],[0.999,1.]]); b=np.array([0.999, 0.999])
"""
x1, convergence =  coordinate_descent(x0, A, b, limits)

plt.plot(convergence)
"""

"""
difference=[]
limits=np.array([[-1.,1.],[-1.,1.]])

for i in range(10000):
    A,b=problem_gen_2()
    x0=np.array([5.,5.])
    x1, _ =  coordinate_descent(x0, A, b, limits)
    x2 = exact_solver(x0, A, b, limits)
    difference.append(np.linalg.norm(x2-x1))
    if(np.linalg.norm(x2-x1)>1E-4):
        A_fail=A.copy(); b_fail=b.copy()
        print(A,b)

print(np.mean(difference))


"""

"""
itera=[]
itera_err=[]
for i in range(10):
    itera_aux=[]
    for j in range(1000):
        A, b, x0, limits= problem_gen(i)
        _,convergence= coordinate_descent(x0, A, b, limits)
        itera_aux.append(len(convergence))
    print(i)
    itera_err.append(np.std(itera_aux))
    itera.append(np.mean(itera_aux))
itera_x=np.linspace(1, len(itera), len(itera))
plt.plot(itera_x,itera)
plt.errorbar(itera_x, itera, yerr=np.array(itera_err)/np.sqrt(1000) , fmt = 'bx', capsize=3, alpha=.6 )
"""
#%%


