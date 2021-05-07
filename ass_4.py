# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:15:26 2021

@author: To+
"""
#epsilon=1E-6; tol= 1E-6

import numpy as np

from ass3.ass_3 import cholesky_modified, steepest_descent, Newton
from ass1.ass_1 import * 
import matplotlib.pyplot as plt

def m(p, f,f_g, f_h):
    return f+f_g.T@p+1./2.*p.T@f_h@p

def subprob_check(p, lamda, B, g, Delta):
    A=B+lamda*np.identity(np.shape(B)[0])
    r1= (np.linalg.norm((A@p)+g))<1E-4
    r2=(lamda*(Delta-np.linalg.norm(p)))<1E-4
    r3=(np.linalg.det(A))>=0; r4=(np.trace(A))>=0
    return((r1 and r2 and r3 and r4)==True)

def subprob(f,f_g, f_h,x, Delta, lambda0=0., itmax=10, tol=1E-4):
    it=0; lamda=lambda0; dif=1.; p=np.array(x)
    while(it<itmax and dif>tol ):
        it+=1
        R= cholesky_modified(f_h+lamda*np.identity(np.shape(f_h)[0]))
        p2=np.linalg.solve(R.T, -f_g(x)); p=np.linalg.solve(R, p2) ; q=np.linalg.solve(R.T, p )
        lamda+= (np.linalg.norm(p)/np.linalg.norm(q))**2*(np.linalg.norm(p)-Delta)/Delta

        dif=abs((np.linalg.norm(p)-Delta)/Delta)
        #pnorms.append(np.linalg.norm(p))
        #lambdas.append(lamda)
    #plt.plot( pnorms)
    
    return p, lamda, subprob_check(p, lamda, f_h, f_g(x), Delta)

def trust_step(f, f_g, f_h, x,eta, Delta_max,Delta):
    p,_, _ = subprob(f,f_g,f_h, x, Delta)
    #print(p)
    rho= (f(x)-f(x+p))/(f(x)-m(p, f(x),f_g(x), f_h))
    if rho<1/4.:
        Delta*=1/4.
    elif(rho>3/4. and np.linalg.norm(p)==Delta):
        Delta=min(2*Delta, Delta_max)
        
    if rho>eta:
        x+=p  
    return (x, p, Delta, rho)


def trust_region(f, f_g, f_h, x0, eta=0.2, Delta_max=10.,Delta=2E-3, itmax=5E4, printer=False):
    x=x0.copy()
    it=0
    convergence=[]
    while(it<itmax):
        convergence.append(f(x))
        it+=1
        
        x, p, Delta, rho= trust_step(f, f_g, f_h(x), x, eta=0.2, Delta_max=10.,Delta=2E-3)

        
    #  if it%1000==0:
         #  print(x,p, np.linalg.norm(p), Delta, rho, it)
            
    return x,convergence


"""
x0=np.array([20., 20.])
x1,monitor, radius=trust_region(f2("0"), f2("1"), f2("2"), x0)
plt.figure()
plt.plot(monitor[:,0], monitor[:,1])
plt.plot(1.,1., 'r.')

plt.plot(x0[0], x0[1], 'b.')
print(x1)


x = np.linspace(-10,10,1000)
y = np.linspace(-10,10,1000)
X,Y = np.meshgrid(x, y) # grid of point
Z = f2("0")([X, Y]) # evaluation of the function on the grid

im = plt.imshow(Z,cmap=plt.cm.RdBu, extent=[min(x), max(x), min(y), max(y)]) # drawing the function
# adding the Contour lines with labels
#cset = plt.contour([X,Y], Z ,np.arange(-1,1.5,0.2),linewidths=2,cmap=plt.cm.Set2)
#plt.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
plt.colorbar(im) # adding the colobar on the right
# latex fashion title
#plt.title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
plt.xlabel("$x_{1}$")
plt.ylabel("$x_{2}$")
plt.show()

residue=[]
for i in monitor:
    residue.append(np.linalg.norm(np.array(i)-np.array([0.,0.])))
    
plt.figure()
plt.plot(residue)


x0=[1.5,1.5]
for i in range(-7, 4, 2):
    x1,monitor=trust_region(f2("0"), f2("1"), f2("2"), x0, Delta_max=10**(i),Delta=10**(i-4), itmax=1E4)
    plt.plot(monitor[:,0], monitor[:,1], label=i)
    print(i)
plt.plot(1.,1., 'r.', label="objective")
plt.plot(x0[0], x0[1], 'b.', label="start")
plt.legend(loc="upper left", fontsize=9,title="$log(\Delta_{max})$")

x = np.linspace(0.3,1.6,1000)
y = np.linspace(0.9,2.2,1000)
X,Y = np.meshgrid(x, y) # grid of point
Z = f1("0")([X, Y]) # evaluation of the function on the grid

im = plt.imshow(Z,cmap=plt.cm.RdBu, extent=[min(x), max(x), min(y), max(y)]) # drawing the function
# adding the Contour lines with labels
#cset = plt.contour([X,Y], Z ,np.arange(-1,1.5,0.2),linewidths=2,cmap=plt.cm.Set2)
#plt.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
plt.colorbar(im) # adding the colobar on the right
# latex fashion title
#plt.title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
plt.xlabel("$x_{1}$")
plt.ylabel("$x_{2}$")
plt.show()

"""