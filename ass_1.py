# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:23:14 2021

@author: To+
"""

import numpy as np
import matplotlib.pyplot as plt

def f1(derivative, alpha=1., noise=False, sigma=0.):
    if(derivative=="0"):
        def f1_aux(x):
            d=np.shape(x)[0]
            f=0
            for i in range(d):
                f += alpha**(i/(d-1))*x[i]**2
            return f
        
    elif(derivative=="1"):
        def f1_aux(x):
            d=np.shape(x)[0]
            f=[]
            for i in range(d):
                f.append(2*alpha**(i/(d-1))*x[i])
            f=np.array(f)  
            if noise==True:
                f+=np.random.normal(0.,sigma, np.shape(f)[0])
            
            return f
    elif(derivative=="2"):
        def f1_aux(x):
            d=np.shape(x)[0]
            f=[]
            for i in range(d):
                fl=[]
                for j in range(d):
                    if(j!=i):
                        fl.append(0)
                    else:
                        fl.append(2*alpha**(i/(d-1)))
                f.append(fl)
            f=np.array(f)
            return f
    return f1_aux 

def f2(derivative):
    if(derivative=="0"):
        def f2_aux(x):
            f= (1-x[0])**2+100*(x[1]-x[0]**2)**2
            return f
    elif(derivative=="1"):
        def f2_aux(x):
            f=np.array([-2*(1-x[0])-2*100*(x[1]-x[0]**2)*2*x[0], 2*100*(x[1]-x[0]**2)])
            return f
    elif(derivative=="2"):
        def f2_aux(x):
            f=np.array([[2-400*(x[1]-x[0]**2)+800*x[0]**2,-400*x[0]],[-400*x[0], 2*100]])
            return f
    return f2_aux

def f3(derivative, epsilon=1E-6, noise=False, sigma=0.):
    if(derivative=="0"):
        def f3_aux(x):
            f=np.log(epsilon+f1("0")(x))
            return f
    elif(derivative=="1"):
        def f3_aux(x):    
            f= 1/(epsilon+f1("0")(x))*f1("1")(x)
            
            if(noise==True):
                f+=np.random.normal(0.,sigma, np.shape(f)[0])
            
            
            return f
    elif(derivative=="2"):
        def f3_aux(x):
            f=-1/(epsilon+f1("0")(x))**2*np.outer(f1("1")(x),f1("1")(x))+1/(epsilon+f1("0")(x))*f1("2")(x)
            return(f)
    return f3_aux

def h(x, derivative2,q=1E8):
    if(derivative2=="0"):
        h=1/q*np.log(1+np.exp(q*x))
    elif(derivative2=="1"):
        h=1/(1+np.exp(q*x))*np.exp(q*x)
    elif(derivative2=="2"):
        h=-q/(1+np.exp(q*x))**2*np.exp(q*x)**2+q/(1+np.exp(q*x))*np.exp(q*x)
    return(h)

def f4(derivative):
    if(derivative=="0"):
        def f4_aux(x):
            d=np.shape(x)[0]
            f4=0
            for i in range(d):
                    f4+= h(x[i], derivative)+100*h(-x[i], derivative)
            return f4

    elif(derivative=="1"):
        def f4_aux(x):
            d=np.shape(x)[0]
            f4=[]
            for i in range(d):
                f4.append(h(x[i], "1")-100*h(-x[i], "1"))       
            return np.array(f4)
        
    elif(derivative=="2"):
        def f4_aux(x):
            d=np.shape(x)[0]
            f4=np.zeros([d,d])
            aux=[]
            for i in range(d):
                aux.append(h(x[i], "2")+100*h(-x[i], "2"))
            np.fill_diagonal(f4,aux)        
            return f4
        
    return f4_aux

def f5(derivative):
    if(derivative=="0"):
        def f5_aux(x):
            d=len(x)
            f4=0
            for i in range(d):
                    f4+= h(x[i], derivative)**2+100*h(-x[i], derivative)**2
            return f4
    elif(derivative=="1"):
        def f5_aux(x):     
            d=len(x)
            f4=[]
            for i in range(d):
                f4.append(2*h(x[i], "0")*h(x[i],"1")-2*100*h(x[i], "0")*h(x[i],"1"))  
            return np.array(f4)
        
    elif(derivative=="2"):
        def f5_aux(x):  
            d=len(x)
            f4=np.zeros([d,d])
            aux=[]
            for i in range(d):
                aux.append(2*h(x[i], "1")*h(x[i],"1")+2*h(x[i], "0")*h(x[i],"2")+2*100*h(-x[i], "1")*h(-x[i],"1")+2*h(-x[i], "0")*h(-x[i],"2"))
            np.fill_diagonal(f4,aux)    
            return f4
    return f5_aux



"""
x = np.linspace(-1,2,1000)
y = np.linspace(-1,2,1000)
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

"""