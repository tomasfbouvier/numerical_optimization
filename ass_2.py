# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:26:53 2021
@author: To+
"""

import scipy.optimize as opt
from ass1.ass_1 import *
import matplotlib.pyplot as plt
import numpy as np

results = {}
results['nit'] = {}
results['nfev'] = {}
results['njev'] = {}
results['x'] = {}
x_0=1E-15
for i in range(0, 50, 5):
    message = 'Optimization terminated successfully.'
    x0 = np.array([x_0,x_0])
    tol = 10**(-i)
    print("tolerance= ", 10**(-i))
    results['nit'][str(tol)] = []
    results['nfev'][str(tol)] = []
    results['njev'][str(tol)] = []
    results['x'][str(tol)] = []
    iterations = 0
    while message == 'Optimization terminated successfully.':
        iterations += 1

        if(iterations == 10000):
            break
        x1 = opt.minimize(f3("0"), x0=x0, method='BFGS', jac=f3(
            "1"), tol=tol, options={'maxiter': 200000})
        x0[:] *= 2
        message = x1.message
        print(message)
#        print(message)
        if(message == 'Optimization terminated successfully.'):
            results['nit'][str(tol)].append(x1.nit)
            results['nfev'][str(tol)].append(x1.nfev)
            results['njev'][str(tol)].append(x1.nfev)
            results['x'][str(tol)].append(np.linalg.norm(x1.x-[0., 0.]))

# %%
plt.figure()
for i in results['x'].keys():
    x = np.arange(0, len(results['x'][i]), 1)
    plt.plot(x_0*np.sqrt(2.)*2.**x, results['x'][i], label=i)
    plt.yscale('log')
    plt.xscale('log')
plt.ylabel("error")
plt.xlabel("x0 distance to real minimum")
plt.legend(loc='best', fontsize=9, title='tolerance', ncol=2)
#plt.title("difference x real")
plt.show()
plt.figure()
for i in results['nit'].keys():
    x = np.arange(0, len(results['nit'][i]), 1)
    plt.plot(x_0*np.sqrt(2.)*2.**x, results['nit'][i], label=i)
    plt.yscale('log')
    plt.xscale('log')
plt.ylabel("number of iterations")
plt.xlabel("x0 distance to real minimum")
plt.legend(loc='best', fontsize=9, ncol=2, title='tolerance')
#plt.ylim(1.9, 7)
#plt.title("number of iterations")
plt.show()
