# -*- coding: utf-8 -*-
"""

@author: Adi Wijaya
"""
from __future__ import division
import numpy as np

def ricker(f=20,n=100,dt=0.001,t0=0.05, tz=None):
    """
    Use ricker function to create ricker wavelet 1d/2d
    
    Parameters
    ----------
    f = frequency max(Hz)
    n = number of points
    dt = sampling rate
    t0 = peak location 
    tz = peak location at z axis    
    
    Example:
    --------
    >>> t,r = ricker()
        
    """
    T = dt*n
    t = np.arange(0,T,dt)
    tau = t[0:n]-t0
    if tz is None:
        return t[0:n], (1-tau*tau*f**2*np.pi**2)*np.exp(-tau**2*np.pi**2*f**2)
    else:
        T = np.meshgrid(tau,t-tz)
        return T, (1-(T[0]**2+T[1]**2)*f**2*np.pi**2)*np.exp(-(T[0]**2+T[1]**2)*np.pi**2*f**2)

def ormsby(f1=20, f2=25, f3=35, f4=40, n=1000, dt=0.001, t0=0.5):
    """
    Function to generate ormsby wavelet with given frequencies f1, f2, f3, and f4
    
    Parameters
    ----------
    f(1-4) = frequency(Hz)
    n = number of points
    dt = sampling rate
    t0 = peak location
    
    Example:
    --------
    >>> t,o = ormsby()
        
    """
    T = dt*n
    t = np.arange(0,T,dt)
    tau = t[0:n]-t0
    o = np.pi*f4**2*np.sinc(f4*tau)**2/(f4-f3)\
        -np.pi*f3**2*np.sinc(f3*tau)**2/(f4-f3)\
        -np.pi*f2**2*np.sinc(f2*tau)**2/(f2-f1)\
        +np.pi*f1**2*np.sinc(f1*tau)**2/(f2-f1)
    return t[0:n], o/np.amax(o)
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()