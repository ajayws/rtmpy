# -*- coding: utf-8 -*-
"""

@author: Adi Wijaya
"""

from __future__ import division
import numpy as np
import scipy.io as sio

def finiteDifference1D(v, w, nt, dt, nx, dx, g):
    """
    1D finite central difference 2nd order. Based on von Neumann Stability analysis, 
    alpha = v*dt/dx <= 1.
    
    Parameter
    ---------
    v = velocity model
    w = model represented by wavelet
    nt,nx,nz = number of time, x and depth samples
    dt,dx,dz = discretization step/ sample rate
    
    
    Return
    ------
    
    References
    ----------
    
    Example
    -------

    """
    a = alpha(v,dt,dx)
    ul, u, up = np.zeros((3,nx))
    data = np.zeros((nt,nx))
    u += g*w[0]  
    # ul = np.copy(w)
    # u = initialCondition1D(ul, a)
    #data[0] = w
    data[0] = np.copy(ul)
    data[1] = np.copy(u)
    for i in xrange(2,nt):
        src = g * w[i]
        up[0]=2*u[0]-ul[0]+a[0]**2*(u[1]-2*u[0]) + src[0]
        up[1:nx-1]=2*u[1:nx-1]-ul[1:nx-1]+a[1:nx-1]**2*(u[2:nx]-2*u[1:nx-1]+ \
                    u[0:nx-2]) + src[1:nx-1]
        up = abc1D(u, up, a, src)
        ul = np.copy(u)
        u = np.copy(up)
        data[i] = np.copy(u)
    return data
def finiteDifference2D_baru(v, w, nt, dt, nx, dx, nz, dz, g):
    """
    2D finite central difference 2nd order. Based on von Neumann Stability analysis, 
    alpha = v*dt/dx <= 1/sqrt(2).
    
    Parameter
    ---------
    v = velocity model
    w = model represented by wavelet
    nt,nx,nz = number of time, x and depth samples
    dt,dx,dz = discretization step/ sample rate
    
    
    Return
    ------
    
    References
    ----------
    
    Example
    -------

    """
    a = alpha(v,dt,dx)
    a.astype('float64')
    ul, u, up, src = np.zeros((4,nz,nx), dtype='float64')
    data = np.zeros((nt,nz,nx),  dtype='float64')
    u = g*w[1]
    data[0] = np.copy(ul)
    data[1] = np.copy(u)
    for i in xrange(2,nt):
        src = g*w[i]
        up[1:nz-1,1:nx-1]=2*u[1:nz-1,1:nx-1]-ul[1:nz-1,1:nx-1]+\
                        a[1:nz-1,1:nx-1]**2*(u[1:nz-1,2:nx]+u[1:nz-1,0:nx-2]\
                        +u[2:nz,1:nx-1]+u[0:nz-2,1:nx-1]-4*u[1:nz-1,1:nx-1]) + src[1:nz-1,1:nx-1]
        
        up = abc2D_baru(ul, u, up, v,dt,dx, src, a)
        ul = np.copy(u)
        u = np.copy(up)
        ul, u = sponge2D(ul,u)
        data[i] = np.copy(u)
    return data

def abc2D_baru(ul, u, up, v,dt,dx, src, a):
    nz, nx = u.shape
    #surface z=0
    up [0, 1:nx-1] = 2*u[0,1:nx-1]-ul[0,1:nx-1]+a[0,1:nx-1]**2*\
                (u[0,2:nx]+u[0,0:nx-2]+u[1,1:nx-1]-4*u[0,1:nx-1]) + src[0, 1:nx-1]
    
    #bottom
    up[nz-1,2:nx-2] = -2.*dx*dt**2.*v[nz-1,2:nx-2]/(dx+v[nz-1,2:nx-2]*dt)*\
       ((-up[nz-2,2:nx-2]/(2*dt*dx) - ul[nz-1,2:nx-2]/(2*dt*dx) +\
       ul[nz-2,2:nx-2]/(2*dt*dx)) + 1/(2*dt**2*v[nz-1,2:nx-2])*\
       (-2*u[nz-1,2:nx-2] + ul[nz-1,2:nx-2] + up[nz-2,2:nx-2] - \
       2*u[nz-2,2:nx-2] + ul[nz-2,2:nx-2]) + (-v[nz-1,2:nx-2]/(4*dx**2))* \
       (up[nz-2,3:nx-1] - 2*up[nz-2,2:nx-2] + up[nz-2, 1:nx-3] + \
       ul[nz-1,3:nx-1] - 2*ul[nz-1,2:nx-2] + ul[nz-1,1:nx-3])) + src[nz-1,2:nx-2]
    #left x=0 
    up[2:nz-2,0] =(2*v[2:nz-2,0]*dx*dt**2)/(dx+v[2:nz-2,0]*dt)* \
       ((up[2:nz-2,1]/(2*dt*dx) - ul[2:nz-2,1]/(2*dt*dx) + \
       ul[2:nz-2,0]/(2*dt*dx)) + (-1/(2*dt**2*v[2:nz-2,0]))* \
       (-2*u[2:nz-2,0] + ul[2:nz-2,0] + up[2:nz-2,1] - \
       2*u[2:nz-2,1] + ul[2:nz-2,1]) + (v[2:nz-2,0]/(4*dx**2))* \
       (up[3:nz-1,1] - 2*up[2:nz-2,1] + up[1:nz-3,1] + \
       ul[3:nz-1,0] - 2*ul[2:nz-2,0] + ul[1:nz-3,0])) + src[2:nz-2,0]

    # right x=nx 
    up[2:nz-2,nx-1] =  -2*dx*dt**2*v[2:nz-2,nx-1]/(dx+v[2:nz-2,nx-1]*dt)* \
       ((-up[2:nz-2,nx-2]/(2*dt*dx) - ul[2:nz-2,nx-1]/(2*dt*dx) + \
       ul[2:nz-2,nx-2]/(2*dt*dx)) + 1/(2*dt**2*v[2:nz-2,nx-1])* \
       (-2*u[2:nz-2,nx-1] + ul[2:nz-2,nx-1] + up[2:nz-2,nx-2] - \
       2*u[2:nz-2,nx-2] + ul[2:nz-2,nx-2]) + (-v[2:nz-2,nx-1]/(4*dx**2))* \
       (up[3:nz-1,nx-2] - 2*up[2:nz-2,nx-2] + up[1:nz-3,nx-2] + \
       ul[3:nz-1,nx-1] - 2*ul[2:nz-2,nx-1] + ul[1:nz-3,nx-1])) + src[2:nz-2,nx-1]
    #corner
    up[0,0]=2*u[0,0]-ul[0,0]+a[0,0]**2*(u[0,1]+u[1,0]-4*u[0,0])+src[0, 0]
    up[0,nx-1]=2*u[0,nx-1]-ul[0,nx-1]+a[0,0]**2*(u[0,nx-2]+u[1,nx-1]-4*u[0,nx-1])+src[0, nx-1]
    up[nz-1,0]=2*u[nz-1,0]-ul[nz-1,0]+a[nz-1,0]**2*(u[nz-1,1]+u[nz-2,0]-4*u[nz-1,0])+ src[nz-1, 0]
    up[nz-1,nx-1]=2*u[nz-1,nx-1]-ul[nz-1,nx-1]+a[nz-1,nx-1]**2*(u[nz-1,nx-2]+\
                    u[nz-2,nx-1]-4*u[nz-1,nx-1])+src[nz-1,nx-1]
    return up

def initialCondition1D(u, a):
    """
    use this function only if initial condition != 0 is needed ?????  
    """
    nx = u.size
    ul = np.zeros(nx)
    ul[1:nx-1] = u[1:nx-1]+0.5*a[1:nx-1]**2*(u[2:]-2*u[1:nx-1]+u[0:nx-2])
    return ul
    
def alpha (v, dt, dx):
    """
    Calculate Courant number, alpha = v*dt/dx.
    For 2D, this function can be used if dx=xz    
    """
    return (v*dt/dx)
    
def abc1D (u, up, alpha, src):
    """
    Absorbing Boundaries Condition based on hybrid scheme which is explained in [2].
    If N=1, the method is the same as [1].
    Parameter
    ---------
    ul = wavefield at time step t-1
    u = wavefield at given time step t
    up = wavefield at time step t+1 (calculated wavefield)
    N = number of transition area
    
    Return
    ------
    up = new up modified by apllying absorbing boundaries condition
    
    References
    ----------
    [1] Clayton, R. W., and B. Engquist, 1977, Absorbing boundary conditions for 
    acoustic and elastic wave equations: Bulletin of the Seismological Society
    of America, 6, 1529–1540.
    [2] Liu, Y., and M. K. Sen, 2010, A hybrid scheme for absorbing edge reflections
    in numerical modeling of wave propagation: Geophysics, vol. 75, no. 2 March-April 2010; 
    P. A1–A6, 8 FIGS. 10.1190/1.3295447
    
    Example
    -------
    
    """
    nx = u.size
    # up[0] = u[1]+(up[1]-u[0])*(alpha[0]-1)/(alpha[0]+1) + src[0]
    up[nx-1] = u[nx-2]+(up[nx-2]-u[nx-1])*(alpha[nx-1]-1)/(alpha[nx-1]+1) + src[nx-1] 
                
    return up

def finiteDifference2D(v, w, nt, dt, nx, dx, nz, dz):
    """
    2D finite central difference 2nd order. Based on von Neumann Stability analysis, 
    alpha = v*dt/dx <= 1/sqrt(2).
    
    Parameter
    ---------
    v = velocity model
    w = model represented by wavelet
    nt,nx,nz = number of time, x and depth samples
    dt,dx,dz = discretization step/ sample rate
    
    
    Return
    ------
    
    References
    ----------
    
    Example
    -------

    """
    a = alpha(v,dt,dx)
    ul, u, up = np.zeros((3,nz,nx))
    data = np.zeros((nt,nz,nx))
    for src in w:
        p, q = src.indexes()
        print (p,q)
        u[p, q + 20] += -((v[p, q + 20] * dt) ** 2) * src(0)
    # u = np.copy(w)
    #ul = initialCondition2D(u, a)
    #data[0] = w
    for i in xrange(1,nt):
        up[1:nz-1,1:nx-1]=2*u[1:nz-1,1:nx-1]-ul[1:nz-1,1:nx-1]+\
                        a[1:nz-1,1:nx-1]**2*(u[1:nz-1,2:nx]+u[1:nz-1,0:nx-2]\
                        +u[2:nz,1:nx-1]+u[0:nz-2,1:nx-1]-4*u[1:nz-1,1:nx-1])
        #up = abc2D(ul, u, up, dt, dx, dz, v)
        up = calculateBoundaries(ul, u, up, a)
        ul = np.copy(u)
        u = np.copy(up)
        ul, u = sponge2D(ul,u)
        # yield u
        for src in w:
            p, q = src.indexes()
            u[p, q + 20] += -((v[p, q + 20] * dt) ** 2) * src(i * dt)
        data[i] = np.copy(u)
    # yield u
    return data

def calculateBoundaries(ul, u, up, a):
    nz, nx = u.shape
    #surface z=0 and bottom z=nz boundaries
    up [0, 1:nx-1] = 2*u[0,1:nx-1]-ul[0,1:nx-1]+a[0,1:nx-1]**2*\
                (u[0,2:nx]+u[0,0:nx-2]+u[1,1:nx-1]-4*u[0,1:nx-1])
    up [nz-1, 1:nx-1] = 2*u[nz-1,1:nx-1]-ul[nz-1,1:nx-1]+a[nz-1,1:nx-1]**2*\
                (u[nz-1,2:nx]+u[nz-1,0:nx-2]+u[nz-2,1:nx-1]-4*u[nz-1,1:nx-1])
    #left x=0 and right x=nx boundaries
    up[1:nz-1,0]=2*u[1:nz-1,0]-ul[1:nz-1,0]+a[1:nz-1,0]**2*(u[1:nz-1,1]+\
                u[2:nz,0]+u[0:nz-2,0]-4*u[1:nz-1,0])
    up[1:nz-1,nx-1]=2*u[1:nz-1,nx-1]-ul[1:nz-1,nx-1]+a[1:nz-1,nx-1]**2*\
                (u[1:nz-1,nx-2]+u[2:nz,nx-1]+u[0:nz-2,nx-1]-4*u[1:nz-1,nx-1])
    #corner
    up[0,0]=2*u[0,0]-ul[0,0]+a[0,0]**2*(u[0,1]+u[1,0]-4*u[0,0])
    up[0,nx-1]=2*u[0,nx-1]-ul[0,nx-1]+a[0,0]**2*(u[0,nx-2]+u[1,nx-1]-4*u[0,nx-1])
    up[nz-1,0]=2*u[nz-1,0]-ul[nz-1,0]+a[nz-1,0]**2*(u[nz-1,1]+u[nz-2,0]-4*u[nz-1,0])
    up[nz-1,nx-1]=2*u[nz-1,nx-1]-ul[nz-1,nx-1]+a[nz-1,nx-1]**2*(u[nz-1,nx-2]+\
                    u[nz-2,nx-1]-4*u[nz-1,nx-1])    
    return up

def sponge2D(ul, u):
    nz, nx = u.shape
    e = np.exp(-(0.015*(20-np.arange(1,21)))**2)
    for ixb in range(0,20):
        ul[0:nz-20,ixb] = e[ixb]*ul[0:nz-20,ixb]
        u[0:nz-20,ixb] = e[ixb]*u[0:nz-20,ixb]
        ixb2 = nx-20+ixb
        ul[0:nz-20,ixb2] = e[nx-ixb2-1]*ul[0:nz-20,ixb2]
        u[0:nz-20,ixb2] = e[nx-ixb2-1]*u[0:nz-20,ixb2]
        izb2 = nz-20+ixb
        ul[izb2] = e[nz-izb2-1]*ul[izb2]
        u[izb2] = e[nz-izb2-1]*u[izb2]
    return ul,u
    
def abc2D (ul, u, up, dt, dx, dz, v, n=None):
    """
    Absorbing Boundaries Condition based on hybrid scheme which is explained in [2].
    If N=1, the method is the same as [1].
    Parameter
    ---------
    ul = wavefield at time step t-1
    u = wavefield at given time step t
    up = wavefield at time step t+1 (calculated wavefield)
    N = number of transition area
    
    Return
    ------
    up = new up modified by apllying absorbing boundaries condition
    
    References
    ----------
    [1] Clayton, R. W., and B. Engquist, 1977, Absorbing boundary conditions for 
    acoustic and elastic wave equations: Bulletin of the Seismological Society
    of America, 6, 1529–1540.
    [2] Liu, Y., and M. K. Sen, 2010, A hybrid scheme for absorbing edge reflections
    in numerical modeling of wave propagation: Geophysics, vol. 75, no. 2 March-April 2010; 
    P. A1–A6, 8 FIGS. 10.1190/1.3295447
    
    Example
    -------
    
    """
    nz, nx = u.shape
    #Calculate right boundary constant
    R = 1/(2*dt*dx)+v[1:nz-1,nx-1]/(2*dt**2)
    R1 =1/(2*dt*dx)-v[1:nz-1,nx-1]/(2*dt**2)+v[1:nz-1,nx-1]/(2*dz**2)
    R2 = (1/(dt*dx)+v[1:nz-1,nx-1]/dt**2)
    R3 = (v[1:nz-1,nx-1]/dt**2-1/(dt*dz))
    R4 = (-v[1:nz-1,nx-1]/(2*dz**2)-v[1:nz-1,nx-1]/(2*dt**2)-1/(2*dt*dx))
    R5 = (1/(2*dt*dx)-v[1:nz-1,nx-1]/(2*dt**2))
    R6 = -v[1:nz-1,nx-1]/(4*dz**2)
    #right boundary 
    up[1:nz-1, nx-1] = R1*up[1:nz-1,nx-2]+R2*u[1:nz-1,nx-1]+R3*u[1:nz-1,nx-2]+\
                    R4*ul[1:nz-1,nx-1]+R5*ul[1:nz-1,nx-2]+R6*(up[2:nz,nx-2]+\
                    up[0:nz-2,nx-2]+ul[2:nz,nx-1]+ul[0:nz-2,nx-1])
    up[1:nz-1, nx-1] /= R

    def corner (up,idx,idz,dt,dx,dz):
        return (up[idz-1,idx]/dz+up[idz,idx-1]/dx+u[idz,idx]*\
                np.sqrt(2)/(dt*v[idz,idx]))/(1/dz+1/dx+np.sqrt(2)/(dt*v[idz,idx]))
    up[nz-1,nx-1] = corner(up,nz-1,nx-1,dt,dx,dz)
    up[nz-1,0] = corner(up,nz-1,0,dt,dx,dz)
    up[0,nx-1] = corner(up,0,nx-1,dt,dx,dz)
    up[0,0] = corner(up,0,0,dt,dx,dz)
    
    return up
    
def initialCondition2D(u, a):
    """
    ??? still unstable
    """
    nz,nx = u.shape
    ul = np.zeros((nz,nx))
    ul[1:nz-1,1:nx-1]=u[1:nz-1,1:nx-1]+0.5*a[1:nz-1,1:nx-1]**2*(u[1:nz-1,2:nx]+\
                    u[1:nz-1,0:nx-2]+u[2:nz,1:nx-1]+u[0:nz-2,1:nx-1]-4*u[1:nz-1,1:nx-1])
    return ul

def addBoundaryRegion (v):
    """
    add 20 nodes for absorbing boundaries 
    """
    z,x = v.shape
    model = np.zeros((z+20,x+40))
    model[0:z,20:x+20] = v.copy()
    model[0:z,0:20] = np.tile(v[:,0:1],(1,20))
    model[0:z,x+20:] = np.tile(v[:,x-1:x],(1,20))
    model[z:] = np.tile(model[z-1],(20,1))
    return model
    
    
    
    