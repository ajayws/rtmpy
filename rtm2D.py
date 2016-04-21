# -*- coding: utf-8 -*-
"""

@author: Adi Wijaya
"""

from __future__ import division
import finite_difference as fd
import numpy as np
import wavelet as wv

def rtm2D(v,shotgt,dt,dx,dz):
    nz,nx = v.shape
    nt = shotgt[:,0].size
    ul, u, up, g = np.zeros((4,nz,nx))
    g[0] = 1
    ul= g*shotgt[nt-1]
    u = g*shotgt[nt-2]
    a = fd.alpha(v,dt,dx)
    data = np.zeros((nt, nz, nx))
    for i in xrange(nt-3, -1, -1):
        src = g*shotgt[i]
        ul[1:nz-1,1:nx-1]=2*u[1:nz-1,1:nx-1]-up[1:nz-1,1:nx-1]+\
                        a[1:nz-1,1:nx-1]**2*(u[1:nz-1,2:nx]+u[1:nz-1,0:nx-2]\
                        +u[2:nz,1:nx-1]+u[0:nz-2,1:nx-1]-4*u[1:nz-1,1:nx-1]) + src[1:nz-1,1:nx-1]
        
        ul = fd.abc2D_baru(up, u, ul, v, dt, dx, src, a)
        up = np.copy(u)
        u = np.copy(ul)
        up, u = fd.sponge2D(up,u)
        data[i] = np.copy(u)
        
    return data

def calculateBoundaries(ul, u, up, a, w):
    nz, nx = u.shape
    #surface z=0 and bottom z=nz boundaries
    ul [0, 1:nx-1] = 2*u[0,1:nx-1]-up[0,1:nx-1]+a[0,1:nx-1]**2*\
                (u[0,2:nx]+u[0,0:nx-2]+u[1,1:nx-1]-4*u[0,1:nx-1]) + w[0, 1:nx-1]
    ul [nz-1, 1:nx-1] = 2*u[nz-1,1:nx-1]-up[nz-1,1:nx-1]+a[nz-1,1:nx-1]**2*\
                (u[nz-1,2:nx]+u[nz-1,0:nx-2]+u[nz-2,1:nx-1]-4*u[nz-1,1:nx-1]) + \
                w[nz-1, 1:nx-1]
    #left x=0 and right x=nx boundaries
    ul[1:nz-1,0]=2*u[1:nz-1,0]-up[1:nz-1,0]+a[1:nz-1,0]**2*(u[1:nz-1,1]+\
                u[2:nz,0]+u[0:nz-2,0]-4*u[1:nz-1,0]) + w[1:nz-1,0] + w[1:nz-1,0]
    ul[1:nz-1,nx-1]=2*u[1:nz-1,nx-1]-up[1:nz-1,nx-1]+a[1:nz-1,nx-1]**2*\
                (u[1:nz-1,nx-2]+u[2:nz,nx-1]+u[0:nz-2,nx-1]-4*u[1:nz-1,nx-1])\
                + w[1:nz-1,nx-1]
    #corner must be modified
    ul[0,0]=2*u[0,0]-up[0,0]+a[0,0]**2*(u[0,1]+u[1,0]-4*u[0,0]) + w[0,0]
    ul[0,nx-1]=2*u[0,nx-1]-up[0,nx-1]+a[0,0]**2*(u[0,nx-2]+u[1,nx-1]-4*u[0,nx-1])\
                +w[0,nx-1]
    ul[nz-1,0]=2*u[nz-1,0]-up[nz-1,0]+a[nz-1,0]**2*(u[nz-1,1]+u[nz-2,0]-4*u[nz-1,0])\
              +w[nz-1,0]  
    ul[nz-1,nx-1]=2*u[nz-1,nx-1]-up[nz-1,nx-1]+a[nz-1,nx-1]**2*(u[nz-1,nx-2]+\
                    u[nz-2,nx-1]-4*u[nz-1,nx-1]) + w[nz-1,nx-1]
    return ul

                
