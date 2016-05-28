# -*- coding: utf-8 -*-
"""

@author: Adi Wijaya
"""

from __future__ import division
import finite_difference as fd
import numpy as np


def rtm1d(v,seis,dt,dz):
    nt = len(seis)
    nx = len(v)
    a = fd.alpha(v,dt,dz)
    ul, u, up = np.zeros((3,nx))
    data = np.zeros((nt,nx))
    g = np.zeros(u.shape)
    g[0] = 1
    ul += g*seis[nt-1]
    u += g*seis[nt-2]
    for i in xrange(nt-3,-1,-1):
        src = g*seis[i]
        ul[0]=2*u[0]-up[0]+a[0]**2*(u[1]-2*u[0]) + src[0]
        ul[1:nx-1]=2*u[1:nx-1]-up[1:nx-1]+a[1:nx-1]**2*(u[2:nx]-2*u[1:nx-1]+ \
                    u[0:nx-2]) + src[1:nx-1]
        ul = fd.abc1D(u, ul, a, src)
        up = np.copy(u)
        u = np.copy(ul)
        data[i] = np.copy(u)
    return data


def rtm2D(v,shotgt,dt,dx,dz):
    # rtm 2D with different algorithm 
    nz,nx = v.shape
    nt = shotgt[:,0].size
    ul, u, up = np.zeros((3,nz,nx))
    up[0,:], u[0,:], ul[0,:] = shotgt[nt-3:nt,:]
    a = fd.alpha(v,dt,dx)**2
    a2 = 2-4*a
    data = np.zeros((nt, nz, nx))
    e = (np.exp(-((0.015*(20-np.arange(1,21)))**2) ))**10
    c = 2    
    for i in xrange(nt-2,-1,-1):
        c+=1
        b = min(c,nz)
        for iz in xrange(b):
            ul[iz,0:20] = e*ul[iz,0:20]
            u[iz,0:20] = e*u[iz,0:20]
            ul[iz,nx-20:] = e[::-1]*ul[iz,nx-20:]
            u[iz,nx-20:] = e[::-1]*u[iz,nx-20:]
        
        if b >= (nz-20):
            for iz in xrange(nz-20,nz):
                ul[iz] = e[nz-iz-1]*ul[iz]
                u[iz] = e[nz-iz-1]*u[iz]
        if b == nz:
            d = nz-2
        else:
            d = b
        up[0:b,1:nx-1] = up[0:b,1:nx-1]-ul[0:b,1:nx-1]
        u[1:d,1:nx-1] = a2[1:d,1:nx-1]*ul[1:d,1:nx-1]+u[1:d,1:nx-1]+a[1:d,2:nx]*ul[1:d,2:nx]\
                        +a[1:d,0:nx-2]*ul[1:d,0:nx-2]+a[2:d+1,1:nx-1]*ul[2:d+1,1:nx-1]+\
                        +a[0:d-1,1:nx-1]*ul[0:d-1,1:nx-1]
                        
        u[0,1:nx-1] = a2[0,1:nx-1]*ul[0,1:nx-1]+u[0,1:nx-1]+a[0,2:nx]*ul[0,2:nx]\
                        +a[0,0:nx-2]*ul[0,0:nx-2]+a[1,1:nx-1]*ul[1,1:nx-1]
        
        if b == nz:
            u[nz-1,1:nx-1] = a2[nz-1,1:nx-1]*ul[nz-1,1:nx-1]+u[nz-1,1:nx-1]\
                       +a[nz-1,2:nx]*ul[nz-1,2:nx]+a[nz-1,0:nx-2]*ul[nz-1,0:nx-2]\
                       +a[nz-2,1:nx-1]*ul[nz-1,1:nx-1]
                       
            u[nz-1,0] = a2[nz-1,0]*ul[nz-1,0]+u[nz-1,0]+a[nz-1,1]*ul[nz-1,1]\
                        +a[nz-2,0]*ul[nz-2,0]
        
        u[1:d,0] = a2[1:d,0]*ul[1:d,0]+u[1:d,0]+a[1:d,1]*ul[1:d,1]+a[2:d+1,0]\
                    *ul[2:d+1,0]+a[0:d-1,0]*ul[0:d-1,0]
        
        u[1:d,nx-1] = a2[1:d,nx-1]*ul[1:d,nx-1]+u[1:d,nx-1]+a[1:d,nx-2]*ul[1:d,nx-2]\
                    +a[2:d+1,nx-1]*ul[2:d+1,nx-1]+a[0:d-1,nx-1]*ul[0:d-1,nx-1]

        u[0,0] = a2[0,0]*ul[0,0]+u[0,0]+a[0,1]*ul[0,1]+a[1,0]*ul[1,0]
        u[0,nx-1] = a2[0,nx-1]*ul[0,nx-1]+u[0,nx-1]+a[0,nx-1]*ul[0,nx-1]+a[1,nx-1]*ul[1,nx-1]

        ul = np.copy(u)
        u = np.copy(up)       
        
        if i > 1:
            up[1:nz-1] = 0;
            up[0] = shotgt[i-3,:]
        data[i] = ul
    return data





                
