#
#  near
#
#  Created by Robert Hetland on 2005-12-12.
#  Copyright (c) 2005 Texas A\&M University. All rights reserved.
#

import numpy as np
from scipy.integrate import odeint


def near_dydx(y, x, drf, weo, we_case):
    'Return RHS derivatives for near field plume ODEs'
    W, dr, u, h = y
    g = 9.8; ro = 1000.0+drf; gp = g*dr/ro
    if we_case == 'var':
        Ri = gp*h/u**2
        we = max(weo, u*(0.08-0.1*Ri)/(1.0+5.0*Ri))
    else:
        we = weo
    W_x = (2.0*np.sqrt(gp*h ))/u
    dr_x = - (we*dr/h)/u
    u_x = (gp*h*(W_x/W+dr_x/dr)-h*(g/ro)*dr_x-we*u/h)/(u-gp*h/u)
    h_x = -h*(u_x/u+W_x/W+dr_x/dr)
    return np.array([W_x, dr_x, u_x, h_x ])

def near(dro=20.0,Qf=1000.,Wo=1000.,weo=1e-4, dx=1.0, ro=1025):
    '''Return solution to near-field plume ODEs
    '''
    drf  = ro - 1000            # Freshwater density anomaly
    g = 9.8                     # Gravity
    gp = g*dro/ro               # Reduced gravity
    Q = Qf*(drf/dro)            # Outflow
    # Initial u and h are defined by Fr == 1 and Q=W*u*h
    uo = (Q*g*dro/(ro*Wo))**(1./3.)
    ho = Q/(uo*Wo)
    x =  np.arange(0.,50000.,dx)    # Integration range (cross-shore meters)
    # Integrate equations
    W, dr, u, h = odeint(near_dydx, np.array([Wo, dro, uo, ho]),x,\
            args=(drf, weo, 'const')).T
    gp = g*dr/ro
    Fr = u/np.sqrt(gp*h)           # Profiles of Fr(x)

    try:
        idx = np.where(Fr < 1)[0][0]
    except:
        return np.nan

    w1 = (1.0 - Fr[idx])/(Fr[idx-1] - Fr[idx])
    w2 = (Fr[idx-1] - 1.0)/(Fr[idx-1] - Fr[idx])

    # return  {'Fr': Fr,'W': W,'dr': dr,'u': u,'h': h, 'x': x, 'Q': Q}
    # return w1*dr[idx-1] + w2*dr[idx], dr[:idx], Fr[:idx]
    return w1*dr[idx-1] + w2*dr[idx]

if __name__ == '__main__':
    # Wo = np.arange(100, 3100, 100)
    # Qf = np.logspace(2, 4, 10)
    # dro = np.logspace(0, 1.4, 10)
    Wo = np.arange(100, 1100, 100)
    Qf = np.logspace(1, 3, 3)
    dro = np.logspace(0, 1.4, 3)
    params = np.array(np.meshgrid(Wo, Qf, dro)).T.reshape(-1,3)

    dr_final = []
    for Wo, Qf, dro in params:
        dr = near(dro=dro, Qf=Qf, Wo=Wo)
        if (dr>=dro):
            dr_final.append(np.nan)
        else:
            dr_final.append(dr)

    Wo = params[:,0]
    Qf = params[:,1]
    dro = params[:,2]
    Qo = Qf*25.0/dro
    dr = np.array(dr_final)
    
