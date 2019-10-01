#
#  near
#
#  Created by Robert Hetland on 2005-12-12.
#  Copyright (c) 2005 Texas A\&M University. All rights reserved.
#

from matplotlib.pyplot import *
from numpy import *
from scipy import integrate

def rho_stp(s,t,p=0.):
    """
    returns density as a function of:
     s = Salinity in psu,
     t = Temperature in deg C,
     p = Pressure in dbar (default = 0)
    """
    p1 = 999.842594
    p2 =   6.793952E-2
    p3 =  -9.09529E-3
    p4 =   1.001685E-4
    p5 =  -1.120083E-6
    p6 =   6.536332E-9
    p7 =   8.24493E-1
    p8 =  -4.0899E-3
    p9 =   7.6438E-5
    p10 = -8.2467E-7
    p11 =  5.3875E-9
    p12 = -5.72466E-3
    p13 =  1.0227E-4
    p14 = -1.6546E-6
    p15 =  4.8314E-4
    k1 = 19652.21
    k2 =   148.4206
    k3 =    -2.327105
    k4 =     1.360477E-2
    k5 =    -5.155288E-5
    k6 =     3.239908
    k7 =     1.43713E-3
    k8 =     1.16092E-4
    k9 =    -5.77905E-7
    k10 =    8.50935E-5
    k11 =   -6.12293E-6
    k12 =    5.2787E-8
    k13 =   54.6746
    k14 =   -0.603459
    k15 =    1.09987E-2
    k16 =   -6.1670E-5
    k17 =    7.944E-2
    k18 =    1.6483E-2
    k19 =   -5.3009E-4
    k20 =    2.2838E-3
    k21 =   -1.0981E-5
    k22 =   -1.6078E-6
    k23 =    1.91075E-4
    k24 =   -9.9348E-7
    k25 =    2.0816E-8
    k26 =    9.1697E-10
    ro_st0 = p1 + p2*t + p3*t**2 + p4*t**3 + p5*t**4 + p6*t**5\
            + p7*s + p8*s*t + p9*t**2*s + p10*t**3*s + p11*t**4*s\
            + p12*s**1.5 + p13*t*s**1.5 + p14*t**2*s**1.5 + p15*s**2
    k_stp = k1 + k2*t + k3*t**2 + k4*t**3 + k5*t**4\
           + k6*p + k7*t*p + k8*t**2*p + k9*t**3*p\
           + k10*p**2 + k11*t*p**2 + k12*t**2*p**2\
           + k13*s + k14*t*s + k15*t**2*s + k16*t**3*s\
           + k17*s**1.5 + k18*t*s**1.5 + k19*t**2*s**1.5\
           + k20*p*s + k21*t*p*s + k22*t**2*p*s + k23*p*s**1.5\
           + k24*p**2*s + k25*t*p**2*s + k26*t**2*p**2*s
    return ro_st0/(1.0 - (p/k_stp))

def near_dydx(y,x,drf,weo,we_case):
    W, dr, u, h = y
    g = 9.8; ro = 1000.0+drf; gp = g*dr/ro
    if we_case == 'var':
        Ri = gp*h/u**2
        we = max(weo,u*(0.08-0.1*Ri)/(1.0+5.0*Ri))
    else:
        we = weo
    W_x = (2.0*sqrt(gp*h ))/u
    dr_x = - (we*dr/h)/u
    u_x = (gp*h*(W_x/W+dr_x/dr)-h*(g/ro)*dr_x-we*u/h)/(u-gp*h/u)
    h_x = -h*(u_x/u+W_x/W+dr_x/dr)
    return array([W_x, dr_x, u_x, h_x ])

def near(Sout=7.,Qf=1000.,Wo=1000.,weo=2.25e-4, dx=1.0):
    So = 32.; To = 10.          # Background ocean properties
    ro   = rho_stp(So, To)      # Oceanic denisty
    rf   = rho_stp(0.0, To)     # Fresh water density
    rout = rho_stp(Sout, To)    # Outflow denisty
    drf  = ro - rf              # Freshwater density anomaly
    dro  = ro - rout            # Outflow density anomaly
    g = 9.8; gp = g*dro/ro      # Gravity; Reduced gravity
    Q = Qf*(drf/dro)            # Outflow
    # Initial u and h are defined by Fr == 1 and Q=W*u*h
    uo = (Q*g*dro/(ro*Wo))**(1./3.)
    ho = Q/(uo*Wo)
    x = arange(0.,50000.,dx)    # Integration range (cross-shore meters)
    # Integrate equations
    y = integrate.odeint(near_dydx,array([Wo, dro, uo, ho]),x,\
            args=(drf,weo,'const'))
    W, dr, u, h = transpose(y)  # Map back to physical variables
    gp = g*dr/ro
    Fr = u/sqrt(gp*h)           # Profiles of Fr(x)
    s = (drf-dr)/0.77           # Salinity (psu, approximate)
    return  {'Fr': Fr[Fr>1],'s': s[Fr>1],'W': W[Fr>1],'dr': dr[Fr>1],'u': u[Fr>1],'h': h[Fr>1], 'x': x[Fr>1], 'Q': Q}


if __name__ == '__main__':
    import pylab as pl
    for Qf in arange(500.0, 2000.0, 500.0):
        for Wo in arange(500.0, 3000.0, 500.0):
            # for weo in array([0.1e-3, 0.3e-3, 0.5e-3, 0.8e-3, 1.0e-3]):
            for weo in array([0.1e-3, 0.5e-3, 1.0e-3]):
                d = near(Sout=0., Qf=Qf, Wo=Wo, weo=weo)
                fac = 2.0 * Qf / (Wo**2 * weo)
                try: L = d['dr'][-1]
                except: L = 0.0
                if fac<1: col = 'ro'
                else: col = 'bo'
                pl.plot([fac], [L], col)
    ax = pl.gca()
    ax.set_xscale('log')
    pl.show()

def plot_plume():
    s = near()
    plot(s['x']/1000.0,  s['W']/2000.0, 'k-')
    plot(s['x']/1000.0, -s['W']/2000.0, 'k-')
    axis('equal')
    for si in arange(6.0, 25.0, 2.0):
        idx = where(abs(s['s']-si) == min(abs(s['s']-si)))
        plot([s['x'][idx]/1000.0,  s['x'][idx]/1000.0], \
             [s['W'][idx]/2000.0, -s['W'][idx]/2000.0], 'k-')
    for si in arange(14., 17., 2.0):
        idx = where(abs(s['s']-si) == min(abs(s['s']-si)))
        plot([s['x'][idx]/1000.0,  s['x'][idx]/1000.0], \
             [s['W'][idx]/2000.0, -s['W'][idx]/2000.0], 'r-')
    axis([0.,7.,-5.,5.])
    axis('equal')
    yticks(np.arange(-5,6))
    savefig('near_frame.eps')

#
# def near(rout=21.0, ro=25.0, Q=1000., Wo=1000., weo=2.25e-4, dx=1.0):
#     rf   = rho_stp(0.0, 10)     # Fresh water density
#     drf  = ro - rf              # Freshwater density anomaly
#     dro  = ro - rout            # Outflow density anomaly
#     g = 9.8; gp = g*dro/ro      # Gravity; Reduced gravity
#     # Initial u and h are defined by Fr == 1 and Q=W*u*h
#     uo = (Q*g*dro/(ro*Wo))**(1./3.)
#     ho = Q/(uo*Wo)
#     x = arange(0.,50000.,dx)    # Integration range (cross-shore meters)
#     # Integrate equations
#     y = integrate.odeint(near_dydx,array([Wo, dro, uo, ho]),x,\
#             args=(drf,weo,'var'))
#     W, dr, u, h = transpose(y)  # Map back to physical variables
#     gp = g*dr/ro
#     Fr = u/sqrt(gp*h)           # Profiles of Fr(x)
#     s = (drf-dr)/0.77           # Salinity (psu, approximate)
#     return  {'Fr': Fr[Fr>1][-1],'W': W[Fr>1][-1], 'dro': dr[0], 'dr': dr[Fr>1],'u': u[Fr>1][-1],'h': h[Fr>1][-1], 'L': x[Fr>1][-1]}
#
