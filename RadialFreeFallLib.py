import numpy as np
from scipy.integrate import quad

G = 6.6743015e-11 #m3 kg−1 s−2
M = 5.9722e24 #kg
r0 = 6.3781e6 #m

# minimum velocity of orbitation
vsat = np.sqrt(G*M/r0)

# escape velocity
vesc = np.sqrt(2*G*M/r0)

def t_integrand(r, v0):
    return 1./np.sqrt(v0**2+2*G*M*(1/r-1/r0))

def t_of_r(r, v0):
    return quad(t_integrand, r0, r, args=(v0))[0]

def rcrit_of_v0(v0):
    return r0/(1-v0**2*r0/(2*G*M))

def tcrit_of_v0(v0):
    return t_of_r(rcrit_of_v0(v0), v0)

def integrate(v0, tmin, tmax, step):
    dt = (tmax-tmin)/step
    r = np.zeros(step)
    v = np.zeros(step)
    sngv = np.ones(step)
    Ec = np.zeros(step)
    Ep = np.zeros(step)
    v[0] = v0
    r[0] = r0
    Ec[0] = 1/2.*v0**2
    Ep[0] = -G*M/r0
    E0 = Ec[0]+Ep[0]
    
    if E0 < 0:
        rcrit = rcrit_of_v0(v0)
        tcrit = tcrit_of_v0(v0)
    else:
        rcrit = -1.
        tcrit = -1.
        
    for i in range(1,step):
        ti = tmin+i*dt
        # update position, first-order Euler method
        r[i] = r[i-1] + dt*v[i-1]
        
        # update velocity from theory
        if E0>=0 or (E0 < 0 and ti <= tcrit and v0**2 + 2*G*M*(1/r[i]-1/r0) > 0):
            v[i] = np.sqrt(v0**2 + 2*G*M*(1/r[i]-1/r0))
        elif 2*G*M*(1/r[i]-1/rcrit) > 0:
            v[i] = -np.sqrt(2*G*M*(1/r[i]-1/rcrit))
        else: # invert trajectory exactly at tcrit
            r[i] = r[i-1]
        
        # update energy
        Ec[i] = 1/2.*v[i]**2
        Ep[i] = -G*M/r[i]
        
        # check if the body has reached the Earth
        if r[i] <= r0:
            r[i] = r0
            v[i] = v[i-1]
            Ec[i] = Ec[i-1]
            Ep[i] = Ep[i-1]

    return r, v, Ec, Ep, rcrit