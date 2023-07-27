import numpy as np
from astropy.cosmology import w0waCDM
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

c_phantom = '#990099'
c_freezing = '#008080'
c_thawing = '#cc6600'

cosmo1={'h':0.6774, 'Omega_r':0., 'Omega_q':0., 'Omega_b':0., 'Omega_m':3., 'm_ncdm':0., 'Omega_k':0.,
       'tau_reio':0.066, 'n_s':0.9667, 'sigma8':0.8159, 'w0_fld':-1., 'wa_fld':0., 'k_max':10.0, 'WhichSpectrum':"CLASS"}
cosmo2={'h':0.6774, 'Omega_r':0., 'Omega_q':0., 'Omega_b':0., 'Omega_m':1., 'm_ncdm':0., 'Omega_k':0.,
       'tau_reio':0.066, 'n_s':0.9667, 'sigma8':0.8159, 'w0_fld':-1., 'wa_fld':0., 'k_max':10.0, 'WhichSpectrum':"CLASS"}
cosmo3={'h':0.6774, 'Omega_r':0., 'Omega_q':0., 'Omega_b':0., 'Omega_m':0.3, 'm_ncdm':0., 'Omega_k':0.,
       'tau_reio':0.066, 'n_s':0.9667, 'sigma8':0.8159, 'w0_fld':-1., 'wa_fld':0., 'k_max':10.0, 'WhichSpectrum':"CLASS"}
cosmo4={'h':0.6774, 'Omega_r':0., 'Omega_q':0.7, 'Omega_b':0., 'Omega_m':0.3, 'm_ncdm':0., 'Omega_k':0.,
       'tau_reio':0.066, 'n_s':0.9667, 'sigma8':0.8159, 'w0_fld':-1., 'wa_fld':0., 'k_max':10.0, 'WhichSpectrum':"CLASS"}
cosmo5={'h':0.6774, 'Omega_r':0., 'Omega_q':0.7, 'Omega_b':0., 'Omega_m':0.3, 'm_ncdm':0., 'Omega_k':0.,
       'tau_reio':0.066, 'n_s':0.9667, 'sigma8':0.8159, 'w0_fld':-2., 'wa_fld':0., 'k_max':10.0, 'WhichSpectrum':"CLASS"}
cosmo6={'h':0.6774, 'Omega_r':0., 'Omega_q':0.7, 'Omega_b':0., 'Omega_m':0.3, 'm_ncdm':0., 'Omega_k':0.,
       'tau_reio':0.066, 'n_s':0.9667, 'sigma8':0.8159, 'w0_fld':-1., 'wa_fld':0.3, 'k_max':10.0, 'WhichSpectrum':"CLASS"}
cosmo7={'h':0.6774, 'Omega_r':0., 'Omega_q':0.7, 'Omega_b':0., 'Omega_m':0.3, 'm_ncdm':0., 'Omega_k':0.,
       'tau_reio':0.066, 'n_s':0.9667, 'sigma8':0.8159, 'w0_fld':-1., 'wa_fld':-1., 'k_max':10.0, 'WhichSpectrum':"CLASS"}

w0waCDM1 = w0waCDM(H0=cosmo1['h']*100., Om0=cosmo1['Omega_m'], Ode0=cosmo1['Omega_q'],
                   Ob0=cosmo1['Omega_b'], w0=cosmo1['w0_fld'], wa=cosmo1['wa_fld'])
w0waCDM2 = w0waCDM(H0=cosmo2['h']*100., Om0=cosmo2['Omega_m'], Ode0=cosmo2['Omega_q'],
                   Ob0=cosmo2['Omega_b'], w0=cosmo2['w0_fld'], wa=cosmo2['wa_fld'])
w0waCDM3 = w0waCDM(H0=cosmo3['h']*100., Om0=cosmo3['Omega_m'], Ode0=cosmo3['Omega_q'],
                   Ob0=cosmo3['Omega_b'], w0=cosmo3['w0_fld'], wa=cosmo3['wa_fld'])
w0waCDM4 = w0waCDM(H0=cosmo4['h']*100., Om0=cosmo4['Omega_m'], Ode0=cosmo4['Omega_q'],
                   Ob0=cosmo4['Omega_b'], w0=cosmo4['w0_fld'], wa=cosmo4['wa_fld'])
w0waCDM5 = w0waCDM(H0=cosmo5['h']*100., Om0=cosmo5['Omega_m'], Ode0=cosmo5['Omega_q'],
                   Ob0=cosmo5['Omega_b'], w0=cosmo5['w0_fld'], wa=cosmo5['wa_fld'])
w0waCDM6 = w0waCDM(H0=cosmo6['h']*100., Om0=cosmo6['Omega_m'], Ode0=cosmo6['Omega_q'],
                   Ob0=cosmo6['Omega_b'], w0=cosmo6['w0_fld'], wa=cosmo6['wa_fld'])
w0waCDM7 = w0waCDM(H0=cosmo7['h']*100., Om0=cosmo7['Omega_m'], Ode0=cosmo7['Omega_q'],
                   Ob0=cosmo7['Omega_b'], w0=cosmo7['w0_fld'], wa=cosmo7['wa_fld'])

def a2z(a):
    return 1/a-1

def z2a(z):
    return 1/(1+z)

# The critical scale factor at which closed SCDM universes recollapse.
# It satisfies - Om*acrit**(-3) - Ok*acrit**(-2) = 0 (i.e. da/dt=0)
def acrit(w0waCDM):
    Om0=w0waCDM.Om(0.)
    Ok0=w0waCDM.Ok(0.)
    return -Om0/Ok0

acrit1 = acrit(w0waCDM1)

# The corresponding cosmic time is the age of the Universe at acrit minus its age now
tcrit1 = w0waCDM1.age(a2z(acrit1)).value - w0waCDM1.age(a2z(1.)).value
# (or equivalently - w0waCDM1.lookback_time(a2z(acrit1)).value + w0waCDM1.lookback_time(a2z(1.)).value)
# tcrit1 = - w0waCDM1.lookback_time(a2z(acrit1)).value + w0waCDM1.lookback_time(a2z(1.)).value

# Case of recollapsing universe
a1=np.linspace(1e-5,acrit1-1e-8,2500)
t1a=-w0waCDM1.lookback_time(a2z(a1)).value # that's when the universe expands
t1b=2*tcrit1-t1a # that's when it recollapses. a is symmetric with respect to acrit
tmax=2*tcrit1-t1a[0] # maximum cosmic time considered
t1=np.concatenate([t1a,np.flip(t1b)])
a1=np.concatenate([a1,np.flip(a1)])

# Case of expanding-forever universes
amax=fsolve(lambda a : - w0waCDM4.lookback_time(a2z(a)).value - tmax, 3.)[0]
a2=a3=a4=np.linspace(1e-5,amax,5000)
t2=-w0waCDM2.lookback_time(a2z(a2)).value
t3=-w0waCDM3.lookback_time(a2z(a3)).value
t4=-w0waCDM4.lookback_time(a2z(a4)).value
tmin=min(t1a.min(),t2.min(),t3.min(),t4.min())

# Case of dynamic dark energy
a5=a6=a7=np.linspace(1e-5,amax,5000)
t5=-w0waCDM5.lookback_time(a2z(a5)).value
t6=-w0waCDM6.lookback_time(a2z(a6)).value
t7=-w0waCDM7.lookback_time(a2z(a7)).value
tminDE=min(t4.min(),t5.min(),t6.min(),t7.min())

# Inverse the t(a) relations using 1d cubic interpolator
a1interp = interp1d(t1, a1, kind = 'cubic')
def a1_of_t(t):
    return a1interp(t) if t>t1.min() else 0.
a1_of_t = np.vectorize(a1_of_t)

a2interp = interp1d(t2, a2, kind = 'cubic')
def a2_of_t(t):
    return a2interp(t) if t>t2.min() else 0.
a2_of_t = np.vectorize(a2_of_t)

a3interp = interp1d(t3, a3, kind = 'cubic')
def a3_of_t(t):
    return a3interp(t) if t>t3.min() else 0.
a3_of_t = np.vectorize(a3_of_t)

a4interp = interp1d(t4, a4, kind = 'cubic', fill_value="extrapolate")
def a4_of_t(t):
    return a4interp(t) if t>t4.min() else 0.
a4_of_t = np.vectorize(a4_of_t)

a5interp = interp1d(t5, a5, kind = 'cubic', fill_value="extrapolate")
def a5_of_t(t):
    return a5interp(t) if t>t5.min() else 0.
a5_of_t = np.vectorize(a5_of_t)

a6interp = interp1d(t6, a6, kind = 'cubic', fill_value="extrapolate")
def a6_of_t(t):
    return a6interp(t) if t>t6.min() else 0.
a6_of_t = np.vectorize(a6_of_t)

a7interp = interp1d(t7, a7, kind = 'cubic', fill_value="extrapolate")
def a7_of_t(t):
    return a7interp(t) if t>t7.min() else 0.
a7_of_t = np.vectorize(a7_of_t)

# Setup lattice simulations
Nframes = 48*10
# Play back in time from present day, then forward in time from (first) big bang
Npast = int(Nframes*(-tmin)/(tmax-2*tmin))
Nfuture = int(Nframes*(tmax-tmin)/(tmax-2*tmin))+1
t = np.concatenate([np.linspace(0.,tmin+1e-8,Npast), np.linspace(tmin+1e-8,tmax,Nfuture)])
a1 = a1_of_t(t)/acrit1
a2 = a2_of_t(t)/acrit1
a3 = a3_of_t(t)/acrit1
a4 = a4_of_t(t)/acrit1

tmaxDE = 60.
NpastDE = int(Nframes*(-tminDE)/(tmaxDE-2*tminDE))
NfutureDE = int(Nframes*(tmaxDE-tminDE)/(tmaxDE-2*tminDE))+1
tDE = np.concatenate([np.linspace(0.,tminDE+1e-8,NpastDE), np.linspace(tminDE+1e-8,tmaxDE,NfutureDE)])
a4DE = a4_of_t(tDE)
a5DE = a5_of_t(tDE)
a6DE = a6_of_t(tDE)
a7DE = a7_of_t(tDE)

# Setup expansion/matter/curvature bars
N = 7 # This factor defines at which point the bars saturate
MAXARG = np.log(np.finfo(np.float64).max)
def efuncsq(a, model):
    w0 = model.w0
    wa = model.wa
    if a>0. and -3.*((1. + w0 + wa)*np.log(a) - wa*(a - 1.)) < MAXARG/2.:
        return model.efunc(a2z(a))**2
    else:
        return N
efuncsq = np.vectorize(efuncsq, excluded=['model'])
        
def Om(a, model):
    return model.Om(a2z(1.))*a**(-3) if a>0. else N
Om = np.vectorize(Om, excluded=['model'])

def Ok(a, model):
    if model == w0waCDM1:
        default = -N
    elif model == w0waCDM3:
        default = N
    else:
        default = 0.
    return model.Ok(a2z(1.))*a**(-2) if a>0. else default
Ok = np.vectorize(Ok, excluded=['model'])

def Oq(a, model):
    if model == w0waCDM4:
        default =  w0waCDM4.Ode(a2z(1.))
    elif model == w0waCDM6:
        default = N
    else:
        default = 0.
    TINY = np.finfo(np.float64).tiny
    if a>TINY:
        w0 = model.w0
        wa = model.wa
        if -3.*((1. + w0 + wa)*np.log(a) - wa*(a - 1.)) < MAXARG:
            return model.Ode(a2z(1.)) * np.exp(-3.*((1. + w0 + wa)*np.log(a) - wa*(a - 1.)) )
        else:
            return N
    else:
        return default
Oq = np.vectorize(Oq, excluded=['model'])

E1 = efuncsq(a1, w0waCDM1)/N
E2 = efuncsq(a2, w0waCDM2)/N
E3 = efuncsq(a3, w0waCDM3)/N
E4 = efuncsq(a4, w0waCDM4)/N
M1 = Om(a1, w0waCDM1)/N
M2 = Om(a2, w0waCDM2)/N
M3 = Om(a3, w0waCDM3)/N
M4 = Om(a4, w0waCDM4)/N
K1 = Ok(a1, w0waCDM1)/N
K2 = Ok(a2, w0waCDM2)/N
K3 = Ok(a3, w0waCDM3)/N
K4 = Ok(a4, w0waCDM4)/N
L4 = w0waCDM4.Ode(a2z(1.))/N
Q1 = Q2 = Q3 = np.zeros_like(t)
Q4 = L4*np.ones_like(t)

E4DE = efuncsq(a4DE, w0waCDM4)/N
E5DE = efuncsq(a5DE, w0waCDM5)/N
E6DE = efuncsq(a6DE, w0waCDM6)/N
E7DE = efuncsq(a7DE, w0waCDM7)/N
M4DE = Om(a4DE, w0waCDM4)/N
M5DE = Om(a5DE, w0waCDM5)/N
M6DE = Om(a6DE, w0waCDM6)/N
M7DE = Om(a7DE, w0waCDM7)/N
K4DE = Ok(a4DE, w0waCDM4)/N
K5DE = Ok(a5DE, w0waCDM5)/N
K6DE = Ok(a6DE, w0waCDM6)/N
K7DE = Ok(a7DE, w0waCDM7)/N
Q4DE = Oq(a4DE, w0waCDM4)/N
Q5DE = Oq(a5DE, w0waCDM5)/N
Q6DE = Oq(a6DE, w0waCDM6)/N
Q7DE = Oq(a7DE, w0waCDM7)/N
