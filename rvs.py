import numpy as np
from uncertainties import unumpy
from scipy.interpolate import interp1d

global G, Msun, Mearth, Rsun, Rearth, AU, pc, kb, mproton
G, Msun, Mearth = 6.67e-11, 1.98849925145e30, 6.04589804468e24
Rsun, Rearth, AU, pc = 695500e3, 6371e3, 1.495978707e11, 3.08567758149137e16
kb, mproton = 1.38e-23, 1.67e-27


# Define conversion functions
def days2sec(t):
    return t*24.*60*60
def sec2days(t):
    return t/(24.*60*60)
def days2yrs(t):
    return t/365.25
def yrs2days(t):
    return t*365.25
def Msun2kg(m):
    return m*Msun
def kg2Msun(m):
    return m/Msun
def Mearth2kg(m):
    return m*Mearth
def kg2Mearth(m):
    return m/Mearth
def Rearth2m(r):
    return r*Rearth
def m2Rearth(r):
    return r/Rearth
def Rsun2m(r):
    return r*Rsun
def m2Rsun(r):
    return r/Rsun
def AU2m(r):
    return r*AU
def m2AU(r):
    return r/AU
def pc2m(r):
    return r*pc
def m2pc(r):
    return r/pc 


def semimajoraxis(P_days, Ms_Msun, mp_Mearth):
    ''' Compute the semimajor axis in AU from Kepler's third law.'''
    P, Ms, mp = days2sec(P_days), Msun2kg(Ms_Msun), Mearth2kg(mp_Mearth)
    return m2AU((G*(Ms+mp)*P*P / (4*np.pi*np.pi)) **(1./3))


def period_sma(sma_AU, Ms_Msun, mp_Mearth):
    '''Compute the orbital period in days from the semimajor axis.'''
    sma, Ms, mp = AU2m(sma_AU), Msun2kg(Ms_Msun), Mearth2kg(mp_Mearth)
    return sec2days(np.sqrt(4*np.pi*np.pi*sma**3 / (G*(Ms+mp))))


def RV_K(P_days, Ms_Msun, mp_Mearth, ecc=0., inc_deg=90.):
    '''Compute the RV semiamplitude in m/s.'''
    P, Ms, mp, inc = days2sec(P_days), Msun2kg(Ms_Msun), Mearth2kg(mp_Mearth), \
                     np.deg2rad(inc_deg)
    return (2*np.pi*G/(P*Ms*Ms))**(1./3) * mp*np.sin(inc) / \
        unumpy.sqrt(1-ecc**2)


def RV_mp(P_days, Ms_Msun, K_ms, ecc=0., inc_deg=90.):
    '''Compute the planet mass from RV semiamplitude in Earth masses.'''
    P, Ms, inc = days2sec(P_days), Msun2kg(Ms_Msun), unumpy.radians(inc_deg)
    return kg2Mearth(K_ms * (P*Ms*Ms/(2*np.pi*G))**(1./3) * \
                     unumpy.sqrt(1-ecc**2)/unumpy.sin(inc))


def impactparam_inc(P_days, Ms_Msun, Rs_Rsun, inc_deg,
                    mp_Mearth=0., ecc=0., omega_deg=0.):
    '''Compute the impact parameter from the inclination and scaled 
    semimajor axis.'''
    sma, inc, omega = semimajoraxis(P_days, Ms_Msun, mp_Mearth), \
                      np.deg2rad(inc_deg), np.deg2rad(omega_deg)
    a_Rs = AU2m(sma) / Rsun2m(Rs_Rsun)
    return a_Rs * np.cos(inc) * ((1-ecc**2)/(1+ecc*np.sin(omega)))


def impactparam_T(P_days, Ms_Msun, Rs_Rsun, T_days,
                  mp_Mearth=0., ecc=0., omega_deg=0.):
    '''Compute the impact parameter from the transit width.'''
    sma, omega = semimajoraxis(P_days, Ms_Msun, mp_Mearth), \
                 np.deg2rad(omega_deg)
    a_Rs = AU2m(sma) / Rsun2m(Rs_Rsun)
    return np.sqrt(1. - (np.pi*T_days/P_days * a_Rs * \
                        (1+ecc*np.sin(omega))/np.sqrt(1-ecc**2))**2)


def inclination(P_days, Ms_Msun, Rs_Rsun, b,
                mp_Mearth=0., ecc=0., omega_deg=0.):
    '''Compute the inclination from the impact parameter and scaled 
    semimajor axis.'''
    sma, omega = semimajoraxis(P_days, Ms_Msun, mp_Mearth), \
                 unumpy.radians(omega_deg)
    a_Rs = AU2m(sma) / Rsun2m(Rs_Rsun)
    return unumpy.degrees(unumpy.arccos(b / a_Rs * \
                                        ((1+ecc*unumpy.sin(omega)) / \
                                         (1-ecc**2))))


def transit_width(P_days, Ms_Msun, Rs_Rsun, rp_Rearth, b,
                  mp_Mearth=0., ecc=0., omega_deg=0.):
    '''Compute the transit width (duration) in days.'''
    sma, omega, Rs = semimajoraxis(P_days, Ms_Msun, mp_Mearth), \
                     np.deg2rad(omega_deg), Rsun2m(Rs_Rsun)
    a_Rs = AU2m(sma) / Rs
    D = (Rearth2m(rp_Rearth) / Rs)**2
    return P_days/(np.pi*a_Rs) * np.sqrt((1+np.sqrt(D))**2 - b*b) * \
        (np.sqrt(1-ecc**2)/(1+ecc*np.sin(omega)))


def RM_K(vsini_kms, rp_Rearth, Rs_Rsun):
    '''Compute the approximate semi-amplitude for the Rossiter-McLaughlin 
    effect in m/s.'''
    D = (Rearth2m(rp_Rearth) / Rsun2m(Rs_Rsun))**2
    return (vsini_kms*D / (1-D)) * 1e3


def logg_model(mp_Mearth, rp_Rearth):
    '''Compute the surface gravity from the planet mass and radius.'''
    mp, rp = Mearth2kg(mp_Mearth), Rearth2m(rp_Rearth)
    return np.log10(G*mp/(rp*rp) * 1e2)


def logg_southworth(P_days, K_ms, aRp, ecc=0., inc_deg=90.):
    '''Compute the surface gravity in m/s^2 from the equation in Southworth 
    et al 2007.'''
    P, inc = days2sec(P_days), unumpy.radians(inc_deg)
    return 2*np.pi*K_ms*aRp*aRp * unumpy.sqrt(1-ecc*ecc) / (P*unumpy.sin(inc))


def tcirc(P_days, Ms_Msun, mp_Mearth, rp_Rearth):
    '''Compute the circularization timescale for a rocky planet 
    in years. From Goldreich & Soter 1966.'''
    Q = 1e2   # for a rocky exoplanet
    P, Ms, mp, rp, sma = days2yrs(P_days), Msun2kg(Ms_Msun), \
                         Mearth2kg(mp_Mearth), Rearth2m(rp_Rearth), \
                         semimajoraxis(P_days, Ms_Msun, mp_Mearth)
    return 2.*P*Q/(63*np.pi) * mp/Ms * (AU2m(sma) / rp)**5

    
#def RVspot(Prot, Rs, photometricA=5e-3, dVc=4., kappa=10.):
#    '''Estimate the radial velocity jitter from the flux effect 
#    convective blueshift effect, and total of a rotating star 
#    spot in m/s.
#    Prot = rotation period in days
#    Rs  = stellar radius in Rs'''
#    dt = .01*Prot/25.
#    t = np.arange(0, 2*Prot, dt)
#    dmag = photometricA*np.sin(2*np.pi*t/Prot)
#    flux = 10**(-.4*dmag)
#    Psi0 = np.max(flux)
#    Thetamin = np.min(flux)
#    f = (Psi0 - Thetamin) / Psi0
#    F = 1. - flux / Psi0
#    Fdot = np.gradient(F, dt) / (24.*60*60)  # F/sec
#    RVrot = -F*Fdot*Rsun2m(Rs)/f
#    RVconv = F*F*dVc*kappa/f
#    RVtot = RVrot + RVconv
#    return np.max(abs(RVrot)), np.max(abs(RVconv)), np.max(abs(RVtot))  # m/s


def transmission_spectroscopy_depth(Rs_Rsun, mp_Mearth, rp_Rearth, Teq, mu,
                                    Nscaleheights=5):
    '''Compute the expected signal in transit spectroscopy in ppm assuming 
    the signal is seen at 5 scale heights.'''
    g = 10**logg_model(mp_Mearth, rp_Rearth) * 1e-2
    rp = Rearth2m(rp_Rearth)
    D = (rp / Rsun2m(Rs_Rsun))**2
    H = kb*Teq / (mu*mproton*g)
    return Nscaleheights * 2e6 * D * H / rp

def emission_spectroscopy_depth(Rs_Rsun, rp_Rearth, Teff, Tp):
    '''Compute the expected signal in emission spectroscopy in ppm.'''
    d = (Rearth2m(rp_Rearth) / Rsun2m(Rs_Rsun))**2
    f = Tp / float(Teff)
    return d * f * 1e6


def stellar_density(P_days, T_days, Rs_Rsun, rp_Rearth, b):
    '''Compute the stellar density in units of the solar density (1.41 g/cm3) 
    from the transit parameters.'''
    rp, Rs, T, P = Rearth2m(rp_Rearth), Rsun2m(Rs_Rsun), days2sec(T_days), \
                   days2sec(P_days)
    D = (rp / Rs)**2
    rho = 4*np.pi**2 / (P*P*G) * (((1+np.sqrt(D))**2 - \
                                   b*b*(1-np.sin(np.pi*T/P)**2)) / \
                                  (np.sin(np.pi*T/P)**2))**(1.5)  # kg/m3
    rhoSun = 3*Msun2kg(1) / (4*np.pi*Rsun2m(1)**3)
    return rho  / rhoSun


def astrometric_K(P_days, Ms_Msun, mp_Mearth, dist_pc):
    '''Compute the astrometric semi-amplitude in micro-arcsec.'''
    P, Ms, mp, dist = days2sec(P_days), Msun2kg(Ms_Msun), \
                      Mearth2kg(mp_Mearth), pc2m(dist_pc)
    Krad = (G*P*P / (4*np.pi*np.pi*Ms*Ms))**(1./3) * mp /dist
    return np.rad2deg(Krad) * 3.6e3 * 1e6


def is_Lagrangestable(Ps, Ms, mps, eccs):
    '''Compute if a system is Lagrange stable (conclusion of barnes+
    greenberg 06).
    mp_i = Mearth'''
    Ps, mps, eccs = np.array(Ps), np.array(mps), np.array(eccs)
    smas = AU2m(semimajoraxis(Ps, Ms, mps))
    stable = np.zeros(mps.size-1)
    for i in range(1, mps.size):
        mu1 = Mearth2kg(mps[i-1]) / Msun2kg(Ms)
        mu2 = Mearth2kg(mps[i]) / Msun2kg(Ms)
        alpha = mu1+mu2
        gamma1 = np.sqrt(1-float(eccs[i-1])**2)
        gamma2 = np.sqrt(1-float(eccs[i])**2)
        delta = np.sqrt(smas[i]/smas[i-1])
        deltas = np.linspace(1.000001, delta, 1e3)
        LHS = alpha**(-3.) * (mu1 + mu2/(deltas**2)) * \
              (mu1*gamma1 + mu2*gamma2*deltas)**2
        RHS = 1. + 3**(4./3) * mu1*mu2/(alpha**(4./3))
        fint = interp1d(LHS, deltas, bounds_error=False, fill_value=1e8)
        deltacrit = fint(RHS)
        stable[i-1] = True if delta >= 1.1*deltacrit else False
    return stable

def sigma_depth(P, rp, Rs, Ms, b, N, Ttot, sig_phot):
     '''Compute the expected uncertainty on the transit depth from a 
     lightcurve with N measurements taken over Ttot days and with 
     measurement uncertainty sig_phot.'''
     delta = (Rearth2m(rp)/Rsun2m(Rs))**2
     sma = AU2m(semimajoraxis(P, Ms, 0))
     tau0 = P/(2*np.pi) * Rsun2m(Rs)/sma  # days
     T = 2*tau0*np.sqrt(1-b*b)
     Gamma = N/Ttot  # days^-1
     Q = np.sqrt(Gamma*T) * delta / sig_phot
     sig_delta = delta / Q
     return sig_delta

