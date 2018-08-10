from imports import *


def sample_ExoTransmitspectra_TrhoRp(spectrum_fname, modeltheta, Ntrain, Ntest,
                                     seed=None, MRfunc=None, Trange=[200,800],
                                     rhorange=[1,20], Rprange=[.5,2.5],
                                     wlrange=[.95,2.5]):
    '''
    Read-in a template spectrum (i.e. transit depth vs wl) such as that which 
    is output by Exo-Transmit and scale it according to sampled values of the 
    planet temperature, density, and radius.

    Parameters
    ----------
    `spectrum_fname` : str
        File name of the spectrum data to be read-in.
    `modeltheta` : list of ints or floats
        The planet temperature, density, and radius used to compute the 
        spectrum given in `spectrum_fname`. Units are kelvin, g/cm^3, and Earth 
        radii respectively.
    `Ntrain` : int
        Number of training spectra to return.
    `Ntest` : int
        Number of testing spectra to return.
    `seed` : int or float
        Value to be fed to np.random.seed if the user wishes a seed to be 
        specified when sampling planetary parameters.
    `MRfunc` : function
        Function name of the mass-radius relation to be used to convert the 
        sampled planetary radii to mass. The function must take a single 
        argument: the planet radii to convert in Earth radii and return an 
        array of planet masses in Earth masses. Default is the relation from 
        Weiss & Marcy 2014 with intrinsic dispersion.
    `Trange` : list of int or float
        The minimum and maximum values of the planet temperature in kelvin to 
        sample when creating the training and testing sets of spectra.
    `rhorange` : list of int or float
        The minimum and maximum values of the planet density in cgs units to 
        sample when creating the training and testing sets of spectra.
    `Rprange` : list of int or float
        The minimum and maximum values of the planet radius in Earth radii to 
        sample when creating the training and testing sets of spectra.
    `wlrange` : list of int or float
        The minimum and maximum values of the observed wavelength to return in 
        microns.

    Returns
    -------
    `wl` : 1d array (Nwl,)
        Numpy array of sampled wavelengths.
    `training_specs` : 2d array (Ntrain, Nwl)
        Numpy array of spectra to train on.
    `testing_specs` : 2d array (Ntest, Nwl)
        Numpy array of spectra to test on.
    `training_samples` : 2d array (Ntrain, 3)
        Numpy array of the sampled model parameter values (i.e. T,rho,Rp) 
        corresponding to the returned Ntrain training spectra.
    `testing_samples` : 2d array (Ntest, 3)
        Numpy array of the sampled model parameter values (i.e. T,rho,Rp) 
        corresponding to the returned Ntest testing spectra.

    Examples
    --------
    >>> fname = "/Users/ryancloutier/Research/Exo_Transmit/Spectra/default.dat"
    >>> theta = 500, 12.5, 1.43
    >>> wl,trs,tes,trp,tep = sample_ExoTransmitspectra(fname, theta, 100, 50)
    >>> print trs.shape, tep.shape
    (100, 970) (50, 3)

    '''
    wl, depth = np.loadtxt(spectrum_fname, skiprows=2).T
    wl *= 1e6   # m -> microns
    goodwl = (wl >= np.min(wlrange)) & (wl <= np.max(wlrange))
    wl, depth = wl[goodwl], depth[goodwl]

    # Sample planet parameters from uniform distributions but use
    # the mass-radius relation to go from sampled radii to densities
    if seed is not None:
        np.random.seed(seed)
    Ts = np.random.uniform(np.min(Trange), np.max(Trange), Ntrain+Ntest)
    MRfunc = mrWM14_scat if MRfunc is None else MRfunc
    rhos, Rps = np.zeros(0), np.zeros(0)
    while rhos[(rhos >= np.min(rhorange)) & \
               (rhos <= np.max(rhorange))].size < Ntrain+Ntest:
        Rp = np.random.uniform(np.min(Rprange), np.max(Rprange))
        rho = 5.51 * MRfunc(np.array([Rp])) / Rp**3
        if np.min(rhorange) <= rho <= np.max(rhorange): 
            rhos = np.append(rhos, rho)
            Rps  = np.append(Rps, Rp)
            
    samples = np.array([Ts, rhos, Rps]).T
    training_samples, testing_samples = samples[:Ntrain], samples[Ntrain:]
    
    # Compute scaling of the chromatic planet transit depth
    Tp, rhop, Rpp = modeltheta
    Xs = (Ts/Tp) * (rhop/rhos) * (Rpp/Rps)
    
    # Construct training and test spectral sets
    specs = np.zeros((Ntrain+Ntest, wl.size)) 
    for i in range(Ntrain+Ntest):
        specs[i] = Xs[i] * (depth - depth.min()) + depth.min()
    training_specs, testing_specs = specs[:Ntrain], specs[Ntrain:]

    return wl, training_specs, testing_specs, training_samples, testing_samples


def sample_ExoTransmitspectra_Tgmu(spectrum_fname, modeltheta, Ntrain, Ntest,
                                   seed=None, MRfunc=None, Trange=[200,800],
                                   grange=[5,40], murange=[1,40],
                                   wlrange=[.95,2.5]):
    '''
    Read-in a template spectrum (i.e. transit depth vs wl) such as that which 
    is output by Exo-Transmit and scale it according to sampled values of the 
    planet temperature, density, and radius.

    Parameters
    ----------
    `spectrum_fname` : str
        File name of the spectrum data to be read-in.
    `modeltheta` : list of ints or floats
        The planet temperature, density, and radius used to compute the 
        spectrum given in `spectrum_fname`. Units are kelvin, g/cm^3, and Earth 
        radii respectively.
    `Ntrain` : int
        Number of training spectra to return.
    `Ntest` : int
        Number of testing spectra to return.
    `seed` : int or float
        Value to be fed to np.random.seed if the user wishes a seed to be 
        specified when sampling planetary parameters.
    `MRfunc` : function
        Function name of the mass-radius relation to be used to convert the 
        sampled planetary radii to mass. The function must take a single 
        argument: the planet radii to convert in Earth radii and return an 
        array of planet masses in Earth masses. Default is the relation from 
        Weiss & Marcy 2014 with intrinsic dispersion.
    `Trange` : list of int or float
        The minimum and maximum values of the planet temperature in kelvin to 
        sample when creating the training and testing sets of spectra.
    `grange` : list of int or float
        The minimum and maximum values of the planet surface gravity in SI 
        units to sample when creating the training and testing sets of spectra.
    `murange` : list of int or float
        The minimum and maximum values of the mean molecular weight to 
        sample when creating the training and testing sets of spectra.
    `wlrange` : list of int or float
        The minimum and maximum values of the observed wavelength to return in 
        microns.

    Returns
    -------
    `wl` : 1d array (Nwl,)
        Numpy array of sampled wavelengths.
    `training_specs` : 2d array (Ntrain, Nwl)
        Numpy array of spectra to train on.
    `testing_specs` : 2d array (Ntest, Nwl)
        Numpy array of spectra to test on.
    `training_samples` : 2d array (Ntrain, 3)
        Numpy array of the sampled model parameter values (i.e. T,rho,Rp) 
        corresponding to the returned Ntrain training spectra.
    `testing_samples` : 2d array (Ntest, 3)
        Numpy array of the sampled model parameter values (i.e. T,rho,Rp) 
        corresponding to the returned Ntest testing spectra.

    Examples
    --------
    >>> fname = "/Users/ryancloutier/Research/Exo_Transmit/Spectra/default.dat"
    >>> theta = 500, 31.8, 2
    >>> wl,trs,tes,trp,tep = sample_ExoTransmitspectra(fname, theta, 100, 50)
    >>> print trs.shape, tep.shape
    (100, 970) (50, 3)

    '''
    wl, depth = np.loadtxt(spectrum_fname, skiprows=2).T
    wl *= 1e6   # m -> microns
    goodwl = (wl >= np.min(wlrange)) & (wl <= np.max(wlrange))
    wl, depth = wl[goodwl], depth[goodwl]

    # Sample planet parameters from uniform distributions but use
    # the mass-radius relation to go from sampled radii to densities
    if seed is not None:
        np.random.seed(seed)
    Ts  = np.random.uniform(np.min(Trange), np.max(Trange), Ntrain+Ntest)
    gs  = np.random.uniform(np.min(grange), np.max(grange), Ntrain+Ntest)
    mus = np.random.uniform(np.min(murange), np.max(murange), Ntrain+Ntest)
    samples = np.array([Ts, gs, mus]).T
    training_samples, testing_samples = samples[:Ntrain], samples[Ntrain:]
    
    # Compute scaling of the chromatic planet transit depth
    Tp, gp, mup = modeltheta
    Xs = (Ts/Tp) * (gp/gs) * (mup/mus)
    
    # Construct training and test spectral sets
    specs = np.zeros((Ntrain+Ntest, wl.size)) 
    for i in range(Ntrain+Ntest):
        specs[i] = Xs[i] * (depth - depth.min()) + depth.min()
    training_specs, testing_specs = specs[:Ntrain], specs[Ntrain:]

    return wl, training_specs, testing_specs, training_samples, testing_samples




def mrWM14_scat(rps):
    '''
    Convert the input list of planet radii to planet masses using the 
    mass-radius relation from Weiss & Marcy 2014. Add a dispersion in planet 
    mass about the mean relation.

    Parameters
    ----------
    `rps` : 1d array
        Numpy array of planet radii to convert to masses in units of Earth's 
        radius.

    Returns
    -------
    `mps` : 1d array
        Numpy array of planet masses in units of Earth masses.

    '''
    rplim1, rplim2 = 1.5, 4.
    rms1, rms2 = 2.7, 4.7

    if type(rps) == float:
        rps = np.array([rps])

    mps = np.zeros(rps.size)
    for i,rp in enumerate(rps):

        if rp < rplim1:
            mp = .44*rp**3 + .614*rp**4
            mptmp = mp + np.random.randn() * rms1
            while mptmp < 0:
                mptmp = mp + np.random.randn() * rms1
            mps[i] = mptmp

        elif rp <= rplim2:
            mp = 2.69 * rp**(.93)
            mptmp = mp + np.random.randn() * rms2
            while mptmp < 0:
                mptmp = mp + np.random.randn() * rms2
            mps[i] = mptmp
            
        else:
            mps[i] = np.nan
        
    return mps



def mrWM14_mean(rps):
    '''
    Convert the input list of planet radii to planet masses using the 
    mean mass-radius relation from Weiss & Marcy 2014. This MR-relation is 
    deterministic.

    Parameters
    ----------
    `rps` : 1d array
        Numpy array of planet radii to convert to masses in units of Earth's 
        radius.

    Returns
    -------
    `mps` : 1d array
        Numpy array of planet masses in units of Earth masses.

    '''
    rplim1, rplim2 = 1.5, 4.

    if type(rps) == float:
        rps = np.array([rps])
    
    mps = np.zeros(rps.size)
    for i,rp in enumerate(rps):
        if rp < rplim1:
            mps[i] = .44*rp**3 + .614*rp**4
        elif rp <= rplim2:
            mps[i] = 2.69 * rp**(.93)
        else:
            mps[i] = np.nan
        
    return mps


def sample_ExoTransmitspectra_good(Ntrain, Ntest, forig=False):
    '''
    Get pre-computed set of transmission spectra and the corresponding 
    model parameters (Teq, g, rp, X, Rs).

    Parameters
    ----------
    `Ntrain` : int
        Number of training spectra to return.
    `Ntest` : int
        Number of testing spectra to return.

    Returns
    -------
    `wl` : 1d array (Nwl,)
        Numpy array of sampled wavelengths.
    `training_specs` : 2d array (Ntrain, Nwl)
        Numpy array of spectra to train on.
    `testing_specs` : 2d array (Ntest, Nwl)
        Numpy array of spectra to test on.
    `training_samples` : 2d array (Ntrain, 3)
        Numpy array of the sampled model parameter values (i.e. T,rho,Rp) 
        corresponding to the returned Ntrain training spectra.
    `testing_samples` : 2d array (Ntest, 3)
        Numpy array of the sampled model parameter values (i.e. T,rho,Rp) 
        corresponding to the returned Ntest testing spectra.

    Examples
    --------
    >>> Ntrain, Ntest = 100, 50
    >>> wl,trs,tes,trp,tep = sample_ExoTransmitspectra_good(Ntrain, Ntest)
    >>> print trs.shape, tep.shape
    (100, 1081) (50, 5)

    '''
    # Get model parameters (Teq, g, rp, X, Rs)
    if forig:
	samples = np.loadtxt('TrainingData/TrainingH2OParameters1.dat')
    else:
	samples = np.loadtxt('TrainingData/TrainingH2OParameters_bigg.dat')

    # Remove bad entries
    if forig:
	d = np.loadtxt('TrainingData/TrainingH2OSpectra1.dat')
    else:
    	d = np.loadtxt('TrainingData/TrainingH2OSpectra_bigg.dat')
    wl = d[:,0]
    tokeep = np.where(np.isfinite(d[0,1:]))[0]
    specs = d[:,tokeep+1]

    # Get training spectra
    try:
        training_specs, training_samples = specs[:,:Ntrain].T, samples[:Ntrain]
    except IndexError:
        training_specs = specs[:,0].T
   	warnings.warn('Ntrain is set to the maximum value of ' + \
		      '%i instead of %i'%(training_specs.shape[0], Ntrain),
		      Warning)

    # Get testing spectra
    try:
	testing_specs, testing_samples = specs[:,Ntrain:Ntrain+Ntest].T, \
					 samples[Ntrain:Ntrain+Ntest]
    except IndexError:
	testing_specs = specs[:,1].T
	warnings.warn('Ntest is set to the maximum value of ' + \
                      '%i instead of %i'%(training_specs.shape[0], Ntest),
                      Warning)
   
    # Translate to min(spectra) == 0
    #training_specs = (training_specs.T - np.min(training_specs, axis=1)).T
    #testing_specs = (testing_specs.T - np.min(testing_specs, axis=1)).T

    return wl, training_specs, testing_specs, training_samples, testing_samples

# GO SEE CREATE_TRAINING_SET_BENNEKESEAGER.PY IN ../TRANSMISSIONSPECTROSCOPY
