from imports import *

global path2exotransmit, path2output
path2exotransmit = '/Users/ryancloutier/Research/Exo_Transmit'
path2output = '/Users/ryancloutier/Research/egrets/TrainingData'


def create_ExoTransmit_spectral_set(prefix, N, Nwl=430,
                                    chems=['CH4','CO2','CO','H2O','NH3','HCN'],
                                    Tlim=(2408,2622), glim=(10.6,12.2),
                                    rplim=(1.68,1.79), Rslim=(1.50,1.64),
                                    maxZ=0.0134, scattering=True, CIA=True):
    '''
    Create a set of transmission spectra for a particular planet using the 
    ExoTransmit code (http://adsabs.harvard.edu/abs/2017PASP..129d4402K).
    Default parameters are for WASP-12b.

    Arguments
    ---------
    `prefix': string
        Prefix for the transmission spectrum file names
    `N': scalar
        The number of transmission to compute
    `chems': tuple or list of strings
        List of chemisty strings (e.g. "H2O") to sample over the spectral set
    `Tlim': tuple or array-like
        Lower and upper limit on the planet's isothermal atmospheric 
        temperature
    `glim': tuple or array-like
        Lower and upper limit on the planet's surface gravity in m/s^2
    `rplim': tuple or array-like
        Lower and upper limit on the planet's opaque radius in Jupiter radii
    `Rslim': tuple or array-like
        Lower and upper limit on the stellar radius in Solar radii
    `maxZ': scalar
        Set the maximum atmospheric mass fraction that is not in H or He. 
        Default value is the solar metallicity
    `scattering': boolean
        If True, include the effect of scattering in the set of transmission
        spectra
    `CIA': boolean
        If True, include the effect of collision-induced absorption in the set 
        of transmission spectra
    '''
    assert (len(Tlim) == 2) & (Tlim[0] < Tlim[1])
    assert (len(glim) == 2) & (glim[0] < glim[1])
    assert (len(rplim) == 2) & (rplim[0] < rplim[1])
    assert (len(Rslim) == 2) & (Rslim[0] < Rslim[1])
    chems = np.sort(chems)
    assert chems.size > 0
    assert 0 < maxZ <= 1

    # setup arrays to hold input parameters and spectra
    N = int(N)
    Ts, gs, rps, Rss, logPcs, rayleighs = np.zeros(N), np.zeros(N), \
                                          np.zeros(N), np.zeros(N), \
                                          np.zeros(N), np.zeros(N)
    logXss = np.zeros((N, chems.size))
    specs = np.zeros((N, Nwl))

    # compile ExoTransmit first
    _compile_exotransmit()
    
    # sample parameters and compute transmission spectra
    i = 0
    while i < N:
        
        # try ExoTransmit and only save results if it run successfully
        try:
            # sample parameters
            T, g, rp, Rs = np.random.uniform(Tlim[0], Tlim[1]), \
                           np.random.uniform(glim[0], glim[1]), \
                           np.random.uniform(rplim[0], rplim[1]), \
                           np.random.uniform(Rslim[0], Rslim[1])
            logPc, rayleigh = np.random.uniform(-8,0), \
                              10**np.random.uniform(-3,3)
            logXs = np.repeat(-50., chems.size)
            for j in range(chems.size):
                logXs[j] = np.random.uniform(-8,
                                             np.log10(maxZ-(10**logXs).sum()))
            # shuffle so not biased towards decreasing X with chemical species
            np.random.shuffle(logXs)
            
            # setup exotransmit files
            suffix = '%i_%i'%(np.random.randint(0,1e8),
                              np.random.randint(0,1e8))
            outfile = 'Spectra/%s_%s.dat'%(prefix, suffix)
            _setup_exotransmit(suffix, T, g, rp, Rs, logPc, rayleigh,
                               chems, logXs, scattering, CIA, outfile)

            # get spectra from exotransmit
            _run_exotransmit(suffix)
            wl, specs[i] = _get_transmission_spectrum(outfile, Nwl)
            _clean_up(prefix, suffix)

            # save parameters
            Ts[i], gs[i], rps[i], Rss[i], logPcs[i], rayleighs[i] = T, g, \
                                                                    rp, Rs, \
                                                                    logPc, \
                                                                    rayleigh
            logXss[i] = logXs
            i+=1 
            
        except ValueError:
            _clean_up(prefix, suffix)

        
    # save input parameters and spectra
    _save_results(prefix, Ts, gs, rps, Rss, logPcs, rayleighs, chems, logXss,
                  wl, specs)


def _setup_exotransmit(suffix, T, g, rp, Rs, logPc, rayleigh, chems, logXs,
                       scattering, CIA, outfile):
    '''
    Configure the Exo-Transmit input files to simulate the transmission 
    spectrum of the planet with a given set of parameters.
    '''
    # Create input files
    _setup_chemistry(suffix, chems, **{'scattering':scattering, 'CIA':CIA})
    TPfile = _create_TPfile(suffix, T)
    EOSfile = _create_EOSfile(suffix, chems, logXs)
    
    # Read in input template file to modify
    f = open('%s/userInput.in'%path2exotransmit, 'r')
    k = f.readlines()
    f.close()
    assert len(k) == 21
    
    # Add required parameters
    k[3] = '%s\n'%path2exotransmit
    k[5] = '/T_P/%s\n'%TPfile
    k[7] = '/EOS/%s\n'%EOSfile
    k[9] = '/%s\n'%outfile
    k[11] = '%.2f\n'%g
    k[13] = '%.2e\n'%_Rjup2m(rp)
    k[15] = '%.2e\n'%rvs.Rsun2m(Rs)
    k[17] = '%.2e\n'%_bar2Pa(10**logPc)
    k[19] = '%.2f\n'%rayleigh
    
    # Write file
    userInputfile = 'userInput_%s.in'%suffix
    userInputfile = userInputfile.replace('.','d').replace('din','.in')
    h = open('%s/%s'%(path2exotransmit, userInputfile), 'w')
    h.write(''.join(k))
    h.close()


def _Rjup2m(r):
    return r*69911e3

def _bar2Pa(P):
    return P*1e5


def _setup_chemistry(suffix, chems, scattering=True, CIA=True):
    '''
    Configure the Exo-Transmit chemistry file to include all the chemical 
    species given in chems.
    '''
    # Read in chemistry template
    f = open('%s/selectChem.in'%path2exotransmit, 'r')
    g = f.readlines()
    f.close()
    assert len(g) == 34
    
    # Add desired chemistry
    assert chems.size > 0
    for i in range(1,31):
        chem = g[i].split(' ')[0]
        g[i] = '%s = 1\n'%chem if chem in chems else '%s = 0\n'%chem

    # Include additional effects
    g[31] = 'Scattering = %i\n'%scattering
    g[32] = 'Collision Induced Absorption = %i\n'%CIA
        
    # Write file
    selectChemfile = 'selectChem_%s.in'%suffix
    selectChemfile = selectChemfile.replace('.','d').replace('din','.in')
    h = open('%s/%s'%(path2exotransmit, selectChemfile), 'w')
    h.write(''.join(g))
    h.close()
    return selectChemfile


def _create_TPfile(suffix, T):
    '''
    Create an isothermal TP profile to be read by ExoTransmit.
    '''
    # Read-in TP template
    f = open('%s/T_P/t_p_1000K.dat'%path2exotransmit, 'r')
    g = f.read()
    f.close()

    # Set temperature structure
    g = g.replace('1.0000000e+03', '%.7e'%T)
    
    # Write TP file
    TPfile = 't_p_%s.dat'%suffix
    h = open('%s/T_P/%s'%(path2exotransmit, TPfile), 'w')
    h.write(g)    
    h.close()
    return TPfile


def  _create_EOSfile(suffix, chems, logXs):
    '''
    Create an EOS file of chemical mixing ratios to be read by ExoTransmit.
    Assume all unspecified mixing ratios are zero with the exception of H2 and 
    He at the solar mass fractions (i.e. X=.7381, Y=0.2485)
    '''
    # get mixing ratios including H and He
    chems = np.ascontiguousarray(chems)
    assert chems.size == logXs.size
    assert np.all(np.in1d(['H','He'], chems, invert=True))
    Xs = 10**logXs
    assert Xs.sum() <= 1
    XH_XHe, XZ = .7381 / .2485, Xs.sum()
    XHe = (1-XZ) / (1+XH_XHe)
    XH = XHe * XH_XHe
    assert XH + XHe + XZ
    
    # Read-in EOS template
    f = open('%s/EOS/eos_1Xsolar_gas.dat'%path2exotransmit, 'r')
    g = f.readlines()
    f.close()

    # get column names
    header = np.array(g[0].split('\t\t'))
    assert np.all(np.in1d(chems, header))
                          
    # add uniformly-mixed chemicals
    for i in range(2,len(g)):
        if len(g[i]) == 1:
            pass
        elif len(g[i]) == 13:
            pass
        elif len(g[i].split('\t')) == header.size:
            ls = g[i].split('\t')
            for j in range(2,header.size-1):
                if header[j] in chems:
                    ls[j] = '%.6e'%Xs[chems == header[j]]
                elif header[j] == 'H2':
                    ls[j] = '%.6e'%XH
                elif header[j] == 'He':
                    ls[j] = '%.6e'%XHe
                else:
                    ls[j] = '%.6e'%0
            g[i] = '\t'.join(ls)
        else:
            raise ValueError('Found a weird row.')

    # save EOS file
    EOSfile = 'eos_%s.dat'%suffix
    EOSfile = EOSfile.replace('.','d').replace('ddat','.dat')
    h = open('%s/EOS/%s'%(path2exotransmit, EOSfile), 'w')
    h.write(''.join(g))
    h.close()
    return EOSfile


def _compile_exotransmit():
    '''
    Compile the ExoTransmit C code.
    '''
    cwd = os.getcwd()
    os.chdir(path2exotransmit)
    os.system('make clean')
    os.system('make')
    os.chdir(cwd)


def _get_transmission_spectrum(outfile, Nwl, sigma=5):
    '''
    Get the ExoTransmit transmission spectrum at a reduced resolution.
    '''
    # get exotransmit spectrum (spectrum in transit depth percent)
    wl, spectrum = np.loadtxt('%s/%s'%(path2exotransmit, outfile),
                              skiprows=2).T 
    wl *= 1e6  # m -> microns

    # some exotransmit runs fail for some reason
    if np.any(np.isnan(spectrum)):
        raise ValueError('ExoTransmit failed: bad input parameters')
    
    # reduce resolution spectrum
    spectrum_conv = gaussian_filter1d(spectrum, sigma)

    # resample wl grid
    wl2 = wl[np.unique(np.arange(0, wl.size, wl.size/float(Nwl)).astype(int))]
    fint = interp1d(wl, spectrum_conv)
    spectrum_conv2 = fint(wl2)
    
    return wl2, spectrum_conv2
    

def _run_exotransmit(suffix):
    '''
    Run exotransmit once using the input parameters associated with the 
    file suffix.
    '''
    cwd = os.getcwd()
    os.chdir(path2exotransmit)
    userInputfile = 'userInput_%s.in'%suffix
    userInputfile = userInputfile.replace('.','d').replace('din','.in')
    selectChemfile = 'selectChem_%s.in'%suffix
    selectChemfile = selectChemfile.replace('.','d').replace('din','.in')
    os.system('./Exo_Transmit %s %s'%(userInputfile, selectChemfile))
    os.chdir(cwd)


def _clean_up(prefix, suffix):
    '''
    Remove the temporary files that were created when computing the 
    transmission spectrum.
    '''
    os.system('rm %s/userInput_%s.in'%(path2exotransmit, suffix))
    os.system('rm %s/selectChem_%s.in'%(path2exotransmit, suffix))
    os.system('rm %s/T_P/t_p_%s.dat'%(path2exotransmit, suffix))
    os.system('rm %s/EOS/eos_%s.dat'%(path2exotransmit, suffix))
    os.system('rm %s/Spectra/%s_%s.dat'%(path2exotransmit, prefix, suffix))

    
def _save_results(prefix, Ts, gs, rps, Rss, logPcs, rayleighs, chems, logXss,
                  wl, specs):
    '''
    Save the input parameter values and ExoTransmit spectra for a set of 
    calculations.
    '''
    assert logXss.shape == (Ts.size, chems.size)
    assert specs.shape == (Ts.size, wl.size)
    params = np.array([Ts, gs, rps, Rss, logPcs, rayleighs]).T
    params = np.append(params, logXss, axis=1)
    chemlabels = ['logX_%s'%c for c in chems]
    labels = np.append(['T [K]','g [m/s^2]','rp [RJup]','Rs [RSun]',
                        'logPc [log(bar)]', 'rayleigh [N/A]'], chemlabels)
    try:
        os.mkdir(path2output)
    except OSError:
        pass
    np.save('%s/%s_InputParams'%(path2output, prefix), params)
    np.save('%s/%s_InputParamsLabels'%(path2output, prefix), labels)
    np.save('%s/%s_Wavelength'%(path2output, prefix), wl)
    np.save('%s/%s_Spectra'%(path2output, prefix), specs)
    np.save('%s/%s_SpectraLabels'%(path2output, prefix),
            np.array(['Wavelength [microns]','Transit depth (percent)']))


if __name__ == '__main__':
    prefix = sys.argv[1]
    N = int(sys.argv[2])
    create_ExoTransmit_spectral_set(prefix, N, chems=['TiO','Na','K','H2O'])
