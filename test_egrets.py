from GetTrainingTestingSpectra import *
from egrets import *
from PyAstronomy.pyasl import instrBroadGaussFast
import time

# Get data (T, g, rp, X, Rs)
suffix = '0100'
Ntrain, Ntest = 2000, 100
#wl,trs,tes,trp,tep = sample_ExoTransmitspectra_good(Ntrain, Ntest)
d = np.loadtxt('TrainingData/TrainingSpectraPost_%s.dat'%suffix)
d = d[:,d[0]>=.6]
wl = d[0]
trs, tes = d[1:Ntrain+1], d[Ntrain+1:Ntrain+1+Ntest]
d = np.loadtxt('TrainingData/TrainingParameters_%s.dat'%suffix)
# convert Xis and Psurf to log units
d[:,:6] = np.log10(d[:,:6])
trp, tep = d[:Ntrain], d[Ntrain:Ntrain+Ntest]
# keep finite spectra
keeptrain = np.unique(np.where(np.isfinite(trs))[0])
trs, trp = trs[keeptrain], trp[keeptrain]
keeptest = np.unique(np.where(np.isfinite(tes))[0])
tes, tep = tes[keeptest], tep[keeptest]

# shuffle
#inds = np.arange(trs.shape[0])
#np.random.shuffle(inds)
#trs, trp = trs[inds], trp[inds]
    
# Reduce spectral resolution
##R = 5e2
##fwhm = wl.mean() / R
##sigma = fwhm / (2*np.sqrt(2*np.log(2)))
'''sigma = 2
trs2 = np.zeros(trs.shape)
for i in range(trs.shape[0]):
    trs2[i] = gaussian_filter1d(trs[i], sigma)
tes2 = np.zeros(tes.shape)
for i in range(tes.shape[0]):
    tes2[i] = gaussian_filter1d(tes[i], sigma)

# Reduce spectral sampling
inds = range(0, wl.size, 2)
wl2, trs2, tes2 = wl[inds], trs2[:,inds], tes2[:,inds]
'''

build = 1
# Construct an `egret`; i.e. a spectral emulator
if build:
    t0 = time.time()
    self = EGRETS(wl, trs, trp, tes, tep, Npc=10, factr=1e1,
                  verbose=1)
    print 'building spectral emulator took %.3e minutes'%((time.time()-t0)/60.)
    dump_pickle('SavedSpectralEmulators/SpectralexampleBIG_%s'%suffix, self)

# Load saved `egret` instead
else:
    self = load_pickle('SavedSpectralEmulators/Spectralexample_%s'%suffix)
