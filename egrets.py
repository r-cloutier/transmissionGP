from imports import *
from GPEmulator import *

#samples = params

class EGRETS():

    
    def __init__(self, wl, spectra, params, Ntest=0,
                 Ntries=10, varthresh=.99, Npc=0, factr=10, pgtol=1e-20, 
		 optimizeGPemulators=True, verbose=False):
        '''
        Construct a Gaussian process emulator of a radiative transfer 
        model (i.e. the simulator) given a set of input spectra to train on. 
        
        Parameters
        ----------
        `wl` : 1d array (Nwl,)
            Wavelength array in microns
        `spectra` : 2d array (Nspec, Nwl)
            Array of model spectra (in transit depth percent) from the 
            simulator function to train/test the GP emulator on. Nspec should 
            be large for a more accurate and comprehensive emulator
        `params` : 2d array  (Nspec, Nparams)
            Array of model parameters used to compute the spectra. The 
            parameters are used as input in the emulator to reproduce a 
            transmission spectrum
        `Ntest': int
            The number of spectra to be used for testing. The remaining spectra 
            are allocated to the training set (i.e. Nspec = Ntest + Ntrain)
        `varthresh` : scalar
            A value between 0 and 1 indicating the fraction of the total 
            variance that the user wishes to be encapsulated by the retained 
            principal components. Alternatively, the user can set Npc directly 
            rather than the desired variance.
        `Npc` : int
            The number of principal components to retain. If Npc > 0 then it 
            over-writes the value of varthresh and retains at most Npc PCs
        `factr` : float
            Determines when the optimization algorithm is terminated. Typical 
            values for factr are: 1e12 for low accuracy; 1e7 for moderate 
            accuracy; 10.0 for extremely high accuracy.
        `pgtol` : float
            Sets the stopping condition for the GP hyperparameter optimization 
            routine.
        `optimizeGPemulators': bool
            If True, try to optimize the GP hyperparameters describing the 
            training set
        `Ntries` : int
            Number of attempts to optimize the hyperparameters of the GP 
            emulator after resampling the initial guess of each hyperparameter 
            in each iteration.
        `verbose` : bool
            Controls the frequency of printed output to the shell when learning 
            the GP hyperparameters. Set `verbose` to False to prevent such 
            output.

        '''
        self.wl, self.Nwl = wl, wl.size
        self.Nparams, self.Ntest = params.shape[1], int(Ntest)
        if (self.Ntest > 0) & (self.Ntest < spectra.shape[0]):
            self.testing_spectra, self.testing_params = spectra[:Ntest], \
                                                        params[:Ntest]
            self.training_spectra, self.training_params = spectra[Ntest:], \
                                                          params[:Ntest]
        else:
            self.testing_spectra = np.zeros((0,self.Nwl))
            self.testing_params  = np.zeros((0,self.Nparams))
            self.training_spectra, self.training_params = spectra, params       
            
        # define constants
        self.Ntrain = self.training_spectra.shape[0]
        assert self.training_spectra.shape == (self.Ntrain, self.Nwl)
        assert self.training_params.shape == (self.Ntrain, self.Nparams)
        assert self.testing_spectra.shape == (self.Ntest, self.Nwl)
        assert self.testing_params.shape == (self.Ntest, self.Nparams)

        self._varthresh = varthresh if 0 < varthresh <= 1 else .99
        self.Npc, self._factr, self._pgtol= int(Npc), factr, pgtol
        assert self.Npc <= self.Ntrain
        
        # shift training spectra to min = zero
        #self._training_spectra_opaque_depths = np.min(self.training_spectra,
        #                                              axis=1)
        #self.training_spectra = (self.training_spectra.T - \
        #                         self._training_spectra_opaque_depths).T
        
        # scale model parameters to zero mean and unit std 
        #self._scale_input_params()

        # do PCA
        #self._decompose_spectra()

        # Get optimized GP emulator of the basis functions
	#if optimizeGPemulators:
        #    self._get_GPemulators(Ntries=Ntries, factr=self._factr,
        #                      	  pgtol=self._pgtol, verbose=verbose)



    def _scale_input_params(self):
        '''
        Scale the input model parameters (training and testing) to mean zero 
        and unit standard deviation.

        Returns
        -------
        `_training_params_means` : 1d array (Nparams,)
            List of training model parameter means. Useful for converting 
            scaled parameters back to their input physical units.
        `_training_params_stds` : 1d array (Nparams,)
            List of training model parameter standard deviations. Useful for 
            converting scaled parameters back to their input physical units.
        `training_params` : 2d array (Ntrain, Nparams)
            Array of training model parameters rescaled to zero mean and unit 
            standard deviation
        `_testing_params_means` : 1d array (Nparams,)
            List of testing model parameter means. Useful for converting 
            scaled parameters back to their input physical units.
        `_testing_params_stds` : 1d array (Nparams,)
            List of testing model parameter standard deviations. Useful for 
            converting scaled parameters back to their input physical units.
        `testing_params` : 2d array (Ntest, Nparams)
            Array of testing model parameters rescaled to zero mean and unit 
            standard deviation
        '''
        self._training_params_means, \
        self._training_params_stds, \
        self.training_params_scaled = scale_params(self.training_params)
        self._testing_params_means, \
        self._testing_params_stds, \
        self.testing_params_scaled = scale_params(self.testing_params)


    def _decompose_spectra(self):
        '''
        Decompose the training spectra into principal components (PCs)
        for dimensionality reduction using SVD (see np.linalg.svd).
        The number of PCs retained is determined either by the the number of PCs
        that are required to encapsulate `varthresh` of the total variance or 
        are fixed to the value of `Npc` if non-zero. 
            
        Returns
        -------
        `self.basis_variances` : 1d array (Nwl,)
            The cumulative variance contained in the principal components from 
            the first to the Nwl^th.
        `self.basis_functions` : 2d array (Npc, Nwl)
            The basis functions from PCA decomposition
        `self.training_spectra_PCs` : 2d array (Npc, Ntrain)
            Projection of the data onto the basis functions; i.e. the 
            principal components
        `self._varthresh` : float
            Variance encapsulated by the retained PCs
        `self.Npc` : int
            Number of retained PCs (<= Ntrain)

        '''
        _,s,V = np.linalg.svd(self.training_spectra, full_matrices=True)
	self.basis_variances = s.cumsum() / s.sum()
        assert self.basis_variances.size == self.Nwl

        if self.Npc == 0:
            ind = np.where(self.basis_variances >= self._varthresh)[0][0] + 1
            self.Npc = ind
        else:
            ind = self.Npc
            self._varthresh = self.basis_variances[:ind][-1]

        self.basis_functions = V[:ind]
        assert self.basis_functions.shape == (self.Npc, self.Nwl)

        self.training_spectra_PCs = np.dot(self.training_spectra,
                                           self.basis_functions.T).T
        assert self.training_spectra_PCs.shape == (self.Npc, self.Ntrain)


    def _get_GPemulators(self, Ntries=3, factr=10, pgtol=1e-20,
			 kernel='SE'):
        '''
        Create a GP emulator for each principal component found via SVD (see 
        self._decompose_spectra). The hyperparameters of each emulator are 
        then optimized and can be used for predictive purposes.

        Parameters
        ----------
        `Ntries` : int
            Number of attempts at optimizing the hyperparameters. For each 
            attempt the initial guess is perturbed and used as a new initial 
            guess in the optimization algorthim. The optimized set of 
            hyperparameters which maximize the lnlikelihood is selected
        `kernel` : str
            String corresponding to the type of assumed covariance kernel.
            SUpported values are given in the `kernel_names` global variable

        Returns
        -------
        `emulators` : list (Npc,)
            List of GPEmulator objects for each principal component (see 
            GPEmulator.GPEmulator).

        '''
        if not hasattr(self, 'basis_functions') or not hasattr(self, 'Npc'):
            raise AttributeError('First run EGRETS._decompose_spectra.')

        # Define GP emulator to each basis function
        print '\nTraining GP emulators on %i principal components:\n'%self.Npc
        self.emulators = []
        for i in xrange(self.Npc):
            print 'PC %i of %i...'%(i+1, self.Npc)
            # initialize the emulator for this PC
            emulator = GPEmulator.GPEmulator(self.training_params,
				             self.training_spectra_PCs[i],
                                             kernel=kernel)
            self.emulators.append(emulator)

            # optimize the hyperparameters
            lnhyperparams0 = 5 * (np.random.rand(self.Nparams + 2) - .5)
            emulator.learn_hyperparams(lnhyperparams0, Ntries=Ntries,
                                       factr=factr, pgtol=pgtol,
				       verbose=verbose)
	    self._optimization_success_fraction += \
                                            emulator._optimization_successful

        # record the fraction of successful optimizations
        self._optimization_success = np.zeros(self.Npc, dtype=bool)
        for i in range(self.Npc):
            self._optimization_success[i] = emulator._optimization_successful
     	self._optimization_success_fraction = self._optimization_success.mean()



    def predict_spectrum(self, pred_sample):
        '''
        Predict the PC weights using the trained GP emulators of the 
        PCs. Then reconstruct the emulated spectrum from the weights.

        Parameters
        ----------
        `pred_sample` : array-like  (Nparams,)
            List of model parameters, in physical/unscaled units, whose 
            corresponding spectrum the user wishes to emulate using the trained 
            GP emulator. 
        
        Returns
        -------
        `pred_spec` : 1d array (Nwl,)
            Numpy array containing the predicted spectrum from the trained 
            GP emulator.
        `epred_spec` : 1d array (Nwl,)
            Numpy array containing the 1 sigma (68% confidence interval) 
            uncertainty on the predicted spectrum from the trained GP emulator.
        `coeffs` : 1d array (Npcs,)
            Numpy array of the emulated weighting coefficients for each  
            principal component.
        `ecoeffs` : 1d array (Npcs,)
            Numpy array of the uncertainties on the emulated weighting 
            coefficients for each principal component.

        Examples
        --------
        >>> T, rho, rp = 300, 10, 1.5
        >>> spectrum,_,_,_ = self.predict_spectrum([T,rho,rp])

        '''
        if not hasattr(self, 'emulators'):
            raise AttributeError('First run EGRETS.get_GPemulators.')

        self.pred_sample = np.ascontiguousarray(pred_sample)
        assert self.pred_sample.size == self.Nparams

        # Ensure that the input vector is within range
        scaled_training_samples = \
                        self.inverse_scale_samples(self.training_samples)
        for i in range(self.Nparams):
            lims = [scaled_training_samples[:,i].min(),
                    scaled_training_samples[:,i].max()]
            if self.pred_sample[i] < lims[0] or self.pred_sample[i] > lims[1]:
                raise ValueError('Element %i in pred_sample '%i + \
                                 '(%.3f) is outside '%self.pred_sample[i] + \
                                 'of the allowable range of ' + \
                                 '[%.3f,%.3f]'%(lims[0], lims[1]))

        # Scale to zero mean and unit std
        self.pred_sample_scaled = self.scale_samples(self.pred_sample)
        
        pred_spec, pred_espec = np.zeros(self.Nwl), np.zeros(self.Nwl)
        coeffs, ecoeffs = np.zeros(self.Npc), np.zeros(self.Npc)
	for i in xrange(self.Npc):
            coeffs[i], ecoeffs[i] = \
			self.emulators[i].predict(self.pred_sample_scaled)
            pred_spec  += coeffs[i] * self.basis_functions[i]
            pred_espec += ecoeffs[i] * self.basis_functions[i]

        return pred_spec, pred_espec, coeffs, ecoeffs


def scale_params(params):
    '''
    Scale the input parameters to mean zero and unit standard deviation.
    
    Parameters
    ----------
    `params': 2d array (N, Nparams)
        Model parameter sets to be scaled

    Returns
    -------
    `means': 1d array (Nparams,)
        The parameter mean values
    `stds': 1d array (Nparams,)
        The parameter standard deviations
    `params_scaled': 2d array (N, Nparams)
        The input array scaled to mean zero and unit standard deviation
    '''
    assert len(params.shape) == 2
    means, stds = np.mean(params, axis=0), np.std(params, axis=0)
    assert np.all(stds > 0)
    params_scaled = (params - means) / stds
    return means, stds, params_scaled


def inverse_scale_params(params_scaled, means, stds):
    '''
    Rescale the input parameters from mean zero and unit standard deviation to 
    their previous physical values.
    
    Parameters
    ----------
    `params_scaled': 2d array (N, Nparams)
        Scaled model parameters to be rescaled to physical values
    `means': 1d array (Nparams,)
        Mean parameter values
    `stds': 1d array (Nparams,)
        Parameter standard deviations

    Returns
    -------
    `params': 2d array (N, Nparams)
        The input array rescaled to physical values
    '''
    assert len(params.shape) == 2
    assert params.shape[1] == means.size
    assert params.shape[1] == stds.size
    return params_scaled * stds + means
