from imports import *
import GPEmulator
from save_load_Emulators import *


class EGRETS(object):

    
    def __init__(self, WLgrid, training_spectra, training_samples,
                 testing_spectra=None, testing_samples=None, Ntries=3,
                 varthresh=.99, Npc=0, factr=10, pgtol=1e-20, 
		 optimizeGPemulators=True, verbose=False):
        '''
        Construct a Gaussian process emulator of a radiative transfer 
        model (i.e. the simulator) given a set of input spectra, computed 
        using the simulator, to train on. Can also do validation if testing 
        spectra are given.

        Parameters
        ----------
        `WLgrid` : 1d array (Nwl,)
            Numpy array of observed wavelengths.
        `training_spectra` : 2d array (Ntrain, Nwl)
            Numpy array of model spectra from the simulator function to train
            the GP emulator on. Ntrain should be large; >~ 100. Units should 
            be transit depth in per cent; (rp/Rs)^2 * 100% 
        `training_samples` : 2d array  (Ntrain, Nparams)
            Numpy array of model parameters used to compute the model training 
            spectra in `training_spectra`.
        `testing_spectra` : 2d array (Ntest, Nwl)
            Numpy array of model spectra from the simulator function to test
            the GP emulator on. Units should be transit depth in per cent; 
            (rp/Rs)^2 * 100%
        `testing_samples` : 2d array (Ntest, Nparams)
            Numpy array of model parameters used to compute the model testing 
            spectra in `testing_spectra`.
        `Ntries` : int
            Number of attempts to optimize the hyperparameters of the GP 
            emulator after resampling the initial guess of each hyperparameter 
            in each iteration.
        `varthresh` : float
            A value between 0 and 1 indicating the fraction of the total 
            variance that the user wishes to be encapsulated by the retained 
            principal components. Alternatively the user can set Npc directly 
            rather than the desired variance.
        `Npc` : int
            The number of principal components to retain. If Npc > 0 then it 
            over-writes the value of varthresh.
        `factr` : float
            Determines when the optimization algorithm is terminated. Typical 
            values for factr are: 1e12 for low accuracy; 1e7 for moderate 
            accuracy; 10.0 for extremely high accuracy.
        `pgtol` : float
            Sets the stopping condition for the GP hyperparameter optimization 
            routine.
        `verbose` : bool
            Controls the frequency of printed output to the shell when learning 
            the GP hyperparameters. Set `verbose` to False to prevent such 
            output.

        '''
        # Get training spectra
        self.WLgrid, self.Nwl = WLgrid, WLgrid.size
        self.training_spectra, self.training_samples = training_spectra, \
                                                       training_samples

        # Scale training spectra to min == zero
        self._training_opaque_depths = np.min(self.training_spectra, axis=1)
        self.training_spectra = (self.training_spectra.T - \
                                 self._training_opaque_depths).T

        # Define constants
        self.Ntrain, self.Nparams = self.training_samples.shape
        self._varthresh = varthresh if 0 < varthresh <= 1 else .99
        self.Npc, self._factr, self._pgtol= int(Npc), factr, pgtol

        # Repeat above for testing set if available
        if (testing_spectra is not None) and (testing_samples is not None):
            self.testing_spectra, self.testing_samples = testing_spectra, \
                                                         testing_samples
            self.Ntest = self.testing_spectra.shape[0]
	    self._testing_opaque_depths = np.min(self.testing_spectra, axis=1)
            self.testing_spectra = (self.testing_spectra.T - \
            			    self._testing_opaque_depths).T
        else:
            self.Ntest = 0

        # Scale model parameters to zero mean and unit std 
        self._scale_input_samples()

        # Do PCA
        self._decompose_spectra()

        # Get optimized GP emulator of the basis functions
	if optimizeGPemulators:
            self._get_GPemulators(Ntries=Ntries, factr=self._factr,
                              	  pgtol=self._pgtol, verbose=verbose)



    def _scale_input_samples(self):
        '''
        Scale the model parameter samples in the training and testing sets to 
        mean zero and unit standard deviation.

        Returns
        -------
        `_training_samples_means` : array-like (Nparams,)
            List of training model parameter means. Useful for converting 
            scaled parameters back to their input physical units.
        `_training_samples_stds` : array-like (Nparams,)
            List of training model parameter standard deviations. Useful for 
            converting scaled parameters back to their input physical units.
        `training_samples` : array-like (Ntrain,Nparams,)
            Numpy array of re-scaled model parameters used to compute the 
            model training spectra in `training_spectra`.
        `_testing_samples_means` : array-like (Nparams,)
            See `_training_samples_means`.
        `_testing_samples_stds` : array-like (Nparams,)
            See `_training_samples_stds`.
        `testing_samples` : array-like (Nparams,)
            See `training_samples`.

        '''
        self._training_samples_means, self._training_samples_stds = \
                                        self._scale_arr(self.training_samples)
        self.training_samples = self.scale_samples(self.training_samples,
                                                   self._training_samples_means,
                                                   self._training_samples_stds)
        
        if self.Ntest > 0:
            self._testing_samples_means, self._testing_samples_stds = \
                                        self._scale_arr(self.testing_samples)
            self.testing_samples = self.scale_samples(self.testing_samples,
                                                      self._testing_samples_means,
                                                      self._testing_samples_stds)



    def _scale_arr(self, samples):
        '''
        Get the means and standard deviations of each column in an input array 
        for scaling.
        
        '''
        mus, sigs = np.mean(samples, axis=0), np.std(samples, axis=0)

	if np.any(sigs == 0):
	    raise ValueError('Need some dispersion in the samples ' + \
			     'to compute the parameter scaling.')

	return mus, sigs



    def scale_samples(self, samples, means=[], stds=[]):
        '''
        Scale the columns of the input parameter samples from physical units 
        to mean zero and unit standard deviation.

        Parameters
        ----------
        `samples` : array-like (Nparams,) or (Nsamples, Nparams,)
            Array of parameters whose columns are to be scaled to mean zero 
            and unit standard deviation.
        `means` : array-like (Nparams,)
            List of model parameter means from the training samples.
        `stds` : array-like (Nparams,)
            List of model parameter standard deviations from the training 
            samples.

        Returns
        -------
        `scaled_samples` : array-like (Nparams,) or (Nsamples, Nparams,)
            Array of scaled parameter samples with the same shape as `samples`.

        '''
        means, stds = np.ascontiguousarray(means), np.ascontiguousarray(stds)
        
        if means.size == 0 or stds.size == 0:
            return (samples - self._training_samples_means) / \
                self._training_samples_stds
        else:
            return (samples - means) / stds



    def inverse_scale_samples(self, scaled_samples, means=[], stds=[]):
        '''Re-scale the columns of an input array of scaled  parameter samples 
        to physical units using either input means and standard deviations 
        or those from the training samples.

        Parameters
        ----------
        `scaled_samples` : array-like (Nparams,) or (Nsamples, Nparams,)
            Array of scaled parameters to scale to physical units given the 
            parameter means and standard deviations.
        `means` : array-like (Nparams,)
            List of model parameter means from the training samples.
        `stds` : array-like (Nparams,)
            List of model parameter standard deviations from the training 
            samples.

        Returns
        -------
        `samples` : array-like (Nparams,) or (Nsamples, Nparams,)
            Array of parameter samples scaled to physical units. Output has the
            same shape as `scaled_samples`.

        Example
        -------
        >>> print self.training_samples[0]
        >>> array([ 0.21195, -0.29068,  0.60322,  0.33398, -0.48876])
        >>> print self.inverse_scale_samples(self.training_samples[0],
                                             self._training_samples_means,)
                                             self._training_samples_stds)
        >>> array([  7.000e+02, 1.183e+01, 2.858e+00, 1.477e+00, 2.780e-01])

        '''
        means, stds = np.array(means), np.array(stds)
        
        if means.size == 0 or stds.size == 0:
            return scaled_samples * self._training_samples_stds + \
                self._training_samples_means        
        else:
            return scaled_samples * stds + means



    def _decompose_spectra(self):
        '''
        Decompose the training set of spectra into principal components (PCs)
        for dimensionality reduction using SVD (see np.linalg.svd).
        The number of PCs retained is determined by the the number of PCs
        that are required to encapsulate `varthresh` of the total variance or 
        are fixed to the value of `Npc`. 
            
        Returns
        -------
        `self.basis_variances` : 1d array (Nwl,)
            The cumulative variance contained in the principal components from 
            the first to the Nwl^th.
        `self.basis_functions` : 2d array (Npc, Nwl,)
            The basis functions from PCA decomposition.
        `self.training_spectra_PCs` : 2d array (Npc, Nwl,)
            Projection of the data onto the basis functions; i.e. the 
            principal components.
        `self._varthresh` : float
            Variance encapsulated by the retained principle components.
        `self.Npc` : int
            Number of retained PCs (<= Ntrain).
        
        '''
        _,s,V = np.linalg.svd(self.training_spectra, full_matrices=True)
	self.basis_variances = s.cumsum() / s.sum()

        if self.Npc == 0:
            ind = np.where(self.basis_variances >= self._varthresh)[0][0] + 1
        else:
            ind = self.Npc
            self._varthresh = self.basis_variances[:ind][-1]

        self.basis_functions = V[:ind]
        self.training_spectra_PCs = np.dot(self.training_spectra,
                                           self.basis_functions.T).T
        self.Npc = self.basis_functions.shape[0]


    def _get_GPemulators(self, Ntries=3, factr=10, pgtol=1e-20,
			 verbose=False):
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
            hyperparameters which maximize the lnlikelihood is selected.
        `factr` : float
            Determines when the optimization algorithm is terminated. Typical 
            values for factr are: 1e12 for low accuracy; 1e7 for moderate 
            accuracy; 10.0 for extremely high accuracy.
	`pgtol` : float
	    Sets the stopping condition for the GP hyperparameter optimization 
	    routine.
        `verbose` : bool
            Controls the frequency of printed output to the shell. Set 
            `verbose` to False to prevent such output.

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
        self.emulators, self._optimization_success_fraction = [], 0.
        for i in xrange(self.Npc):
            print 'PC %i of %i...'%(i+1, self.Npc)
            self.emulators.append(GPEmulator.GPEmulator(self.training_samples,
				  self.training_spectra_PCs[i]))

            # Compute hyperparameters
            lnhyperparams0 = 5 * (np.random.rand(self.Nparams + 2) - .5)
            self.emulators[i].learn_hyperparams(lnhyperparams0, Ntries=Ntries,
                                                factr=factr, pgtol=pgtol,
						verbose=verbose)
	    self._optimization_success_fraction += self.emulators[i]._optimization_successful
     	self._optimization_success_fraction /= self.Npc



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
