from imports import *


class GPEmulator(object):


    def __init__(self, params, targets, kernel='SE', lnhyperparams=[]):
	'''
	GP object is a GP with a unique set of hyperparameters 
	which can be given or found via the built-in optimization 
	routine.

	Arguments
	---------
	`params` : 2d array (Nsamples, Nparams)
	    Array of sampled model parameters; i.e. input for the Gaussian 
            process
	`targets` : 1d (Nsamples,)
	    Array of the target values; i.e. the PCs from projecting the 
	    simulator spectra onto their basis functions
	`kernel` : str
	    String corresponding to the type of assumed covariance kernel. 
            Supported values are given in the `kernel_names` global variable
        `lnhyperparams` : 1d array-like (Nhyperparams,)
             List of the logarithmic GP hyperparameters: log(covariance model 
             parameters and amplitude). If unspecified, need to run 
             GPEmulator.learn_hyperparams to get hyperparameters and initialize 
             the GP
	'''
        assert len(params.shape) == 2
        assert len(targets.shape) == 1
        
        self.params, self.targets = params, targets
	self.Nsamples, self.Nparams = self.params.shape
	assert self.targets.shape == (self.Nsamples,)
        self._kernel_name = kernel
        self.Nhyperparams = _define_Nhyperparams(self)

        if len(lnhyperparams) > 0:
            _initialize_GP(self)



    def _negloglike(self, lnhyperparams):
        '''
        Compute the negative of the log likelihood given a set of 
        hyperparameters.
        '''
        self.lnhyperparams = np.array(lnhyperparams)
        assert self.lnhyperparams.size == self.Nparams+2

        self._compute_covariance()

        return .5*(np.dot(self.targets.T, np.dot(self.Kinv, self.targets)) + \
                   self._logdetK + \
                   self.Nsamples * np.log(2*np.pi))

    
    def _partials(self, lnhyperparams):
        '''
        Compute the partial derivatives of the log likelihood with the 
        hyperparameters.
        '''
        self.lnhyperparams = np.array(lnhyperparams)
        assert self.lnhyperparams.size == self.Nparams+2
        
        partials = np.zeros(self.Nparams+2)
        self._compute_covariance()
        Kinvt = np.dot(self.Kinv, self.targets)
        
        for i in xrange(self.Nparams):
            V = (((np.tile(self.samples[:,i], (self.Nsamples, 1)) - \
                   np.tile(self.samples[:,i], (self.Nsamples, 1)).T))**2).T
            V *= self.Z
            Kinvt = np.dot(self.Kinv, self.targets)
            partials[i] = np.exp(self.lnhyperparams[i]) * .25 * \
                          (np.dot(Kinvt, np.dot(V, Kinvt)) - \
                           np.sum(self.Kinv * V))            
            
        partials[self.Nparams] = .5 * (np.sum(self.Kinv * self.Z) - \
                                       np.dot(Kinvt, np.dot(self.Z, Kinvt)))
        partials[self.Nparams+1] = .5 * \
                                   np.exp(self.lnhyperparams[self.Nparams+1])* \
                                   (np.trace(self.Kinv) - np.dot(Kinvt, Kinvt))
        return partials
    
        
    def learn_hyperparams(self, lnhyperparams0, Ntries=3, factr=10, pgtol=1e-20,
                          verbose=False):
        '''
        Optimize the GP hyperparameters by maximizing the lnlikelihood using 
        L-BFGS-B algorithm in scipy.optimize.

        Parameters
        ----------
        `lnhyperparams0` : 1d array-like (Nparams+2,)
             List of initial guesses of the logarithmic GP hyperparameters; 
             log(lengthscales of each model parameter, amplitude, diagonal 
             covariance term).
        `Ntries` : int
            Number of attempts at optimizing the hyperparameters. For each 
            attempt the initial guess `lnhyperparams0` is perturbed and used 
            as a new initial guess in the optimization algorthim. The 
            optimized set of hyperparameters which maximize the lnlikelihood is 
            selected.
        `factr` : float
            Determines when the optimization algorithm is terminated. Typical 
            values for factr are: 1e12 for low accuracy; 1e7 for moderate 
            accuracy; 10.0 for extremely high accuracy.
	`pgtol` : float
	    Sets the stopping condition for the optimization routine.
        `verbose` : bool
            Controls the frequency of printed output to the shell. Set verbose 
            to False to prevent such output.

        Returns
        -------
        `lnhyperparams` : array-like (Nparams+2,)
            The optimized set of logarithmic hyperparameters; log(lengthscales 
        of each model parameter, amplitude, diagonal covariance term). 

        '''
        lnhyperparams0 = np.array(lnhyperparams0)
        assert lnhyperparams0.size == self.Nparams+2

        iprint = 1 if verbose else -1
        
        #negloglikes, lnhyperparams = [], []
	attempt, self._optimization_successful = 0, False
        
	while (attempt < Ntries) and (not self._optimization_successful):
	
            lnhyperparams0_try = lnhyperparams0
            if attempt > 0:
                # resample the initial guess 
                lnhyperparams0_try = 5 * (np.random.rand(self.Nparams + 2) - .5)

            try: 
                self.lnhyperparams,_,d = fmin_l_bfgs_b(self._negloglike,
                                               	       lnhyperparams0_try,
                                                       fprime=self._partials,
                                                       factr=factr,
                                                       pgtol=pgtol,
                                                       iprint=iprint)
		#negloglikes.append(negll)

            	if d['warnflag'] > 0:
                    warnings.warn('\nGP hyperparameter optimization failed. '+ \
                                  '\nAdopting the initialized ' + \
                                  'lnhyperparameters which may lead to a ' + \
                                  'poor spectral prediction.', Warning)
		else:
		    self._optimization_successful = True

            except np.linalg.LinAlgError:
                self.lnhyperparams = lnhyperparams0_try
                #negloglikes.append(np.inf)
                warnings.warn('\nGP hyperparameter optimization failed. '+ \
                              '\nAdopting the initialized ' + \
                              'lnhyperparameters which may lead to a ' + \
                              'poor spectral prediction.', Warning)

	    #lnhyperparams.append(lnhp_opt)

	    attempt += 1

        # Find minimized neglnlike
        self.lnhyperparams = np.array(self.lnhyperparams)
        self._compute_covariance()

        
    def _compute_covariance(self):
        '''
        Given a set of log hyperparameters in `self.lnhyperparams`, compute the 
        covariance matrix K, the inverse of K, and log determinant of K.
        '''
	kernel_funcs = ['SE']
	if self.kernel not in kernel_funcs:
	    raise AttributeError('%s is not a known covariance ' + \
                                 'function.'%self.kernel)
	if self.kernel == 'SE':
	    self.K = self._cov_SE(self.samples, self.samples)

        L = np.linalg.cholesky(self.K)
        self.Kinv = np.linalg.inv(L.T).dot(np.linalg.inv(L))
        self._logdetK = 2. * np.sum(np.log(np.diag(L))) 


    def _cov_SE(self, x, xp):
        '''
        Compute the covariance matrix K using a squared-exponential kernel.
        '''
        if x.shape[0] != xp.shape[0] and x.shape[0] != 1:
            raise ValueError('`xp` must have the same number of entries as' + \
                             ' `x` or have only one entry for prediction.')
        Ktmp = np.zeros((x.shape[0], xp.shape[0]))
        for i in xrange(self.Nparams):
            if x.shape[0] == xp.shape[0]:
                xmx = abs(np.tile(x[:,i], (x.shape[0],1)) - \
                          np.tile(xp[:,i], (xp.shape[0],1)).T)
            else:
                xmx = abs(x[:,i].reshape(x.shape[0],1) - xp[:,i])
            Ktmp += np.exp(self.lnhyperparams[i]) * xmx**2
        self.Z = np.exp(self.lnhyperparams[self.Nparams]) * np.exp(-.5 * Ktmp)
        K = np.zeros_like(self.Z) + self.Z
        if x.shape[0] == xp.shape[0]:
            K += np.exp(self.lnhyperparams[self.Nparams+1]) * np.eye(x.shape[0])
        return K


    def predict(self, testsamples, mean_only=False):
        '''
        Predict the value of the target (basis function) weight coefficient 
        using the GP emulator.

        Parameters
        ----------
        `testsamples` : list or 1d array (Nparams,)
            List of model parameters whose spectrum you wish to emulate.
            The GP emulator will predict the weight coefficient of this 
            object's basis function (i.e. target).
        `mean_only` : boolean
            If True return the mean of the GP emulator. Otherwise, return 
            the mean prediction and its standard deviation.

        Returns
        -------
        `mu` : float
            The mean prediction of the GP emulator.
        `std` : float
            The standard deviation of the prediction of the GP emulator. 
            Returned only if mean_only == False.

        '''
        testsamples = np.array(testsamples)
        assert testsamples.size % self.Nparams == 0
        try:
            if len(testsamples.shape) != 2:
                testsamples = np.array(testsamples).reshape(1,len(testsamples))
        except AttributeError:
            testsamples = np.array(testsamples).reshape(1,len(testsamples))

        self._compute_covariance()
        if self.kernel == 'SE':
            Kp = self._cov_SE(testsamples, self.samples)

        a = np.exp(self.lnhyperparams[self.Nparams])
        # TEMP this is the same as Gomez
	mu = float(np.dot(Kp, np.dot(self.Kinv, self.targets))) 

        if mean_only:
            return mu
        else:
            # what is difference between these two methods of computing std?
            # TEMP this is the same as Gomez
            std = float(np.sqrt(a - np.sum(Kp.T * np.dot(self.Kinv, Kp.T),
                                           axis=0)))
            #Kpp = self._cov_SE(testsamples, testsamples)
            #std = np.sqrt(float(Kpp - np.dot(Kp, np.dot(self.Kinv, Kp.T))))
            return mu, std
