from imports import *
import george

global kernel_names
kernel_names = ['SE','M32','M52','S2','QP']


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
        
        self.params, self.targets = samples, targets
	self.Nsamples, self.Nparams = self.params.shape
	assert self.targets.size == self.Nsamples
        self._kernel_name = kernel

        if len(lnhyperparams) > 0:
            _initialize_GP(self)


    def _initialize_GP(self, lnhyperparams):
        '''
        Initialize the Nparam-dimensional GP given a set of hyperparameters.

        Arguments
        ---------
        `lnhyperparams` : 1d array-like (Nhyperparams,)
             List of the logarithmic GP hyperparameters: log(covariance model 
             parameters and amplitude). The number of hyperparameters will 
             depend on the covariance kernel. For example the squared 
             exponential kernel requires Nparams+1 hyperparameters: 
            log(lengthscales of each model parameter and amplitude)

        Returns
        -------
        `GP': george.GP object
            The gp object of the model parameters using a specified kernel 
            and set of lnhyperparameters
        '''
        self.lnhyperparams = np.ascontiguousarray(lnhyperparams)
        self.Nhyperparams = self.lnhyperparams.size
        if self._kernel_name == 'SE':
            assert self.Nhyperparams == self.Nparams+1
            ls = np.exp(self.lnhyperparams[:self.Nparams])
            k = george.kernels.ExpSquaredKernel(ls, ndim=self.Nparams)

        elif self._kernel_name == 'M32':
            assert self.Nhyperparams == self.Nparams+1
            ls = np.exp(self.lnhyperparams[:self.Nparams])
            k = george.kernels.Matern32Kernel(ls, ndim=self.Nparams)

        elif self._kernel_name == 'M52':
            assert self.Nhyperparams == self.Nparams+1
            ls = np.exp(self.lnhyperparams[:self.Nparams])
            k = george.kernels.Matern52Kernel(ls, ndim=self.Nparams)

        elif self._kernel_name == 'S2':
            assert self.Nhyperparams == self.Nparams*2+1
            Gs = np.exp(self.lnhyperparams[:self.Nparams])
            Ps = np.exp(self.lnhyperparams[self.Nparams:2*self.Nparams])
            k = george.kernels.ExpSine2Kernel(Gs, Ps, ndim=self.Nparams)

        elif self._kernel_name == 'QP':
            assert self.Nhyperparams == self.Nparams*3+1
            ls = np.exp(self.lnhyperparams[:self.Nparams])
            Gs = np.exp(self.lnhyperparams[self.Nparams:2*self.Nparams])
            Ps = np.exp(self.lnhyperparams[2*self.Nparams:3*self.Nparams])
            k = george.kernels.ExpSquaredKernel(ls, ndim=self.Nparams) + \
                george.kernels.ExpSine2Kernel(Gs, Ps, ndim=self.Nparams)

        else:
            kernels = ', '.join(kernel_names)
            raise ValueError('Unknown kernel. Must be one of %s'%kernels)

        a = np.exp(self.lnhyperparams[-1])
        self.GP = george.GP(a * k)
        #self.GP.compute(self.params, )
        #bnds = ((-20,0),(-3,10),(-5,5),(-3,10))
        #results = gp.optimize(tbin, fbin, efbin, **{'bounds':bnds})



    def learn_hyperparams(self, lnhyperparams0, Ntries=1, lnmaxvar=1,
                          **kwargs):
        '''
        GP hyperparameter optimization method by maximizing the lnlikelihood 
        using scipy.optimize.

        Arguments
        ---------
        `lnhyperparams0` : 1d array-like (Nhyperparams,)
             List of initial guesses of the logarithmic GP hyperparameters; 
             log(covariance model parameters, amplitude, and diagonal 
             covariance term). If unspecified, need to run 
             GPEmulator.learn_hyperparams to get hyperparameters and initialize 
             the GP
        `Ntries` : scalar
            Number of attempts at optimizing the hyperparameters. For each 
            attempt the initial guess `lnhyperparams0` is perturbed and used 
            as a new initial guess in the optimization algorthim. The 
            optimized set of hyperparameters which maximize the lnlikelihood is 
            selected
        `lnmaxvar`: scalar or array-like (Nhyperparams,)
            If Ntries > 1, the maximum variance permitted when resampling the 
            lnhyperparameters. The resampling will perturb each 
            lnhyperparameter by a uniformly sampled value from  
            U(-lnmaxvar, lnmaxvar)

        Returns
        -------
        `lnhyperparams` : 1d array-like
            The optimized set of logarithmic hyperparameters; log(covariance 
            model parameters, amplitude, diagonal covariance term)

        '''
        self._initialize_GP(lnhyperparams0)

        Ntries = int(Ntries) 
        lnhyperparams, lnlikes = np.zeros((Ntries,self.Nhyperparams)), \
                                 np.zeros(Ntries)
        attempt, lnmaxvar = 0, np.ascontiguousarray(lnmaxvar)
        while attempt < Ntries:
	
            lnhyperparams0_try = lnhyperparams0
            
            # resample the initial guess 
            if attempt > 0:
                if lnmaxvar.size == 1:
                    lnperturb = np.random.uniform(-lnmaxvar, lnmaxvar,
                                                self.Nhyperparams)
                else:
                    assert lnmaxvar.size == self.Nhyperparams
                    lnperturb = np.array([np.random.uniform(-i,i)
                                          for i in lnmaxvar]) 
                lnhyperparams0_try += lnperturb
                
            # optimize the hyperparameters
            try:
                self._initialize_GP(lnhyperparams0_try)
                results = self.GP.optimize(self.params, self.targets, **kwargs)
                lnhyperparams[attempt] = results[0]
                self._optimization_successful = True

            except ValueError:
                lnhyperparams[attempt] = lnhyperparams0_try
                warnings.warn('\nGP hyperparameter optimization failed. '+ \
                              '\nAdopting the initialized ' + \
                              'lnhyperparameters which may lead to a ' + \
                              'poor spectral prediction.', Warning)

            # save lnlikelihood
            lnlikes[attempt] = self.GP.lnlikelihood(self.targets, quiet=True)
	    attempt += 1

        # retain max likelihood hyperparameter set
        g = lnlikes == lnlikes.max()
        if g.sum() == 1:
            lnhyperparams = lnhyperparams[g]
            self._optimization_successful = True
        elif g.sum() > 1:
            lnhyperparams = lnhyperparams[g][0]
            self._optimization_successful = True
        else:
            lnhyperparams = np.ascontiguousarray(lnhyperparams0)
            self._optimization_successful = False

        self._initialize_GP(lnhyperparams)


    def emulate(self, params, mean_only=False):
        '''
        Method to emulate the target (i.e. basis function) weight coefficient 
        using the GP emulator.

        Arguments
        ---------
        `params` : 1d array (Nparams,)
            List of model parameters whose spectrum you wish to emulate.
            The GP emulator will predict the weight coefficient of this 
            object's basis function (i.e. target)
        `mean_only` : bool
            If True, return the mean of the GP emulator. Otherwise, return 
            the mean prediction and its standard deviation

        Returns
        -------
        `mu` : float
            The mean prediction of the GP emulator
        `std` : float
            The standard deviation of the prediction of the GP emulator. 
            Returned only if mean_only == False

        '''
        params = np.ascontiguousarray(params)
        assert params.size == self.Nparams

        p = self.GP.predict(self.targets, params.reshape(1,self.Nparams),
                            mean_only=mean_only)
        if mean_only:
            return p
        else:
            mu, cov = p
            return mu, np.sqrt(cov)
