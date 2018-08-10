from imports import *
from egrets import *

global blue
blue = '#08519c'

def plot_model_emulator_scatter(EGRETS, model_spectrum, model_params,
				fname=''):
    '''
    Given an input spectrum with its corresponding set of model parameters 
    and EGRETS object, plot the scatter plot of the input model spectrum 
    and the predicted spectrum.

    Parameters
    ----------
    See predict_spectrum_and_scale.
    `fname` : str
	Set `fname` to the destination of the saved .png of this figure.
	If `fname` == '' then no .png is saved.

    '''
    model_spectrumplt, emulated_muplt, emulated_sigplt = \
    predict_spectrum_and_scale(EGRETS, model_spectrum, model_params)
    
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    ax.scatter(model_spectrumplt, emulated_muplt, s=20, facecolor='none',
               edgecolors='g', alpha=.5)
    dspecm, dspece = model_spectrum.max()*.1, emulated_muplt.max()*.1
    if dspecm > dspece:
	spectrumplt_tmp = np.append(np.append(-dspecm, model_spectrumplt),
				    model_spectrumplt.max()+dspecm)
    else:
	spectrumplt_tmp = np.append(np.append(-dspece, emulated_muplt),
                                    emulated_muplt.max()+dspece)
    ax.plot(np.sort(spectrumplt_tmp), np.sort(spectrumplt_tmp), 'k--', lw=1,
            label='y = x')

    m, b = np.polyfit(model_spectrumplt, emulated_muplt, 1)
    linfit = np.poly1d((m, b))
    ax.plot(np.sort(model_spectrumplt), linfit(np.sort(model_spectrumplt)),
            '-', c=blue, lw=2)
    ax.text(.1, .76, 'y = mx + b\nm = %.3f\nb = %.3f'%(m, b),
            color=blue, transform=ax.transAxes, fontsize=14)
    ax.legend(bbox_to_anchor=(.36, .99))

    ax.set_xlabel('Model Spectrum')
    ax.set_ylabel('Emulated Spectrum')
    ax.set_xlim((spectrumplt_tmp.min(), spectrumplt_tmp.max()))
    ax.set_ylim((spectrumplt_tmp.min(), spectrumplt_tmp.max()))
    ax.set_aspect(1./ax.get_data_ratio())

    plt.subplots_adjust(bottom=.12, left=.12, top=.94, right=.94)
    
    if fname != '':
	plt.savefig(fname)

    plt.show()


    
def plot_model_emulator_residuals(EGRETS, model_spectrum, model_params,
				  resfactor=1e3, fname=''):
    '''
    Given an input spectrum with its corresponding set of model parameters 
    and EGRETS object, plot the residual spectrum (O-C) vs. wavelength.

    Parameters
    ----------
    See predict_spectrum_and_scale.
    `resfactor` : scalar
        The multiplicative factor to scale the residuals to make plotting the 
        residual abiscca more readable.
    `fname` : str
        Set `fname` to the destination of the saved .png of this figure.
        If `fname` == '' then no .png is saved.

    '''
    model_spectrumplt, emulated_muplt, emulated_sigplt = \
    predict_spectrum_and_scale(EGRETS, model_spectrum, model_params)
    
    fig = plt.figure(figsize=(7,6))
    ax1 = fig.add_subplot(2,1,1)
    WL = EGRETS.WLgrid
    ax1.fill_between(WL, emulated_muplt-emulated_sigplt,
                     emulated_muplt+emulated_sigplt, color=blue, alpha=.5) 
    ax1.plot(WL, model_spectrumplt, 'k-', lw=.8)

    ax1.text(.03, .88, 'Model Spectrum', color='k', transform=ax1.transAxes)
    ax1.text(.03, .80, 'Emulated Spectrum', color=blue, transform=ax1.transAxes)

    ax2 = fig.add_subplot(2,1,2)
    resmu = (model_spectrumplt-emulated_muplt)*resfactor
    resp1s = (model_spectrumplt-emulated_muplt-emulated_sigplt)*resfactor
    resm1s = (model_spectrumplt-emulated_muplt+emulated_sigplt)*resfactor
    ax2.fill_between(WL, resp1s, resm1s, color=blue, alpha=.5)
    ax2.plot(WL, resmu, 'k-', lw=.8)
    ax2.text(.03, .88, 'RMS = %.3e'%(resmu.std()/resfactor), 
	     transform=ax2.transAxes)
    ax2.axhline(0, ls='--', color='k', lw=.9)
    
    ax1.set_ylabel('$\Delta$ Transit Depth\n(per cent)')
    ax1.set_xticklabels('')
    ax2.set_xlabel('Wavelength (microns)')
    ax2.set_ylabel('(Model - Emulated)\n'+r'$\times %.e$'%resfactor)
    
    plt.subplots_adjust(bottom=.12, left=.19, top=.97, right=.95, hspace=0)
    
    if fname != '':
	plt.savefig(fname)

    plt.show()
    


def predict_spectrum_and_scale(EGRETS, model_spectrum, model_params):
    '''    
    Given an input spectrum with its corresponding set of model parameters 
    and EGRETS object, predict the emulated spectrum, scale both spectra 
    to the opaque transit depth and return the spectra.
    
    Parameters
    ----------
    `EGRETS` : object
        Object containing the trained GP emulator. Used to predict the emulated
        spectrum given a set of model parameters (see `model_params`).
    `model_spectrum` : 1d array (Nwl,)
        Numpy array of the model spectrum computed using the model parameters 
        given in `model_params`. Note that the corresponding wavelength grid 
        must be equivalent to `EGRETS.WLgrid`.
    `model_params`: array-like (Nparams,)
        List of model parameters in physical units corresponding to  
	`model_spectrum` and used to predict the emulated spectrum.

    Returns
    -------
    `model_spectrum` : 1d array (Nwl,)
        Numpy array of the input model spectrum scaled to the opaque transit 
        depth.
    `emulated_mu` : 1d array (Nwl,)
        Numpy array of the mean emulated spectrum scaled to the opaque transit 
        depth.

    '''
    if model_spectrum.size != EGRETS.Nwl:
        raise ValueError('Input model spectrum must have the same ' + \
                         'wavelength grid as EGRETS.WLgrid.')

    emulated_mu,emulated_sig,_,_ = EGRETS.predict_spectrum(model_params)

    return model_spectrum, emulated_mu, emulated_sig    
