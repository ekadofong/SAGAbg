import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import specutils
from specutils import analysis, spectra, manipulation
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty
from SAGAbg import SAGA_get_spectra

# \\ mock NB parameters
line_wavelengths = {'Halpha':6563.,'OIII4363':4363., 'Hbeta':4861., }

keys = ['OIII4363','Hbeta','Halpha']
labels = [r'[OIII]$\lambda$4363$\rm \AA$', r'H$\beta$', r'H$\alpha$']
def fit_line ( subspec, line_wl, fit_continuum=False ):
    # Fit the spectrum and calculate the fitted flux values (``y_fit``)
    g_init = models.Gaussian1D(amplitude=subspec.flux.max(), mean=line_wl, stddev=1.*u.AA)
    
    c_init = models.Const1D ( amplitude = np.nanmedian(subspec.flux) )
    model_init = g_init + c_init
    if not fit_continuum:
        model_init.amplitude_1.fixed = True
        
    g_fit = fit_lines(subspec, model_init)
    #y_fit = g_fit(subspec.spectral_axis)

    line_flux = g_fit.amplitude_0 * np.sqrt(2.*np.pi) * g_fit.stddev_0
    continuum_specflux = g_fit.amplitude_1  

    return g_fit, (line_flux.value, continuum_specflux.value)  

def define_subregion (spec, line_wl, subspec_window = 70.*u.AA  ):
    region = spectra.SpectralRegion( line_wl - subspec_window, line_wl + subspec_window )
    subspec = manipulation.extract_region ( spec, region )      
    return subspec

def singleton (obj, dropbox_directory = '/Users/kadofong/DropBox/SAGA/'):
    flux, wave, ivar, _ = SAGA_get_spectra.saga_get_spectrum(obj, dropbox_directory)
    isfinite = np.isfinite(flux)
    flux = flux[isfinite]
    wave = wave[isfinite]
    
    uncertainty = StdDevUncertainty ( np.sqrt(flux) )
    spec = specutils.Spectrum1D ( flux=flux*u.count, spectral_axis=wave/(1.+obj['SPEC_Z'])*u.AA, 
                                uncertainty=uncertainty )    
    
    # \\ set up table
    arr = np.zeros([len(keys),7])
    
    # \\ set up figure
    fig, axarr = plt.subplots(len(keys),1,figsize=(10,3*len(keys)))
    
    for key_index, key in enumerate(keys):
        subspec = define_subregion ( spec, line_wavelengths[key]*u.AA )
        if subspec.flux.size == 0:
            arr[key_index] = np.NaN
            continue

        lfit, fluxes = fit_line ( subspec, line_wavelengths[key]*u.AA )
        arr[key_index,:2] = fluxes
        arr[key_index,2] = lfit.amplitude_0.value
        arr[key_index,3] = lfit.mean_0.value
        arr[key_index,4] = lfit.stddev_0.value
        arr[key_index,5] = lfit.amplitude_1.value
        arr[key_index,6] = np.subtract(*np.quantile(subspec.flux.value, [0.84,.16]))
        
        ax = axarr[key_index]
        xs = subspec.spectral_axis
        ax.plot ( xs, subspec.flux, color='k' )
        ax.axhline ( lfit.amplitude_1.value, color='tab:blue', lw=3)     
        ax.axhline ( np.nanquantile(subspec.flux.value, .5), color='grey')
        ax.axhspan ( *np.nanquantile(subspec.flux.value, [0.84,.16]), color='#e0e0e0',zorder=0 )
        ax.plot ( xs, lfit(xs), color='r' )
        ax.set_ylabel(r'$\rm F_\lambda$ (counts)')
        
        ax.set_xlabel(r'Wavelength ($\rm \AA$)')
        ax.text ( 0.025, 0.975, labels[key_index] + r' [EW=%.1f $\rm \AA$]' % (fluxes[0]/fluxes[1]), 
                 transform=ax.transAxes, fontsize=13, ha='left', va='top')
    axarr[2].text ( 0.975, 0.975, r'''z = %.3f
M$_{\rm r}$ = %.1f
$\log_{10}(M_\bigstar/M_\odot)$ = %.1f ''' % (obj["SPEC_Z"], obj['Mr'], obj['cm_logmstar']), 
                 transform=axarr[2].transAxes, fontsize=13, ha='right', va='top'
)
    plt.tight_layout ()
    plt.savefig(f'../figures/exploration/line_fits/fit_{obj["wordid"]}.png')  
    return arr