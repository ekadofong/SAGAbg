import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from astropy.modeling import fitting

from SAGAbg import SAGA_get_spectra, line_fitting

import sys

#from scipy.fftpack import ss_diff
sys.path.append('../scripts/')
import catalogs

# \\ CONSTANTS
line_wavelengths = line_fitting.line_wavelengths

def define_objects ():
    clean = catalogs.build_saga_catalog ()

    first_objects = clean[(clean['selection']==3)&(clean['ZQUALITY']>=3)&((clean['TELNAME']=='AAT')|(clean['TELNAME']=='MMT'))]
    #all_the_good_spectra = clean[(clean['ZQUALITY']>=3)&((clean['TELNAME']=='AAT')|(clean['TELNAME']=='MMT'))]
    return first_objects

def visualize (restwave, flux, line_fluxes, model_fit, model_fit_noabs, frandom, windowwidth, linewidth):
    fig, axarr = plt.subplots(3,1,figsize=(10,6))

    for ax_index,key in enumerate(['Hgamma','Hbeta','Halpha']):
        line_wl=line_wavelengths[key]
        ax = axarr[ax_index]
        xs = np.arange(line_wl - windowwidth/2. + 1, line_wl + windowwidth/2. - 1,.1)
        in_transmission = abs(restwave-line_wl) <= (windowwidth/2.)
        ax.scatter ( restwave[in_transmission], frandom[in_transmission], edgecolor='lightgrey', facecolor='None')
        ax.scatter ( restwave[in_transmission], flux[in_transmission],color='grey')
        xlim = ax.get_xlim()
        ax.plot ( xs, model_fit(xs), color='r', ls='--', lw=2)
        ax.plot ( xs, model_fit_noabs(xs), color='r', lw=2)
        

        for wv in line_wavelengths.values():
            ax.axvline(wv, color='r', ls=':')
            ax.axvspan(wv-linewidth/2.,wv+linewidth/2., color='r',alpha=0.05)
        ax.set_xlim(xlim)  
    
    axarr[0].text ( 0.025, 0.95, r'$\rm F_{H\gamma}$: %.2f (%.2f) [$\pm$%.2f]' % (line_fluxes[0,0], line_fluxes[1,0], line_fluxes[2,0]),
                   transform=axarr[0].transAxes, ha='left', va='top', color='r' )
    axarr[0].text ( 0.025, 0.8, r'$\rm F_{[OIII]\lambda 4363}$: %.2f (%.2f) [$\pm$%.2f]' % (line_fluxes[0,1], line_fluxes[1,1], line_fluxes[2,1]),
                   transform=axarr[0].transAxes, ha='left', va='top', color='r' )   
    axarr[1].text ( 0.025, 0.95, r'$\rm F_{H\beta}$: %.2f (%.2f) [$\pm$%.2f]' % (line_fluxes[0,2], line_fluxes[1,2], line_fluxes[2,2]),
                   transform=axarr[1].transAxes, ha='left', va='top', color='r' )
    axarr[2].text ( 0.025, 0.95, r'$\rm F_{H\alpha}$: %.2f (%.2f) [$\pm$%.2f]' % (line_fluxes[0,3], line_fluxes[1,3], line_fluxes[2,3]),
                   transform=axarr[2].transAxes, ha='left', va='top', color='r' )    
    
    for ax in axarr:
        rectangle = patches.Rectangle ( (0.022, 0.65), 0.35, 0.35, facecolor='w', alpha=0.65, transform=ax.transAxes )     
        ax.add_patch ( rectangle )
      
    
def singleton (obj, dropbox_directory, npull = 100, verbose=True, savefig=True):        
    windowwidth = line_fitting._DEFAULT_WINDOW_WIDTH
    linewidth = line_fitting._DEFAULT_LINE_WIDTH
    
    flux, wave, _, _ = SAGA_get_spectra.saga_get_spectrum(obj, dropbox_directory)
    finite_mask = np.isfinite(flux)
    flux = flux[finite_mask]
    wave = wave[finite_mask]
    restwave = wave / (1. + obj['SPEC_Z'] )
        
    # \\ define spectrum
    outside_windows, outside_lines = line_fitting.define_lineblocs ( restwave )
    this_model = line_fitting.build_linemodel ( restwave, flux )
    this_model_noabs = line_fitting.build_linemodel ( restwave, flux, False )
    fitter = fitting.LevMarLSQFitter ()    
    model_fit = fitter ( this_model, restwave[~outside_windows], flux[~outside_windows] )
    model_fit_noabs = fitter ( this_model_noabs, restwave[~outside_windows], flux[~outside_windows] )
    
    # \\ compute fit fluxes
    halpha_flux = line_fitting.compute_lineflux ( model_fit.amplitude_0, model_fit.stddev_0 )
    oiii_flux = line_fitting.compute_lineflux ( model_fit.amplitude_8, model_fit.stddev_0 )
    hbeta_flux = line_fitting.compute_lineflux ( model_fit.amplitude_6, model_fit.stddev_0 )
    hgamma_flux = line_fitting.compute_lineflux ( model_fit.amplitude_10, model_fit.stddev_0 )
    flux_arr = np.array([hgamma_flux, oiii_flux, hbeta_flux, halpha_flux])    
    
    # \\ same, for no absorption model
    halpha_flux = line_fitting.compute_lineflux ( model_fit_noabs.amplitude_0, model_fit_noabs.stddev_0 )
    oiii_flux = line_fitting.compute_lineflux   ( model_fit_noabs.amplitude_8, model_fit_noabs.stddev_0 )
    hbeta_flux = line_fitting.compute_lineflux  ( model_fit_noabs.amplitude_6, model_fit_noabs.stddev_0 )
    hgamma_flux = line_fitting.compute_lineflux ( model_fit_noabs.amplitude_10,model_fit_noabs.stddev_0 )
    flux_arr_noabs = np.array([hgamma_flux, oiii_flux, hbeta_flux, halpha_flux])     
    
    # \\ let's also estimate the uncertainty in the line fluxes
    halpha_bloc = line_fitting.get_linewindow ( restwave, line_wavelengths['Halpha'], windowwidth )
    hbeta_bloc = line_fitting.get_linewindow ( restwave, line_wavelengths['Hbeta'], windowwidth )
    hgamma_bloc = line_fitting.get_linewindow ( restwave, line_wavelengths['Hgamma'], windowwidth )
    
    u_flux_arr = np.zeros([npull, 4])
    start = time.time ()
    for pull in range(npull):
        # \\ repull from non-line local areas of the spectrum
        frandom = np.zeros_like(restwave)
        frandom[halpha_bloc] = np.random.choice(flux[halpha_bloc&outside_lines], size=halpha_bloc.sum(), replace=True)
        frandom[hbeta_bloc] = np.random.choice(flux[hbeta_bloc&outside_lines], size=hbeta_bloc.sum(), replace=True)
        frandom[hgamma_bloc] = np.random.choice(flux[hgamma_bloc&outside_lines], size=hgamma_bloc.sum(), replace=True)
        
        random_fit = fitter ( this_model_noabs, restwave[~outside_windows], frandom[~outside_windows] )
        u_flux_arr[pull,3] = line_fitting.compute_lineflux ( random_fit.amplitude_0, random_fit.stddev_0 ) # Halpha
        u_flux_arr[pull,1] = line_fitting.compute_lineflux  (  random_fit.amplitude_8, random_fit.stddev_0 ) # OIII
        u_flux_arr[pull,2] = line_fitting.compute_lineflux  ( random_fit.amplitude_6, random_fit.stddev_0 ) # Hbeta
        u_flux_arr[pull,0] = line_fitting.compute_lineflux ( random_fit.amplitude_10, random_fit.stddev_0 ) # Hgamma
    
    line_fluxes = np.array([flux_arr_noabs,flux_arr, u_flux_arr.std(axis=0)])
    elapsed = time.time() - start
    
    if verbose:
        print(f'[u_flux] {elapsed:.0f} sec elapsed; {elapsed/npull:.2f} avg. laptime')    
    
    if isinstance(savefig, str):
        if savefig == 'if_detect':
            if line_fluxes[0,1]/line_fluxes[1,1] > 1.:
                visualize ( restwave, flux, line_fluxes, model_fit, model_fit_noabs, frandom, windowwidth, linewidth )
    elif savefig:
        visualize ( restwave, flux, line_fluxes, model_fit, model_fit_noabs, frandom, windowwidth, linewidth )
    return line_fluxes, model_fit, model_fit_noabs
    

def main (dbdir, savedir, verbose=True, nrun=None):
    clean = catalogs.build_saga_catalog (dropbox_directory=dbdir)
    first_objects = clean[(clean['selection']==3)&(clean['ZQUALITY']>=3)&((clean['TELNAME']=='AAT')|(clean['TELNAME']=='MMT'))]
    
    #all_the_good_spectra = clean[(clean['ZQUALITY']>=3)&((clean['TELNAME']=='AAT')|(clean['TELNAME']=='MMT'))]    
    ncompleted = 0
    nfailed = 0
    with open(f'{savedir}/run.log', 'w') as f:
        for wordid in first_objects['wordid']:
            obj = clean.loc[wordid]
            try:
                line_fluxes, model_fit, model_fit_noabs = singleton ( obj, dropbox_directory=dbdir, verbose=verbose )
            except Exception as e:
                print(f'[{wordid}] {e}', file=f)
                nfailed += 1
                continue
            
            objdir = f'{savedir}/{wordid}/'
            if not os.path.exists ( objdir ):
                os.makedirs ( objdir )
            np.savetxt ( f'{objdir}/{wordid}_fluxes.dat', line_fluxes  )
            np.savetxt ( f'{objdir}/{wordid}_lineparams.dat', model_fit.parameters  )
            np.savetxt ( f'{objdir}/{wordid}_lineparamsNOABSORPTION.dat', model_fit_noabs.parameters  )
            plt.savefig( f'{objdir}/{wordid}.png')
            plt.close()
            if verbose:
                print(f'[main] Saved to {objdir}', file=f)
            ncompleted +=1 
            if nrun is not None:
                if (ncompleted+nfailed) > nrun:
                    break
        print(f'N_completed = {ncompleted}', file=f)
        print(f'N_failed = {nfailed} [{nfailed/(ncompleted+nfailed):.2f}]', file=f)
    