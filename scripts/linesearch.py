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


def visualize (wave, flux, line_fluxes, u_fc, model_fit, model_fit_noabs, frandom, windowwidth, linewidth, z=0.):
    fig, axarr = plt.subplots(3,1,figsize=(10,6))

    for ax_index,key in enumerate(['Hgamma','Hbeta','Halpha']):
        line_wl=line_wavelengths[key] * ( 1. + z)
        ax = axarr[ax_index]
        xs = np.arange(line_wl - windowwidth/2. + 1, line_wl + windowwidth/2. - 1,.1)
        in_transmission = abs(wave-line_wl) <= (windowwidth/2.)
        ax.scatter ( wave[in_transmission], frandom[in_transmission], edgecolor='lightgrey', facecolor='None')
        ax.scatter ( wave[in_transmission], flux[in_transmission],color='grey')
        xlim = ax.get_xlim()
        if model_fit is not None:
            ax.plot ( xs, model_fit(xs), color='r', ls='--', lw=2)
        ax.plot ( xs, model_fit_noabs(xs), color='r', lw=2)
        
        for wv in line_wavelengths.values():
            wv = (1.+z)*wv
            ax.axvline(wv, color='r', ls=':')
            ax.axvspan(wv-linewidth/2.,wv+linewidth/2., color='r',alpha=0.05)
        ax.set_xlim(xlim)  
    
    ew_uncertainty = line_fitting.ew_uncertainty
    if model_fit is not None:
        u_ew_hgamma = ew_uncertainty ( line_fluxes[0,0], model_fit_noabs.amplitude_11.value, 
                                    line_fluxes[2,0], u_fc[0] )
        axarr[0].text ( 0.025, 0.95, r'$\rm EW_{H\gamma}$: %.2f (%.2f) [$\pm$%.2f] $\rm \AA$' % (line_fluxes[0,0]/model_fit_noabs.amplitude_11, 
                                                                                    line_fluxes[1,0]/model_fit_noabs.amplitude_11, 
                                                                                    u_ew_hgamma),
                    transform=axarr[0].transAxes, ha='left', va='top', color='r' )
        
        u_ew_oiii = ew_uncertainty ( line_fluxes[0,1], model_fit_noabs.amplitude_11.value, 
                                    line_fluxes[2,1], u_fc[0] )
        axarr[0].text ( 0.025, 0.8, r'$\rm EW_{[OIII]\lambda 4363}$: %.2f (%.2f) [$\pm$%.2f]  $\rm \AA$' % (line_fluxes[0,1]/model_fit_noabs.amplitude_11, 
                                                                                                line_fluxes[1,1]/model_fit_noabs.amplitude_11, 
                                                                                                u_ew_oiii ),
                    transform=axarr[0].transAxes, ha='left', va='top', color='r' )
        
        u_ew_hbeta = ew_uncertainty ( line_fluxes[0,2], model_fit_noabs.amplitude_7.value, 
                                    line_fluxes[2,2], u_fc[1] )   
        axarr[1].text ( 0.025, 0.95, r'$\rm EW_{H\beta}$: %.2f (%.2f) [$\pm$%.2f] $\rm \AA$' % (line_fluxes[0,2]/model_fit_noabs.amplitude_7, 
                                                                                    line_fluxes[1,2]/model_fit_noabs.amplitude_7, 
                                                                                    u_ew_hbeta),
                    transform=axarr[1].transAxes, ha='left', va='top', color='r' )
        
        u_ew_halpha = ew_uncertainty ( line_fluxes[0,3], model_fit_noabs.amplitude_1.value,
                                    line_fluxes[2,3], u_fc[2] )
        axarr[2].text ( 0.025, 0.95, r'$\rm EW_{H\alpha}$: %.2f (%.2f) [$\pm$%.2f] $\rm \AA$' % (line_fluxes[0,3]/model_fit_noabs.amplitude_1, 
                                                                                    line_fluxes[1,3]/model_fit_noabs.amplitude_1, 
                                                                                    u_ew_halpha),
                    transform=axarr[2].transAxes, ha='left', va='top', color='r' ) 
    else:
        u_ew_hgamma = ew_uncertainty ( line_fluxes[0,0], model_fit_noabs.amplitude_11.value, 
                                    line_fluxes[1,0], u_fc[0] )
        axarr[0].text ( 0.025, 0.95, r'$\rm EW_{H\gamma}$: %.2f [$\pm$%.2f] $\rm \AA$' % (line_fluxes[0,0]/model_fit_noabs.amplitude_11,                                                                                     
                                                                                    u_ew_hgamma),
                    transform=axarr[0].transAxes, ha='left', va='top', color='r' )
        
        u_ew_oiii = ew_uncertainty ( line_fluxes[0,1], model_fit_noabs.amplitude_11.value, 
                                    line_fluxes[1,1], u_fc[0] )
        axarr[0].text ( 0.025, 0.8, r'$\rm EW_{[OIII]\lambda 4363}$: %.2f [$\pm$%.2f]  $\rm \AA$' % (line_fluxes[0,1]/model_fit_noabs.amplitude_11,                                                                                                 
                                                                                                u_ew_oiii ),
                    transform=axarr[0].transAxes, ha='left', va='top', color='r' )
        
        u_ew_hbeta = ew_uncertainty ( line_fluxes[0,2], model_fit_noabs.amplitude_7.value, 
                                    line_fluxes[1,2], u_fc[1] )   
        axarr[1].text ( 0.025, 0.95, r'$\rm EW_{H\beta}$: %.2f [$\pm$%.2f] $\rm \AA$' % (line_fluxes[0,2]/model_fit_noabs.amplitude_7,                                                                                     
                                                                                    u_ew_hbeta),
                    transform=axarr[1].transAxes, ha='left', va='top', color='r' )
        
        u_ew_halpha = ew_uncertainty ( line_fluxes[0,3], model_fit_noabs.amplitude_1.value,
                                    line_fluxes[1,3], u_fc[2] )
        axarr[2].text ( 0.025, 0.95, r'$\rm EW_{H\alpha}$: %.2f [$\pm$%.2f] $\rm \AA$' % (line_fluxes[0,3]/model_fit_noabs.amplitude_1,                                                                                     
                                                                                    u_ew_halpha),
                    transform=axarr[2].transAxes, ha='left', va='top', color='r' )               
    
    for ax in axarr:
        rectangle = patches.Rectangle ( (0.022, 0.65), 0.35, 0.35, facecolor='w', alpha=0.65, transform=ax.transAxes )     
        ax.add_patch ( rectangle )
  
def load_spectrum ( obj, dropbox_directory, restframe=False ):
    flux, wave, _, _ = SAGA_get_spectra.saga_get_spectrum(obj, dropbox_directory)
    finite_mask = np.isfinite(flux)
    flux = flux[finite_mask]
    wave = wave[finite_mask]
    if not restframe:
        return wave, flux
    restwave = wave / (1. + obj['SPEC_Z'] )
    restflux = flux * (1. + obj['SPEC_Z'])
    return restwave, restflux


def singleton (obj, dropbox_directory, **kwargs ):        
    restwave, restflux = load_spectrum ( obj, dropbox_directory, restframe=True )
    output = fit ( restwave, restflux, **kwargs )
    return output

def fit ( wave, flux, z=0., npull = 100, verbose=True, fit_with_absorption=False, savefig='if_detect' ):
    windowwidth = line_fitting._DEFAULT_WINDOW_WIDTH*(1.+z)
    linewidth = line_fitting._DEFAULT_LINE_WIDTH*(1.+z)
        
    # \\ define spectrum
    outside_windows, outside_lines = line_fitting.define_lineblocs ( wave, z=z )
    this_model = line_fitting.build_linemodel ( wave, flux, z=z )
    this_model_noabs = line_fitting.build_linemodel ( wave, flux, include_absorption=False , z=z)
    fitter = fitting.LevMarLSQFitter ()    

    if fit_with_absorption:
        model_fit = fitter ( this_model, wave[~outside_windows], flux[~outside_windows], )
        
    
        # \\ compute fit fluxes
        halpha_flux = line_fitting.compute_lineflux ( model_fit.amplitude_0, model_fit.stddev_0 )
        oiii_flux = line_fitting.compute_lineflux ( model_fit.amplitude_8, model_fit.stddev_0 )
        hbeta_flux = line_fitting.compute_lineflux ( model_fit.amplitude_6, model_fit.stddev_0 )
        hgamma_flux = line_fitting.compute_lineflux ( model_fit.amplitude_10, model_fit.stddev_0 )
        flux_arr = np.array([hgamma_flux, oiii_flux, hbeta_flux, halpha_flux])
    else:
        model_fit = None    
    
    model_fit_noabs = fitter ( this_model_noabs, wave[~outside_windows], flux[~outside_windows] )
    # \\ same, for no absorption model
    halpha_flux = line_fitting.compute_lineflux ( model_fit_noabs.amplitude_0, model_fit_noabs.stddev_0 )
    oiii_flux = line_fitting.compute_lineflux   ( model_fit_noabs.amplitude_8, model_fit_noabs.stddev_0 )
    hbeta_flux = line_fitting.compute_lineflux  ( model_fit_noabs.amplitude_6, model_fit_noabs.stddev_0 )
    hgamma_flux = line_fitting.compute_lineflux ( model_fit_noabs.amplitude_10,model_fit_noabs.stddev_0 )
    flux_arr_noabs = np.array([hgamma_flux, oiii_flux, hbeta_flux, halpha_flux])     
    
    # \\ let's also estimate the uncertainty in the line fluxes
    halpha_bloc = line_fitting.get_linewindow ( wave, line_wavelengths['Halpha']*(1.+z), windowwidth )
    hbeta_bloc = line_fitting.get_linewindow ( wave, line_wavelengths['Hbeta']*(1.+z), windowwidth )
    hgamma_bloc = line_fitting.get_linewindow ( wave, line_wavelengths['Hgamma']*(1.+z), windowwidth )
    
    u_flux_arr = np.zeros([npull, 4])
    u_fc_arr = np.zeros([npull,3])
    
    start = time.time ()
    for pull in range(npull):
        # \\ repull from non-line local areas of the spectrum
        frandom = np.zeros_like(wave)
        frandom[halpha_bloc] = np.random.choice(flux[halpha_bloc&outside_lines], size=halpha_bloc.sum(), replace=True)
        frandom[hbeta_bloc] = np.random.choice(flux[hbeta_bloc&outside_lines], size=hbeta_bloc.sum(), replace=True)
        frandom[hgamma_bloc] = np.random.choice(flux[hgamma_bloc&outside_lines], size=hgamma_bloc.sum(), replace=True)
        
        random_fit = fitter ( this_model_noabs, wave[~outside_windows], frandom[~outside_windows] )
        u_flux_arr[pull,3] = line_fitting.compute_lineflux ( random_fit.amplitude_0, random_fit.stddev_0 ) # Halpha
        u_flux_arr[pull,1] = line_fitting.compute_lineflux  (  random_fit.amplitude_8, random_fit.stddev_0 ) # OIII
        u_flux_arr[pull,2] = line_fitting.compute_lineflux  ( random_fit.amplitude_6, random_fit.stddev_0 ) # Hbeta
        u_flux_arr[pull,0] = line_fitting.compute_lineflux ( random_fit.amplitude_10, random_fit.stddev_0 ) # Hgamma
        
        # \\ also track continuum uncertainty
        u_fc_arr[pull,2] =  random_fit.amplitude_1.value # Halpha
        u_fc_arr[pull,1] =  random_fit.amplitude_7.value # Hbeta
        u_fc_arr[pull,0] = random_fit.amplitude_11.value # Hgamma
    
    if fit_with_absorption:
        line_fluxes = np.array([flux_arr_noabs,flux_arr, u_flux_arr.std(axis=0)])
    else:
        line_fluxes = np.array([flux_arr_noabs, u_flux_arr.std(axis=0)])
    u_fc = u_fc_arr.std(axis=0)
    elapsed = time.time() - start
    
    if verbose:
        print(f'[u_flux] {elapsed:.0f} sec elapsed; {elapsed/npull:.2f} avg. laptime')    
    
    if isinstance(savefig, str):
        if savefig == 'if_detect':
            random_trip = np.random.uniform ( 0., 1. ) > .9
            if (line_fluxes[0,1]/line_fluxes[2,1] > 1.) or random_trip:
                visualize ( wave, flux, line_fluxes,u_fc, model_fit, model_fit_noabs, frandom, windowwidth, linewidth, z=z )
    elif savefig:
        visualize ( wave, flux, line_fluxes, u_fc, model_fit, model_fit_noabs, frandom, windowwidth, linewidth, z=z )
    return line_fluxes, u_fc, model_fit, model_fit_noabs
    

def main (dbdir, savedir, verbose=True, nrun=None, clobber=False):
    clean = catalogs.build_saga_catalog (dropbox_directory=dbdir)
    #first_objects = clean[(clean['selection']==3)&(clean['ZQUALITY']>=3)&((clean['TELNAME']=='AAT')|(clean['TELNAME']=='MMT'))]    
    all_the_good_spectra = clean[(clean['ZQUALITY']>=3)&((clean['TELNAME']=='AAT')|(clean['TELNAME']=='MMT'))]    
    low_mass = all_the_good_spectra[(all_the_good_spectra['cm_logmstar']<9.)&(all_the_good_spectra['SPEC_Z']<0.2)]
    
    ncompleted = 0
    nfailed = 0
    with open(f'{savedir}/run.log', 'a') as f:
        for wordid in low_mass['wordid']:
            obj = clean.loc[wordid]
            
            # \\ check for rerun
            objdir = f'{savedir}/{wordid}/'
            if clobber:
                pass
            elif os.path.exists(f'{objdir}/{wordid}_fluxes.dat'):
                if verbose:
                    print(f'[main] {wordid} already run. Skipping...', file=f)
                continue

            
            try:
                line_fluxes, u_fc, model_fit, model_fit_noabs = singleton ( obj, dropbox_directory=dbdir, verbose=verbose )
            except Exception as e:
                print(f'[{wordid}] {e}', file=f)
                nfailed += 1
                continue
            
            
            if not os.path.exists ( objdir ):
                os.makedirs ( objdir )
            np.savetxt ( f'{objdir}/{wordid}_fluxes.dat', line_fluxes  )
            np.savetxt ( f'{objdir}/{wordid}_ucontinuum.dat', u_fc)
            np.savetxt ( f'{objdir}/{wordid}_lineparams.dat', model_fit.parameters  )
            np.savetxt ( f'{objdir}/{wordid}_lineparamsNOABSORPTION.dat', model_fit_noabs.parameters  )
            if len(plt.get_fignums()) > 0: # \\ If singleton produced a figure
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

if __name__ == '__main__':
    dbdir = sys.argv[1]
    savedir = sys.argv[2]
    main ( dbdir, savedir )