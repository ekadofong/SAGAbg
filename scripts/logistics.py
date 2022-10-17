import numpy as np
from SAGAbg import line_fitting, SAGA_get_spectra

AAT_BREAK = 5790. # lambda @ blue -> red arm break in AAT

def do_fluxcalibrate (obj, tdict, dropbox_dir):
    flux, wave, _, _ = SAGA_get_spectra.saga_get_spectrum(obj, dropbox_dir)
    if len(flux) == 0:
        return None, None
    finite_mask = np.isfinite(flux)
    flux = flux[finite_mask]
    wave = wave[finite_mask]

    _, qfactors = line_fitting.flux_calibrate( wave, flux, obj, tdict )
    
    if obj['TELNAME'] == 'AAT':
        fluxcal = np.where ( wave < AAT_BREAK, flux*qfactors[0], flux*qfactors[1])*1e17
    else:
        fluxcal = flux * np.nanmean(qfactors)*1e17    
    return wave, fluxcal    

def load_filters ( filterset='DECam' ):
    if filterset=='DECam':
        tdict = {}
        for fname in 'grz':
            transmission = np.genfromtxt(f'../local_data/filter_curves/decam/CTIO_DECam.{fname}.dat')
            tdict[fname] = transmission
        return tdict
    elif filterset == 'SDSS':
        sloan_filters = {}
        for fname in 'gr':
            transmission = np.genfromtxt(f'../local_data/filter_curves/sloan/SLOAN_SDSS.{fname}.dat')
            sloan_filters[fname] = transmission  
        return sloan_filters          
