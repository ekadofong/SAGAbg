import requests
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
    
def download_gamaspec ( urlname, specid ):
    filename = f'../../gama/spectra/{specid}.fits'
    with open(filename,'wb') as f:
        response = requests.get( urlname )
        f.write( response.content )   
    return filename
 

def load_gamaspec ( gamaspec ):
    ''' 
    Convert GAMA FITS file to a spectrum.
    Flux in 1e-17 erg/s/cm^2/AA
    '''
    hdr = gamaspec[0].header
    if len(gamaspec) == 1:
        flux = gamaspec[0].data[0]
        var = gamaspec[0].data[1]
    else:
        flux = gamaspec['PRIMARY'].data
        var = gamaspec['VARIANCE'].data
        
    #calibrated = gamaspec[0].data[0]
    # \\ some of the GAMA spectra are log10(wv), some are linear 
    if 'log10' in hdr.comments['CRVAL1']:
        wave = 10.**((np.arange(hdr['NAXIS1'])-hdr['CRPIX1']) * hdr['CD1_1'] + hdr['CRVAL1'])
    else:
        wave = (np.arange(hdr['NAXIS1'])-hdr['CRPIX1']) * hdr['CD1_1'] + hdr['CRVAL1']

    finite_mask = np.isfinite(flux)
    flux = flux[finite_mask]
    wave = wave[finite_mask] 
    var = var[finite_mask]
    return wave, flux, var  