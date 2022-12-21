import requests
import numpy as np
import pandas as pd
from ekfstats import sampling
from SAGAbg import line_fitting, SAGA_get_spectra, models, temdenext

ge_filename = '../local_data/galactic_extinction/output/extinction_formatted.csv'
ge = pd.read_csv(ge_filename, index_col=0)

gama_ge_filename = '../local_data/galactic_extinction/output/GSB_extinction_formatted.csv'
gama_ge = pd.read_csv(gama_ge_filename, index_col=0)

AAT_BREAK = 5790. # lambda @ blue -> red arm break in AAT

#def get_gamaspec ( specid, specdir='../../gama/gsb/spectra' ):
#    '''
#    Load [assuming flux-calibrated] AAT & SDSS spectra from the GAMA survey
#    '''
#    from astropy.io import fits
#
#    #specid = row['SPECID'].strip()
#    if specid[0] == 'G':
#        prefix = specid.split('_')[0]
#    else:
#        prefix = 'etc'
#    specfile = f'{specdir}/{prefix}/{specid}.fit'    
#    
#    hdulist = fits.open(specfile)
#
#    hdr = hdulist[0].header
#    pix = np.arange(1,hdulist[0].header['NAXIS1']+1) - hdulist[0].header['CRPIX1']
#    wave = hdulist[0].header['CRVAL1'] + hdulist[0].header['CD1_1'] * pix 
#
#    flux = hdulist[0].data[0]
#    uflux = hdulist[0].data[1]    
#    
#    return wave, flux, uflux

def do_fluxcalibrate (obj, tdict, dropbox_dir, cut_red=True, alpha=0.05, apply_GEcorrection=True ):
    '''
    Calibration quality:
    0 : not well-calibrated
    1 : well-calibrated (AAT)
    2 : MMT (no break)
    '''
    flux, wave, ivar, _ = SAGA_get_spectra.saga_get_spectrum(obj, dropbox_dir)
    if len(flux) == 0:
        return None, None
    finite_mask = np.isfinite(flux)
    flux = flux[finite_mask].astype(float)
    wave = wave[finite_mask].astype(float)
    #ivar = ivar[finite_mask].astype(float)

    _, qfactors = line_fitting.flux_calibrate( wave, flux, obj, tdict )
    
    if obj['TELNAME'] == 'AAT':
        fluxcal = np.where ( wave < AAT_BREAK, flux*qfactors[0], flux*qfactors[1])*1e17
        #ivarcal = np.where ( wave < AAT_BREAK, ivar*qfactors[0]**2, ivar*qfactors[1]**2)*1e17
    else:
        fluxcal = flux * np.nanmean(qfactors)*1e17    
        #ivarcal = ivar * (np.nanmean(qfactors)*1e17)**2
    if cut_red:
        # \\ if AAT, flux calibration is not reliable past 8000 AA
        if obj['TELNAME'] == 'AAT':
            wvmask = wave < 8000.
            wave = wave[wvmask]
            fluxcal = fluxcal[wvmask]
            #ivarcal = ivarcal[wvmask]
        elif obj['TELNAME'] == 'MMT':
            wvmask = wave < 8200.
            wave = wave[wvmask]
            fluxcal = fluxcal[wvmask]   
            #ivarcal = ivarcal[wvmask]     
        else:
            raise ValueError ('spectra from %s not flux calibrated' % obj['TELNAME'])
        
    if obj['TELNAME'] == 'AAT':
        break_quantile = check_fluxcalibration ( wave, fluxcal, )        
        if (break_quantile < alpha) or (break_quantile > (1.-alpha)):
            qcalibration = 0
        else:
            qcalibration = 1
        #print(f'{break_quantile} ({qcalibration})')
    elif obj['TELNAME'] == 'MMT':
        qcalibration = 2
        
    # \\ apply galactic extinction correction
    if apply_GEcorrection:
        #idx = obj['number']
        #ge = pd.read_csv('../local_data/galactic_extinction/output/extinction_formatted.csv', skiprows = lambda x: 0<x<idx+1, 
        #                 index_col=0, nrows=1)
        Av = ge.loc[obj.name, 'AV_SandF'] 
        gecorr = temdenext.gecorrection (wave, Av,)
        fluxcal *= gecorr
        #ivarcal *= gecorr**2
        
    return wave, fluxcal,  qcalibration  

def check_fluxcalibration ( wave, flux, window=500, break_window=30, kernel_kwargs=None ):
    if kernel_kwargs is None:
        kernel_kwargs = {}
    kernel = spectrum_kernel ( **kernel_kwargs )
    
    blu = ((AAT_BREAK - window) < wave)&(wave < (AAT_BREAK - break_window ))
    red = ((AAT_BREAK + window) > wave)&(wave > (AAT_BREAK + break_window ))
    
    blu_dev = flux[blu] - np.convolve(flux, kernel, mode='same')[blu]
    red_dev = flux[red] - np.convolve(flux, kernel, mode='same')[red]
    dev = np.concatenate([blu_dev, red_dev])
    break_val = np.median(flux[blu]) - np.median(flux[red])
    quantile = sampling.get_quantile_of_value ( dev, break_val )
    return quantile

def ge_of_photometry ( objname, filter_name, filter_dir='../local_data/filter_curves/decam/'):
    transmission = np.loadtxt(f'{filter_dir}/CTIO_DECam.{filter_name}.dat')
    filter_pdf = transmission[:,1]/np.trapz(transmission[:,1],transmission[:,0])
    Alambda = temdenext.gecorrection ( transmission[:,0], ge.loc[objname, 'AV_SandF'], return_magcorr=True )
    ge_ev = np.trapz(filter_pdf*Alambda, transmission[:,0] ) # expected value of galactic extinction correction
    return ge_ev
    

def spectrum_kernel ( size=10, type='gaussian' ):
    xs = np.arange(-50,51)
    if type == 'gaussian':
        kernel = models.gaussian ( xs, 'normalize', 0., size )
    elif type == 'tophat':
        kernel = np.ones(size)
        kernel /= kernel.sum()
    return kernel

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
 
def load_gamaspec ( specid, specdir='../../gama/gsb/spectra', apply_GEcorrection=True ):
    ''' 
    Convert GAMA FITS file to a spectrum.
    Flux in 1e-17 erg/s/cm^2/AA
    '''
    from astropy.io import fits

    if specid[0] == 'G':
        prefix = specid.split('_')[0]
    else:
        prefix = 'etc'
    specfile = f'{specdir}/{prefix}/{specid}.fit'    
    
    gamaspec = fits.open(specfile)    
    
    hdr = gamaspec[0].header
    if len(gamaspec) == 1:
        flux = gamaspec[0].data[0]
        var = gamaspec[0].data[1]**2
    else:
        flux = gamaspec['PRIMARY'].data
        var = gamaspec['VARIANCE'].data
        
    #calibrated = gamaspec[0].data[0]
    # \\ some of the GAMA spectra are log10(wv), some are linear 
    if 'log10' in hdr.comments['CRVAL1']:
        wave = 10.**((1,1+np.arange(hdr['NAXIS1'])-hdr['CRPIX1']) * hdr['CD1_1'] + hdr['CRVAL1'])
    else:
        wave = (np.arange(1,1+hdr['NAXIS1'])-hdr['CRPIX1']) * hdr['CD1_1'] + hdr['CRVAL1']

    finite_mask = np.isfinite(flux)
    flux = flux[finite_mask]
    wave = wave[finite_mask] 
    var = var[finite_mask]
    
    # \\ apply galactic extinction correction
    if apply_GEcorrection:
        Av = gama_ge.loc[specid, 'AV_SandF'] 
        gecorr = temdenext.gecorrection (wave, Av,)
        flux *= gecorr
        var *= gecorr**2
            
    return wave, flux, var  

