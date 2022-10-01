#!/usr/bin/env python

import numpy as np
import os
from astropy.io import ascii
from astropy import table

from astropy.table import Table
from astropy.io import fits


import SAGA  
from SAGA import ObjectCuts as C

from easyquery import QueryMaker
from easyquery import Query
from SAGA.database import FitsTable 



#  READ AAT SPECTRA
def read_aat_mzfile(mzfile,dir): 
    
        n = mzfile.split('_mg.mz')
        if np.size(n) > 1:
            spec_file = n[0] + '.fits.gz'
        else:
            n = mzfile.split('.mz')
            spec_file = n[0] + '.fits.gz'

        n=mzfile.split('.zlog')
        if np.size(n) > 1:
            spec_file = n[0] + '.fits.gz'
        
        
        hdulist = fits.open(dir+'/Spectra/Final/AAT/'+spec_file)
        
        return hdulist
    
def aat_spectra(obj,saga_dropbox_dir):
    
    hdulist = read_aat_mzfile(obj['MASKNAME'],saga_dropbox_dir)
    slit = int(obj['SPECOBJID'])-1

    # READ DATA
    flux = hdulist[0].data[slit,:]
    ivar = 1./hdulist[1].data[slit,:]
    

    hdr = hdulist[0].header
    pix = np.arange(1,hdulist[0].header['NAXIS1']+1,1) - hdulist[0].header['CRPIX1']
    wave = hdulist[0].header['CRVAL1'] + hdulist[0].header['CDELT1'] * pix   

        
    return flux, wave, ivar


# READ MMT SPECTRA
def mmt_spectra(obj,saga_dropbox_dir):

    mzfile = obj['MASKNAME']
    n=mzfile.split('.zlog')
    if np.size(n) > 1:
        spec_file = n[0] + '.fits.gz'

    hdulist = fits.open(saga_dropbox_dir+'/Spectra/Final/MMT/'+spec_file)
    
    # READ DATA
    slit = int(obj['SPECOBJID'])-1
    wave = hdulist[0].data[slit,:]
    flux = hdulist[1].data[slit,:]
    ivar = hdulist[2].data[slit,:]


    return flux, wave, ivar


# READ PALOMAR SPECTRA
# SATELLITES ONLY RIGHT NOW!!
def pal_spectra(obj,saga_dropbox_dir):
    

    file = saga_dropbox_dir+'Spectra/Final/Palomar/Pal_'+str(obj['SPECOBJID'])+'_'+str(obj['MASKNAME'])+'.fits'
    
    if os.path.isfile(file):
        hdu   = fits.open(file)
        wave  = np.array((hdu[1].data).flat)
        rflux = np.array((hdu[2].data).flat)
        ivar  = np.array((hdu[3].data).flat)
        if obj['SPECOBJID'] != 'geha1141m0626':
            flux   = scipynd.gaussian_filter1d(rflux,3,truncate=3)
        else:
            flux=rflux

    if not os.path.isfile(file):
        return [],[],[]
        
    return flux, wave, ivar

# SALT
def salt_spectra(obj,saga_dropbox_dir):
    
    file = saga_dropbox_dir+'/Spectra/Final/SALT/SALT_'+str(obj['SPECOBJID'])+'_'+str(obj['MASKNAME'])+'.fits'
    
    if os.path.isfile(file):
        hdu = fits.open(file)
        
        try:
            wave = np.array(hdu['wavelength'].data)
            flux = np.array(hdu['intensity'].data)
            ivar = np.array(hdu['ivar'].data)
        except:
            data = hdu[1].data
            wave = data['wave']
            flux = data['flux']
            ivar = data['ivar']
              
    if not os.path.isfile(file):
        print(file)
        return [],[],[]

        
    return flux, wave, ivar



def keck_spectra(obj,saga_dropbox_dir):
    

    file = saga_dropbox_dir+'/Spectra/Final/Keck/KECK_'+ str(obj['SPECOBJID'])+'_'+str(obj['MASKNAME'])+'.fits'
    
    if os.path.isfile(file):
        hdulist = fits.open(file)
        data    = hdulist[1].data
        wave    = np.array(data['WAVE'].flat)
        rflux   = np.array(data['FLUX'].flat)
        flux    = scipynd.gaussian_filter1d(rflux,5,truncate=3)  
        ivar    = np.array(data['IVAR'].flat)


    if not os.path.isfile(file):
        print(file)


    return flux, wave, ivar


def spectra_6dF(obj,saga_dropbox_dir):

    file = saga_dropbox_dir+'/Spectra/Final/6dF/'+str(obj['SPECOBJID'])+'.fits'
    if os.path.isfile(file):
        hdulist = fits.open(file)
        
        data = hdulist[7].data
        wave = np.array(data[3,:].flat)
        flux = np.array(data[0,:].flat)
        ivar = 1./np.array(data[1,:].flat)
    else:
        return [],[],[]

         
    return flux, wave, ivar



# READ COADDED SPECTRA
def coadd_spectra(obj,saga_dropbox_dir):
        

    mzfile = obj['MASKNAME']
    n=mzfile.split('.mz')
    file = n[0] + '.fits'
    spec_file = saga_dropbox_dir+'/Spectra/Final/COADD/'+file
    
    if os.path.isfile(spec_file):
        hdulist = fits.open(spec_file)

        slit = int(obj['SPECOBJID'])-1
        flux = hdulist[0].data[slit,:]
        ivar = 1./hdulist[1].data[slit,:]
        wave = hdulist[2].data[slit,:]
        

    if not os.path.isfile(spec_file):
        print(spec_file)

    
    return flux, wave, ivar




def saga_get_spectrum(obj,saga_dropbox_dir):
    
    wave, flux, ivar = [],[],[]
    
    # Flag is zero if spectrum not found
    flag = 0
    
    if obj['TELNAME'] == 'AAT':
        wave, flux, ivar = aat_spectra(obj,saga_dropbox_dir)

    if obj['TELNAME'] == 'MMT':    
        wave, flux, ivar = mmt_spectra(obj,saga_dropbox_dir)

    if obj['TELNAME'] == 'PAL':    
        wave, flux, ivar = pal_spectra(obj,saga_dropbox_dir)
        
    if obj['TELNAME'] == 'KECK':    
        wave, flux, ivar = keck_spectra(obj,saga_dropbox_dir)
        
    if obj['TELNAME'] == 'SALT':    
        wave, flux, ivar = salt_spectra(obj,saga_dropbox_dir)
  
    if obj['TELNAME'] == '6dF':    
        wave, flux, ivar = spectra_6dF(obj,saga_dropbox_dir)
          
    if obj['TELNAME'] == 'COADD':    
        wave, flux, ivar = coadd_spectra(obj,saga_dropbox_dir)

        
#    if (obj['TELNAME'] == 'NSA') | (obj['TELNAME'] == 'SGA') | (obj['TELNAME'] == 'SDSS'):
#    if obj['TELNAME'] == 'GAMA':
#    if obj['TELNAME'] == 'ALFALF':

    if (np.size(wave) > 1):
        flag=1

    return wave, flux, ivar, flag
    
    



