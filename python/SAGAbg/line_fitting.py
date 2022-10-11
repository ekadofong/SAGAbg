from turtle import window_width
import numpy as np
from astropy.modeling import models

line_wavelengths = {'Halpha':6563.,'OIII4363':4363., 'Hbeta':4862., 'Hgamma':4341., 'NII6548':6548., 'NII6583':6583.  }
_DEFAULT_WINDOW_WIDTH = 150.
_DEFAULT_LINE_WIDTH = 14.

def get_linewindow ( restwave, line_wl, width=None):
    if width is None:
        width = _DEFAULT_LINE_WIDTH
    in_transmission = abs(restwave-line_wl) <= (width/2.)
    return in_transmission

def define_linemodel ( restwave, flux, line_wl, ltype='emission', linewidth=None, windowwidth=None, line_wiggle=1.,
                       stddev_init=3. ):
    if linewidth is None:
        linewidth = _DEFAULT_LINE_WIDTH         
    if windowwidth is None:
        windowwidth = _DEFAULT_WINDOW_WIDTH
        
    cmask = get_linewindow ( restwave, line_wl, windowwidth )
    continuum_init = np.median(flux[cmask])
    if ltype=='emission':
        lmask = get_linewindow ( restwave, line_wl, linewidth )
        amplitude_init = flux[lmask].max()
        model_continuum = models.Box1D ( amplitude = continuum_init, x_0=line_wl, width=windowwidth )
    elif ltype=='absorption':
        amplitude_init = -1. * continuum_init * 0.25
        
    model_line = models.Gaussian1D ( amplitude = amplitude_init,
                                   mean = line_wl, 
                                   stddev = stddev_init )
    if ltype=='emission':
        lmodel = model_line + model_continuum
        lmodel.mean_0.bounds = (line_wl-line_wiggle, line_wl+line_wiggle)
    elif ltype=='absorption':
        lmodel = model_line
   
    if ltype == 'emission':
        lmodel.amplitude_0.bounds = (0., np.infty)
        lmodel.stddev_0.bounds = (0., 4.)
    elif ltype == 'absorption':
        lmodel.amplitude.bounds = (-continuum_init*1.5, 0.)
        lmodel.stddev.bounds = (0., 10.)
        
        
    return lmodel

def tie_vdisp ( this_model ):
    return this_model.stddev_0

def tie_vdisp12 ( this_model ):
    return this_model.stddev_12

def tie_mean0 ( this_model ):
    return this_model.mean_0 

def tie_mean6 ( this_model ):
    return this_model.mean_6

def tie_mean10 ( this_model ):
    return this_model.mean_10

def define_lineblocs ( restwave,  windowwidth=None, linewidth=None ):
    if windowwidth is None:
        windowwidth = _DEFAULT_WINDOW_WIDTH
    if linewidth is None:
        linewidth = _DEFAULT_LINE_WIDTH
    outside_windows = abs(restwave - line_wavelengths['Halpha']) > windowwidth/2.
    outside_windows &= abs(restwave - line_wavelengths['Hbeta']) > windowwidth/2.
    outside_windows &= abs(restwave - line_wavelengths['Hgamma']) > windowwidth/2.

    outside_lines = abs(restwave - line_wavelengths['Halpha']) > linewidth/2.
    outside_lines &= abs(restwave - line_wavelengths['NII6548']) > linewidth/2.  
    outside_lines &= abs(restwave - line_wavelengths['NII6583']) > linewidth/2.   
    outside_lines &= abs(restwave - line_wavelengths['Hbeta']) > linewidth/2. 
    outside_lines &= abs(restwave - line_wavelengths['OIII4363']) > linewidth/2. 
    outside_lines &= abs(restwave - line_wavelengths['Hgamma']) > linewidth/2. 
    return outside_windows, outside_lines

def build_linemodel ( restwave, flux, include_absorption=True):
    '''
    A line model which fits continuum + lines for Halpha, Hbeta, Hgamma and surrounding lines (especially OIII4363).
    The local continuum for each bloc is fit by a constant (Box1D); the linewidths are all tied to be equal.
    '''
    model_ha = define_linemodel ( restwave, flux, line_wavelengths['Halpha'] )
    model_niib = define_linemodel ( restwave, flux, line_wavelengths['NII6548'] )
    model_niir = define_linemodel ( restwave, flux, line_wavelengths['NII6583'] )
    model_hb = define_linemodel ( restwave, flux, line_wavelengths['Hbeta'] )
    model_oiii4363 = define_linemodel ( restwave, flux, line_wavelengths['OIII4363'] )
    model_hgamma = define_linemodel ( restwave, flux, line_wavelengths['Hgamma'] )
    
    if include_absorption:
        model_ha_abs = define_linemodel ( restwave, flux, line_wavelengths['Halpha'], 'absorption')
        model_hb_abs = define_linemodel ( restwave, flux, line_wavelengths['Hbeta'], 'absorption')
        model_hg_abs = define_linemodel ( restwave, flux, line_wavelengths['Hgamma'], 'absorption')
        
    # \\ Line dictionaries
    # 0 : Halpha
    # 2 : NII6548
    # 4 : NII6583
    # 6 : Hbeta
    # 8 : OIII4363
    # 10: Hgamma
    # 12 : Halpha (absorption)
    # 13 : Hbeta (absorption)
    # 14 : Hgamma (absorption)
    this_model = model_ha + model_niib + model_niir + model_hb + model_oiii4363 + model_hgamma
    if include_absorption:
        this_model = this_model + model_ha_abs + model_hb_abs  + model_hg_abs
        this_model.stddev_13.tied = tie_vdisp12
        this_model.stddev_14.tied = tie_vdisp12
        
        this_model.mean_12.tied = tie_mean0
        this_model.mean_13.tied = tie_mean6
        this_model.mean_14.tied = tie_mean10

    # \\ all linewidths are constrained to be the same
    this_model.stddev_2.tied = tie_vdisp # tie linewidths together
    this_model.stddev_4.tied = tie_vdisp # tie linewidths together
    this_model.stddev_6.tied = tie_vdisp # tie linewidths together
    this_model.stddev_8.tied = tie_vdisp # tie linewidths together
    this_model.stddev_10.tied = tie_vdisp # tie linewidths together

    # \\ NII continuum should be the same as Halpha
    this_model.amplitude_3 = 0.
    this_model.amplitude_3.fixed = True
    this_model.amplitude_5 = 0.
    this_model.amplitude_5.fixed = True

    # \\ OIII continuum should be the same as Hgamma
    this_model.amplitude_9 = 0.
    this_model.amplitude_9.fixed = True        
    
    return this_model

def compute_lineflux (amplitude, stddev):
    '''
    Integrate over astropy's Gaussian from -infty->infty
    '''
    line_flux = amplitude * np.sqrt(2.*np.pi) * stddev
    return line_flux
