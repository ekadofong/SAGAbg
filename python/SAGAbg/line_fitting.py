#import time
#from webbrowser import get
from operator import add
import numpy as np
import pandas as pd
from astropy import constants as co
from astropy import units as u
from astropy import modeling
#import pyneb as pn 
from ekfparse import strings
from .models import *
from .line_db import *
#import models, fitting

models = modeling.models
fitting = modeling.fitting

c_angpers = co.c.to(u.AA/u.s).value
gAB_nu = (3631.*u.Jy).to(u.erg/u.s/u.cm**2/u.Hz).value

def establish_tie ( this_model, tied_attribute ):
    def tie ( this_model ):
        return getattr ( this_model, tied_attribute )
    return tie



def build_ovlmodel ( wave, flux, z=None, line_width=None, window_width=None, add_absorption=True, wv_cutoff=8000. ):
    '''
    A more generalized line model that includes all of lines described at top.
    Each line gets its own continuum, so we'll see how the fits go.
    '''
    is_finite = np.isfinite(flux)
    wave = wave[is_finite]
    flux = flux[is_finite]
    if z is None:
        z=0.

    # \\ initialize Balmer lines based of Halpha
    balmer_levels = np.array([1., 2.86, 6.11, 11.05 ])
    in_halpha, in_hwindow = get_lineblocs ( wave, z=z, lines=line_wavelengths['Halpha'] )
    cinit = np.median(flux[~in_halpha&in_hwindow])
    halpha_init = (np.max(flux[in_halpha]) - cinit)
    balmer_init = halpha_init / balmer_levels
    balmer_init = dict(zip(BALMER_ABSORPTION, balmer_init))
    
    model_list = []
    parameter_indices = []
    for linename in line_wavelengths.keys():
        if (line_wavelengths[linename]*(1.+z)) > wv_cutoff:
            continue
        if linename not in CONTINUUM_TAGS:
            add_continuum = False
        else:
            add_continuum = True
            
        if linename in BALMER_ABSORPTION:
            ainit = balmer_init[linename]            
        else:
            ainit = None 
        
        component = define_linemodel ( wave, flux, line_wavelengths[linename], z=z, linewidth=line_width,
                                             windowwidth=window_width, add_continuum=add_continuum,
                                             amplitude_init=ainit)
        parameter_indices.append(f'emission{linename}')
        if add_continuum:
            parameter_indices.append(f'continuum{linename}')
            
        # \\ add Balmer emission where
        # \\ the EW is held constant
        if (linename in BALMER_ABSORPTION) and add_absorption:
            absorption = define_linemodel ( wave, flux, line_wavelengths[linename], z=z, stddev_init=6., ltype='absorption')
            parameter_indices.append(f'absorption{linename}')           
            component = component + absorption 
        model_list.append(component)
    model_init = model_list[0]
    for ix in range(1, len(model_list)):
        model_init = model_init + model_list[ix]
     
    #####   
    # \\ START tying parameters together
    #####  
    # \\ all emission linewidths are constrained to be the same
    emission_indices = strings.where_substring ( parameter_indices, 'emission' )
    emission_stddevs = [ 'stddev_%i'%i for i in emission_indices ]
    tie_emstd = establish_tie ( model_init, emission_stddevs[0] )
    for sname in emission_stddevs[1:]:
        getattr(model_init, sname).tied = tie_emstd
    
    if add_absorption:
        # \\ all absorption linewidths are constrained to be the same
        absorption_indices = strings.where_substring ( parameter_indices, 'absorption' )
        absorption_stddevs = [ 'stddev_%i'%i for i in absorption_indices ]
        tie_abstd = establish_tie ( model_init, absorption_stddevs[0] )
        for sname in absorption_stddevs[1:]:
            getattr(model_init, sname).tied = tie_abstd
        
        # \\ all absorption equivalent widths are constrained to be the same
        absorption_ews = [ 'EW_%i'%i for i in absorption_indices ]
        tie_abew = establish_tie ( model_init, absorption_ews[0] )
        for sname in absorption_ews[1:]:
            getattr(model_init, sname).tied = tie_abew
            
        # \\ the absorption continuum values should be drawn from the
        # \\ fitted continuum models
        # \\ assume that the continuum model directly precedes
        # \\ the absorption model
        for index in absorption_indices:
            tie2continuum = establish_tie ( model_init, 'amplitude_%i' % (index-1) )
            fc = 'fc_%i' % index
            getattr(model_init, fc).tied = tie2continuum
            
        # \\ all absorption line positions should be tied to their emission counterparts
        for index in absorption_indices:
            tie2emission = establish_tie ( model_init, 'mean_%i' % (index-2) )
            mc = 'mean_%i' % index
            getattr(model_init, mc).tied = tie2emission       
        
    return model_init, np.asarray(parameter_indices)

def fit ( wave, flux, z=0., npull = 100, add_absorption=True ):
    window_width = DEFAULT_WINDOW_WIDTH*(1.+z)
    line_width = DEFAULT_LINE_WIDTH*(1.+z)
        
    # \\ define spectrum
    this_model, indices = build_ovlmodel ( wave, flux, z=z, window_width=window_width, line_width=line_width,
                                           add_absorption=add_absorption)
    #pmodel = models.Polynomial1D(2, c0=np.median(flux), c1=0., c2=0. )
    #this_model = this_model + pmodel 
    in_lines, in_window = get_lineblocs ( wave, z, window_width=window_width, line_width=line_width )
    
    fitter = fitting.LevMarLSQFitter ()    
    model_fit = fitter ( this_model, wave[in_window], flux[in_window] )
    #return model_fit, indices
    # \\ let's also estimate the uncertainty in the line fluxes    
    u_flux_arr = np.zeros([npull, Nlines])
    u_fc_arr = np.zeros([npull, len(CONTINUUM_TAGS)])
    u_global_arr = np.zeros([npull, 3])
    
    emission_indices = strings.where_substring(indices, 'emission')
    continuum_indices = strings.where_substring(indices, 'continuum')
    #absorption_indices = strings.where_substring(indices, 'absorption')

    for pull in range(npull):
        # \\ repull from non-line local areas of the spectrum
        frandom = np.zeros_like(wave)
        for continuum_tag in CONTINUUM_TAGS:
            line_bloc, window_bloc = get_lineblocs ( wave, z=z, lines=line_wavelengths[continuum_tag] )
            frandom[window_bloc] = np.random.choice ( flux[window_bloc&~line_bloc], size=window_bloc.sum(), replace=True)
        #\\ refit                
        random_fit = fitter ( this_model, wave[in_window], frandom[in_window] )
        
        #\\ slot in random fit values
        for index,ei in enumerate(emission_indices):
            u_flux_arr[pull,index] = compute_lineflux ( getattr(random_fit, 'amplitude_%i'%ei), getattr(random_fit, 'stddev_%i'%ei) )
       
        for index,ci in enumerate(continuum_indices):  
                      
            u_fc_arr[pull, index] = getattr ( random_fit, 'amplitude_%i' % ci ).value
            
        u_global_arr[pull, 0] = random_fit.stddev_0.value
        u_global_arr[pull, 1] = random_fit.stddev_2.value
        if add_absorption:
            u_global_arr[pull, 2] = random_fit.EW_2.value

    # \\ a better way (?) to estimate uncertainties
    # for idx in range(Nlines):
    #     # \\ get window for each linel, excluding where there are lines
    #     inbloc = abs(wave - (line_rf*(1.+z))) < (window_width/2.)        
    #     fs = np.nanstd(flux[inbloc&~in_lines])
        
        
    #elapsed = time.time() - start
    emission_df = pd.DataFrame ( index=line_wavelengths.keys(), columns=['flux', 'u_flux'] )
    global_params = pd.DataFrame ( index=['sigma_em', 'sigma_abs', 'EW_abs'], columns=['val','u_val'] )
    continuum_df = pd.DataFrame ( index=CONTINUUM_TAGS, columns=['fc','u_fc'] )
    
    for ei in emission_indices:
        lineflux = compute_lineflux ( getattr(model_fit, 'amplitude_%i'%ei), getattr(model_fit, 'stddev_%i'%ei) ) 
        #\\ x 10^-17 erg / s / cm^2
        emission_df.loc[indices[ei].strip('emission'),'flux'] = lineflux
    emission_df['u_flux'] = np.nanstd(u_flux_arr,axis=0)
    
    for ci in continuum_indices:
        fc = getattr(model_fit, 'amplitude_%i' % ci).value
        continuum_df.loc[indices[ci].strip('continuum'), 'fc'] = fc
    continuum_df['u_fc'] = np.nanstd(u_fc_arr, axis=0)
    
    global_params.loc['sigma_em', 'val'] = model_fit.stddev_0.value
    global_params.loc['sigma_abs', 'val'] = model_fit.stddev_2.value
    if add_absorption:
        global_params.loc['EW_abs', 'val'] = model_fit.EW_2.value
    
    ug = np.nanstd(u_global_arr, axis=0)
    global_params.loc['sigma_em', 'u_val'] = ug[0]
    global_params.loc['sigma_abs', 'u_val'] = ug[1]
    if add_absorption:
        global_params.loc['EW_abs', 'u_val'] = ug[2]   
    
    return (emission_df, continuum_df, global_params), (model_fit, indices)
    

def compute_absew ( amplitude, fwhm, fc ):
    abs_flux = amplitude * fwhm * np.pi # < 0
    return abs_flux / fc


################
#### LINE UTILS
###############
def compute_lineflux (amplitude, stddev):
    '''
    Integrate over astropy's Gaussian from -infty->infty
    '''
    line_flux = amplitude * np.sqrt(2.*np.pi) * stddev
    return line_flux

def ew_uncertainty ( lflux, specflux_c, u_lflux, u_specflux_c):  
    '''
    Compute uncertainty on EW measure given line flux, continuum specific
    flux, and uncertainties on each.
    
    TODO: how to handle possible sky under/over-subtraction?
    '''  
    u_ew = (u_lflux / specflux_c)**2
    u_ew += (lflux / specflux_c**2 * u_specflux_c)**2
    u_ew = np.sqrt(u_ew)
    return u_ew


def flux_calibrate ( wave, flux, photometry, transmission_dictionary ):
    flux_calibs = np.zeros(2)
    for ix,fname in enumerate('gr'):
        transmission = transmission_dictionary[fname]
        band_flux = np.trapz(flux*wave*np.interp(wave, transmission[:,0], transmission[:,1]), wave) # f_lambda
        #lambda_pivot = np.sqrt ( np.trapz(transmission[:,1], transmission[:,0]) / np.trapz(transmission[:,1]*transmission[:,0]**-2, transmission[:,0]) )

        #zp = -48.6 # erg/s/cm^2/Hz
        zp = np.trapz ( gAB_nu * c_angpers / transmission[:,0]**2 * transmission[:,0] * transmission[:,1], transmission[:,0] )
        specmag = -2.5*np.log10(band_flux/zp)
        convert = 10.**((photometry[f'{fname}_mag'] - specmag)/-2.5)
        #photflux = 10.**((clean.loc[objname, f'{fname}_fibermag'] - zp)/-2.5)
        #photflux_lambda = photflux * c_angpers / lambda_pivot**2

        # conversion to go to erg / s / cm^2 / Angstrom
        #convert = photflux_lambda / band_flux 
        
        flux_calibs[ix] = convert
    conversion = np.mean(flux_calibs)    
    return conversion, flux_calibs

def compute_zeropoint ( wave, transmission ):
    gAB_nu = (3631.*u.Jy).to(u.erg/u.s/u.cm**2/u.Hz).value
    c_angpers = co.c.to(u.AA/u.s).value

    numerator = np.trapz ( wave*transmission, wave ) 
    denominator = np.trapz ( gAB_nu * c_angpers / wave**2 * wave * transmission, wave )
    zp = -2.5*np.log10(numerator/denominator)    
    return zp

def get_mag ( wave, flux, transmission ):
    '''
    Compute AB magnitude through filter curve described by [transmission], assuming
    that the [flux] is in erg/s/cm^2/Angstrom and the [wave] is in Angstrom
    '''
    numerator = np.trapz(flux*wave*np.interp(wave, transmission[:,0], transmission[:,1]), wave) # f_lambda
    zp = np.trapz ( gAB_nu * c_angpers / transmission[:,0]**2 * transmission[:,0] * transmission[:,1], transmission[:,0] )
    specmag = -2.5*np.log10(numerator/zp)
    return specmag

def compute_kcorrect ( wave, flux,z, filter_curve ):
    '''
    Compute the K correction through the filter curve described by [filter_curve],
    correcting from redshift [z] to z=0.
    '''
    m_observed = get_mag ( wave, flux, filter_curve )
    m_restframe = get_mag ( wave/(1.+z), flux*(1.+z), filter_curve )    
    return m_observed - m_restframe