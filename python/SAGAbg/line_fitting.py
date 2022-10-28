import numpy as np
from astropy import constants as co
from astropy import units as u
from astropy import modeling
import pyneb as pn 
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



def build_ovlmodel ( wave, flux, z=None, line_width=None, window_width=None, add_absorption=True ):
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
    
    # \\ initialize OIII lines based on [OIII]5007
    

    model_list = []
    parameter_indices = []
    for linename in line_wavelengths.keys():
        # \\ sort out continuum
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

def fit ( wave, flux, z=0., npull = 100, verbose=True, savefig=False, add_absorption=True ):
    window_width = DEFAULT_WINDOW_WIDTH*(1.+z)
    line_width = DEFAULT_LINE_WIDTH*(1.+z)
        
    # \\ define spectrum
    this_model, indices = build_ovlmodel ( wave, flux, z=z, window_width=window_width, line_width=line_width,
                                           add_absorption=add_absorption)
    #pmodel = models.Polynomial1D(2, c0=np.median(flux), c1=0., c2=0. )
    #this_model = this_model + pmodel 
    _, in_window = get_lineblocs ( wave, z, window_width=window_width, line_width=line_width )
    
    fitter = fitting.LevMarLSQFitter ()    
    model_fit = fitter ( this_model, wave[in_window], flux[in_window] )
    return model_fit, indices
    
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
    
def get_linewindow ( wave, line_wl, width=None):
    if width is None:
        width = DEFAULT_LINE_WIDTH
    in_transmission = abs(wave-line_wl) <= (width/2.)
    return in_transmission


def define_absorptionmodel ( wave, flux, line_wl, fwhm_init=10., windowwidth=None ):
    if windowwidth is None:
        windowwidth = DEFAULT_WINDOW_WIDTH
    
    cmask = get_linewindow ( wave, line_wl, windowwidth )
    continuum_init = np.median(flux[cmask])
    amplitude_init = -1. * continuum_init * 0.25
    
    model_init = models.Lorentz1D ( amplitude=amplitude_init, 
                                    x_0 = line_wl, 
                                    fwhm = fwhm_init  )
    model_init.x_0.fixed = True
    model_init.amplitude.bounds = [-np.infty, 0.]
    
    #model_continuum = models.Box1D ( amplitude = continuum_init, x_0=line_wl, width=windowwidth )
    
    total_model = model_init #+ model_continuum
    return total_model
     
def define_continuummodel ( wave, flux, line_wl, windowwidth=None ):
    if windowwidth is None:
        windowwidth = DEFAULT_WINDOW_WIDTH
    
    #cmask = get_linewindow ( wave, line_wl, windowwidth )
    #continuum_init = np.median(flux[cmask])
    
    model_init = models.Linear1D (slope=0., intercept = 0.)
    window = models.Box1D ( amplitude = 1., x_0=line_wl, width=windowwidth )
    window.amplitude.fixed = True

    total_model = model_init * window    
    return total_model


def tie_vdisp12 ( this_model ):
    return this_model.stddev_12

def tie_mean0 ( this_model ):
    return this_model.mean_0 

def tie_mean6 ( this_model ):
    return this_model.mean_6

def tie_mean10 ( this_model ):
    return this_model.mean_10

def define_lineblocs ( wave, z=0., windowwidth=None, linewidth=None ):
    if windowwidth is None:
        windowwidth = DEFAULT_WINDOW_WIDTH
    if linewidth is None:
        linewidth = DEFAULT_LINE_WIDTH
    outside_windows = abs(wave - line_wavelengths['Halpha']*(1.+z)) > windowwidth/2.
    outside_windows &= abs(wave - line_wavelengths['Hbeta']*(1.+z)) > windowwidth/2.
    outside_windows &= abs(wave - line_wavelengths['Hgamma']*(1.+z)) > windowwidth/2.

    outside_lines = abs(wave - line_wavelengths['Halpha']*(1.+z)) > linewidth/2.
    outside_lines &= abs(wave - line_wavelengths['NII6548']*(1.+z)) > linewidth/2.  
    outside_lines &= abs(wave - line_wavelengths['NII6583']*(1.+z)) > linewidth/2.   
    outside_lines &= abs(wave - line_wavelengths['Hbeta']*(1.+z)) > linewidth/2. 
    outside_lines &= abs(wave - line_wavelengths['OIII4363']*(1.+z)) > linewidth/2. 
    outside_lines &= abs(wave - line_wavelengths['Hgamma']*(1.+z)) > linewidth/2. 
    return outside_windows, outside_lines


def tie_fwhm ( this_model ):
    return this_model.fwhm_0

def compute_absew ( amplitude, fwhm, fc ):
    abs_flux = amplitude * fwhm * np.pi # < 0
    return abs_flux / fc

def build_absorptionmodel ( wave, flux, z=0. ):
    # 0 Halpha
    # 2 Hbeta
    # 4 Hgamma
    model_haABS = define_absorptionmodel ( wave, flux, line_wavelengths['Halpha']*(1.+z))
    model_hbABS = define_absorptionmodel ( wave, flux, line_wavelengths['Hbeta'] *(1.+z))
    model_hgABS = define_absorptionmodel ( wave, flux, line_wavelengths['Hgamma']*(1.+z))
    
    this_model = model_haABS + model_hbABS + model_hgABS
    this_model.fwhm_1.tied = tie_fwhm
    this_model.fwhm_2.tied = tie_fwhm
    
    return this_model

def build_continuummodel ( wave, flux, z=0., windowwidth=None ):
    # 0 Halpha
    # 2 Hbeta
    # 4 Hgamma
    model_haBOX = define_continuummodel ( wave, flux, line_wavelengths['Halpha']*(1.+z), windowwidth=windowwidth)
    model_hbBOX = define_continuummodel ( wave, flux, line_wavelengths['Hbeta'] *(1.+z), windowwidth=windowwidth)
    model_hgBOX = define_continuummodel ( wave, flux, line_wavelengths['Hgamma']*(1.+z), windowwidth=windowwidth)
    
    this_model = model_haBOX + model_hbBOX + model_hgBOX
    
    return this_model


def build_restrictedlinemodel ( wave, flux, z=None, include_absorption=True):
    '''
    A line model which fits continuum + lines for Halpha, Hbeta, Hgamma and surrounding lines (especially OIII4363).
    The local continuum for each bloc is fit by a constant (Box1D); the linewidths are all tied to be equal.
    '''
    if z is None:
        z = 0.
    model_ha = define_linemodel ( wave, flux, line_wavelengths['Halpha']*(1. + z) )
    model_niib = define_linemodel ( wave, flux, line_wavelengths['NII6548']*(1. + z ) )
    model_niir = define_linemodel ( wave, flux, line_wavelengths['NII6583']*(1. + z ) )
    model_hb = define_linemodel ( wave, flux, line_wavelengths['Hbeta']*(1. + z ) )
    model_oiii4363 = define_linemodel ( wave, flux, line_wavelengths['OIII4363']*(1. + z ) )
    model_hgamma = define_linemodel ( wave, flux, line_wavelengths['Hgamma']*(1. + z ) )
    
    if include_absorption:
        model_ha_abs = define_linemodel ( wave, flux, line_wavelengths['Halpha']*(1. + z), 'absorption')
        model_hb_abs = define_linemodel ( wave, flux, line_wavelengths['Hbeta']*(1. + z), 'absorption')
        model_hg_abs = define_linemodel ( wave, flux, line_wavelengths['Hgamma']*(1. + z), 'absorption')
        
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