#import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import fitting
import fsps
from . import line_fitting

fitter = fitting.LevMarLSQFitter () 

def instantiate_fsps (dust2=0., metallicity=0.3, ):
    '''
    A generic "Dwarf-like" model; constant SFH (see Weisz+2014), 
    low metallicity & no dust
    
    Note that the SFH is actually a "tau model", but with const=1.
    all SF happens in the "constant mode"
    '''
    sp = fsps.StellarPopulation ( vactoair_flag=True, imf_type = 2, sfh = 1, #tau = tau, 
                                  const=1., dust_type=2, dust2=dust2,
                                  zcontinuous=1., logzsol = np.log10(metallicity) )
    return sp
    

def generate_optical_spectrum ( sp, age ):
    wave, spec = sp.get_spectrum ( tage = age, peraa=True )

    optical = (wave>3000)&(wave<9000)
    wave_opt = wave[optical] # Angstrom
    spec_opt = spec[optical] # Lsun/Angstrom    
    return wave_opt, spec_opt

def measure_balmer_ews ( wave_opt, spec_opt, windowwidth = 200., linewidth = 80. ):
    '''
    Measure Balmer absorption feature equivalent widths and flux deficits
    '''
    outside_windows, outside_lines = line_fitting.define_lineblocs ( wave_opt, windowwidth=windowwidth,
                                                                    linewidth=linewidth )    
        
    this_model = line_fitting.build_continuummodel ( wave_opt, spec_opt, windowwidth=windowwidth )
    model_fit = fitter ( this_model, wave_opt[~outside_windows&outside_lines], spec_opt[~outside_windows&outside_lines] )

    ew_arr = np.zeros(3)
    for idx,key in enumerate(['Halpha','Hbeta','Hgamma']):
        line_wl=line_fitting.line_wavelengths[key]     
        xs = np.arange(line_wl - linewidth/2. + 1, line_wl + linewidth/2. - 1,.1)
        in_transmission = abs(wave_opt-line_wl) <= (linewidth/2.)

        model_flux = np.trapz ( model_fit(xs), xs )
        spec_flux = np.trapz( spec_opt[in_transmission], wave_opt[in_transmission])
        fc_lambda = model_flux / linewidth
        ew = spec_flux / fc_lambda - linewidth
        ew_arr[idx] = ew
    return ew_arr
