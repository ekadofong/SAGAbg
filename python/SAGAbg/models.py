import numpy as np
from astropy import modeling 
from .line_db import *

def get_lineblocs ( wave, z=0., lines=None, window_width=None, line_width=None ):
    '''
    Define the region under a line and the region in the window of analysis for that
    line (line + continuum)
    '''
    if window_width is None:
        window_width = DEFAULT_WINDOW_WIDTH*(1.+z)
    if line_width is None:
        line_width = DEFAULT_LINE_WIDTH*(1.+z)
    if lines is None:
        lines = np.asarray(list(line_wavelengths.values()))
    elif isinstance(lines, float):
        lines = np.asarray([lines])
        

    lines = lines*(1. + z)
    
    distance_from_lines =abs(wave[np.newaxis,:] - lines[:,np.newaxis])
    minimum_distance = np.min(distance_from_lines, axis=0)
    
    in_line = minimum_distance <= line_width/2.
    in_window = minimum_distance <= window_width/2.
    return in_line, in_window
    
def construct_absorbermodel ( ):
    def ew_abs ( x, mean=0., fc=1., EW=-2., stddev=4. ):
        return EW*fc*(stddev*np.sqrt(2.*np.pi))**-1 * np.exp ( -(x-mean)**2/(2.*stddev**2) )

    def fit_deriv(x, mean=0., fc=1., EW=-2., stddev=4.):
        """
        Gaussian1D model function derivatives.
        """
        amplitude = (EW * fc)/(stddev * np.sqrt(2.*np.pi))
        d_amplitude = np.exp(-0.5 / stddev ** 2 * (x - mean) ** 2)
        d_mean = amplitude * d_amplitude * (x - mean) / stddev ** 2
        d_stddev = amplitude * d_amplitude * (x - mean) ** 2 / stddev ** 3
        return [d_amplitude, d_mean, d_stddev]

    mod = modeling.custom_model ( ew_abs, fit_deriv=fit_deriv )   
    return mod 

def define_linemodel ( wave, flux, line_wl, z=0., ltype='emission', linewidth=None, windowwidth=None, line_wiggle=2.,
                       stddev_init=3., add_continuum=True, amplitude_init=None ):
    if linewidth is None:
        linewidth = DEFAULT_LINE_WIDTH
    if windowwidth is None:
        windowwidth = DEFAULT_WINDOW_WIDTH
        
    #line_wl = line_wl*(1.+z)
    cmask, wmask = get_lineblocs ( wave, z=z, lines=line_wl, window_width=windowwidth, line_width=linewidth )

    continuum_init = np.median(flux[wmask&~cmask])
    if ltype=='emission':        
        if amplitude_init is None:
            amplitude_init = flux[cmask].max() #- continuum_init
        
        if add_continuum:
            model_continuum = modeling.models.Box1D ( amplitude = continuum_init, x_0=line_wl*(1.+z), 
                                                     width=windowwidth )
            
        line_model = modeling.models.Gaussian1D ( amplitude = amplitude_init,
                                    mean = line_wl*(1.+z), 
                                    stddev = stddev_init )  
        line_model.mean.bounds = (line_wl*(1.+z)-line_wiggle, line_wl*(1.+z)+line_wiggle)   
        line_model.amplitude.bounds = (0., np.infty)
        line_model.stddev.bounds = (0., 4.)    
        
        if add_continuum:
            lmodel = line_model + model_continuum        
        else:
            lmodel = line_model                          
    elif ltype=='absorption':
        '''
        F = EW * fc
        F = A * sqrt(2 pi) sigma
        EW * fc = A sqrt(2 pi) sigma
        EW = A sqrt(2 pi) sigma / fc
        '''
        amplitude_init = -1. * continuum_init * 0.25
        ew_init = amplitude_init * np.sqrt(2.*np.pi) * stddev_init / continuum_init
        
        lmodel = construct_absorbermodel()( mean = line_wl*(1.+z), fc=continuum_init, stddev=stddev_init, 
                                            EW = ew_init )                
        lmodel.EW.bounds = (-10., 0.)
        lmodel.stddev.bounds = (3., 10.)
        lmodel.mean.bounds = (line_wl*(1.+z) - line_wiggle, line_wl*(1.+z) + line_wiggle)
   
    return lmodel