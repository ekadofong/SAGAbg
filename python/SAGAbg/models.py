import numpy as np
from astropy import modeling 
from SAGAbg import line_db
from .line_db import *

def produce_gaussianfn ( A, m, s):
    fn = lambda x: gaussian(x, A,m,s)
    return fn

def gaussian ( x, A, m, s):
    if A == 'normalize':
        A = np.sqrt(2.*np.pi * s**2)**-1
    
    return A * np.exp ( -(x-m)**2 / (2.*s**2) )

def gaussian_ew ( x, EW, fc, m, s):
    return EW*fc*(s*np.sqrt(2.*np.pi))**-1 * np.exp ( -(x-m)**2/(2.*s**2) )

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
            amplitude_init = flux[cmask].max() 
        
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

# \\\ EMCEE IMPLEMENTATION OF FITTING

def get_linefluxes ( fchain, n_lines ):
    fluxes = fchain[:,:n_lines] * np.sqrt(2.*np.pi) * fchain[:,-2].reshape(-1,1)
    return fluxes

def get_equivalentwidths ( fchain, cl ):
    fluxes = get_linefluxes ( fchain, cl.n_emission )
    fc =  fchain[:,cl.n_emission:(cl.n_emission+cl.n_continuum)]
    

class CoordinatedLines ( object ):
    def __init__ ( self, emission_lines=None, absorption_lines=None, continuum_windows=None, z=0., window_width=140 ):
        if emission_lines is None:
            emission_lines = line_db.line_wavelengths
        if absorption_lines is None:
            absorption_lines = dict(zip(line_db.BALMER_ABSORPTION, 
                                    [line_db.line_wavelengths[x] for x in line_db.BALMER_ABSORPTION]))
        if continuum_windows is None:
            continuum_windows = dict(zip(line_db.CONTINUUM_TAGS, 
                                    [line_db.line_wavelengths[x] for x in line_db.CONTINUUM_TAGS]))
        
        self.emission_lines = emission_lines
        self.absorption_lines = absorption_lines
        self.continuum_windows = continuum_windows
        self.ew_ratio = {'Halpha':0.5, 'Hbeta':1., 'Hgamma':1., 'Hdelta':1.}
        self.z = z
        self.window_width = window_width
    
    @property
    def n_emission ( self ):
        return len(self.emission_lines)
    
    @property
    def n_absorption ( self ): 
        return len(self.absorption_lines)
    
    @property 
    def n_continuum ( self ):
        return len(self.continuum_windows)
    
    @property
    def n_arguments ( self ):
        return 2*len(self.emission_lines) + len(self.continuum_windows) + 3
    
    @property
    def linenames ( self ):
        return self.emission_lines.keys()

    def has_line ( self, line ):
        return line in self.linenames
    
    def get_line_index ( self, line, type='emission'):
        if type == 'emission':
            tag = 'emission_lines'
            start = 0
        #elif type == 'absorption':
        #    tag = 'absoprtion_lines'            
        elif type == 'continuum':
            tag = 'continuum_windows'
            start = 2*self.n_emission
        return list(getattr(self, tag).keys()).index(line) + start
        
    def set_arguments ( self, args ):        
        self.amplitudes = dict(zip(self.emission_lines.keys(), args[:self.n_emission]))   
        self.wiggle = dict(zip(self.emission_lines.keys(), args[self.n_emission:(2*self.n_emission)])) 
        Ncwindows = len(self.continuum_windows)
        self.continuum_specflux = dict(zip(self.continuum_windows.keys(), args[(2*self.n_emission):(2*self.n_emission+Ncwindows)]))
        #self.wiggle = args[-4]
        self.EW_abs = args[-3]
        self.stddev_em = args[-2]
        self.stddev_abs = args[-1]
    
    @property
    def arguments (self):
        args = []
        args.extend ( [ f'emission_{x}' for x in self.linenames] )
        args.extend ( [f'wiggle_{x}' for x in self.linenames]) 
        args.extend( [f'fc_{x}' for x in self.continuum_windows.keys() ])
        args.append('EW_abs')
        args.append('stddev_emission')
        args.append('stddev_absorption')
        return args
    
    def evaluate (self, xs):
        A = self.amplitudes
        #EW = self.EW_abs
        #fc = self.continuum_specflux
        wiggle = self.wiggle
        stddev_em = self.stddev_em
        #stddev_abs = self.stddev_abs
        z = self.z
        
        value = np.zeros_like(xs, dtype=float)
        # \\ add emission components
        for emission_key in self.emission_lines:               
            value += gaussian (xs, A[emission_key], (self.emission_lines[emission_key] + wiggle[emission_key])*(1.+z), stddev_em )
        nonemcomponents = self.evaluate_no_emission ( xs )
        
        return value + nonemcomponents
    
    def evaluate_no_emission ( self, xs ):        
        EW = self.EW_abs
        fc = self.continuum_specflux        
        stddev_abs = self.stddev_abs
        z = self.z
        wiggle = self.wiggle
        
        value = np.zeros_like(xs, dtype=float)
        for absorption_key in self.absorption_lines:
            value += gaussian_ew (xs, EW*self.ew_ratio[absorption_key], fc[absorption_key], 
                                  (self.absorption_lines[absorption_key] + wiggle[absorption_key])*(1.+z), 
                                  stddev_abs)
            
        for continuum_key in self.continuum_windows:   
            wl_offset = (self.continuum_windows[continuum_key] + wiggle[continuum_key])*(1.+z)         
            window = abs(xs - wl_offset) <= (self.window_width*(1.+z)/2.)
            value += window * fc[continuum_key]
        return value        
    
    def construct_specflux_uncertainties ( self, wave, flux ):
        u_flux = np.zeros_like(flux)
        for continuum_tag in self.continuum_windows.keys():
            line_bloc, window_bloc = get_lineblocs ( wave, z=self.z, lines=self.continuum_windows[continuum_tag] )
            fs = np.nanstd ( flux[window_bloc&~line_bloc])
            u_flux[window_bloc] = fs
            #frandom[window_bloc] += np.random.normal ( 0., fs, window_bloc.sum())    
        return u_flux
    

class EmceeSpec ( object ):
    def __init__ ( self, model, wave=None, flux=None, u_flux=None, lineratio_eps=0.5 ):
        self.model = model
        self.wave = wave
        self.flux = flux
        self.u_flux = u_flux
        self.lineratio_eps = lineratio_eps
        self.pcode = 0
        
    def load_posterior ( self, fname, sample_fluxes=True, nsample=1000):
        gkde_d = {}
        with np.load ( fname ) as npz:
            for key in npz.keys():
                gkde_d[key] = npz[key]
        self.psample = gkde_d
        
        if sample_fluxes:
            from ekfstats import sampling
            flux_keys = [ key for key in self.psample.keys() if 'flux' in key ]
            self.obs_fluxes = np.zeros([nsample, self.model.n_emission])
            for idx,fkey in enumerate(flux_keys):
                self.obs_fluxes[:,idx] = sampling.rejection_sample_fromarray (*self.psample[fkey], nsamp=nsample)

        
    def _values_to_arr ( self, x ):
        return np.asarray(list(x.values()))

    def physratio_logprior ( self, lr, bound, k=30, b=0.925):#k=30, b=0.925 ):
        '''
        A logistic function which penalizes line ratios below the physical bound. In order
        for this to make sense, the redder line must be in the numerator (such that 
        reddening increases the observed line ratio over the intrinsic emissivity ratio.) This
        can be neglected in the case where the lines are sufficiently close together in
        wavelength space.
        '''
        if lr >= bound:
            return 1.
        elif lr < bound:
            return ( 1. + np.exp(-k*(lr - bound*b)))**-1

    def log_prior ( self ):
        # \\ common-sense bounds
        if (self._values_to_arr(self.model.amplitudes) < 0.).any():
            self.pcode = 1
            return -np.inf
        elif (self._values_to_arr(self.model.continuum_specflux) < 0.).any():
            self.pcode = 3
            return -np.inf
        #elif self.model.EW_abs < -10:
        #    self.pcode = 2
        #    return -np.inf
        elif self.model.EW_abs > 0.:
            self.pcode = 2
            return -np.inf
        elif self.model.stddev_em <= 1.:
            self.pcode = 4
            return -np.inf
        elif self.model.stddev_abs <= 1.:
            self.pcode = 4 
            return -np.inf
        
        # \\ Gaussian prior on abs EW
        ews = 3.
        lp = np.log(gaussian(self.model.EW_abs, (np.sqrt(2.*np.pi) * ews**2)**-1, 0., ews))
            
        # \\ Gaussian prior on line wiggle    
        wiggle_s = 0.3
        lp += sum(np.log(gaussian(self._values_to_arr(self.model.wiggle), (np.sqrt(2.*np.pi) * wiggle_s**2)**-1, 0., wiggle_s)))
        nii_doublet = 2.9421684623736297
        nii_lr = self.model.amplitudes['[NII]6583']/self.model.amplitudes['[NII]6548']
        lp += np.log(gaussian(nii_lr, (np.sqrt(2.*np.pi) * (0.1*nii_doublet)**2)**-1, nii_doublet,  (0.1*nii_doublet)**2 ))
        # \\ physics-based bounds: computed at T=1e4 K and ne = 100 cc 
        
        if self.model.has_line('Halpha'):
            if self.model.has_line('Hbeta'):
                lp += np.log(self.physratio_logprior(self.model.amplitudes['Halpha'] /self.model.amplitudes['Hbeta'], 2.86 ))
            if self.model.has_line('Hgamma'):
                lp += np.log(self.physratio_logprior(self.model.amplitudes['Halpha'] /self.model.amplitudes['Hgamma'], 6.11 ))
            if self.model.has_line('Hdelta'):
                lp += np.log(self.physratio_logprior(self.model.amplitudes['Halpha'] /self.model.amplitudes['Hdelta'], 11.06 ))
                
        if self.model.has_line('[OIII]5007'):
            if self.model.has_line('[OIII]4959'):
                lp += np.log(self.physratio_logprior(self.model.amplitudes['[OIII]5007'] / self.model.amplitudes['[OIII]4959'], 2.98 ))
            if self.model.has_line('[OIII]4363'):
                lp += np.log(self.physratio_logprior(self.model.amplitudes['[OIII]5007'] / self.model.amplitudes['[OIII]4363'], 6.25 ))
        
        # \\ for the SII and OII doublets, make a constraint on both sides
        if self.model.has_line('[SII]6717') and self.model.has_line('[SII]6731'):
            lp += np.log(self.physratio_logprior(self.model.amplitudes['[SII]6717'] / self.model.amplitudes['[SII]6731'], 0.45 ))
            lp += np.log(self.physratio_logprior(self.model.amplitudes['[SII]6731'] / self.model.amplitudes['[SII]6717'], 0.67 )) 
            
        #if self.model.has_line('[OII]3727') and self.model.has_line('[OII]3729'):
        #    lp += np.log(self.physratio_logprior(self.model.amplitudes['[OII]3729']/self.model.amplitudes['[OII]3727'], 0.38 ))
        #    lp += np.log(self.physratio_logprior(self.model.amplitudes['[OII]3727']/self.model.amplitudes['[OII]3729'], 0.64 ))  
        #if self.model.has_line('[OII]3729') and self.model.has_line('[OII7320]'):
        #    doublet_amplitude = self.model.amplitudes['[OII]3729']
        #    lp += np.log(self.physratio_logprior(self.model.amplitudes['[OII]7330']/doublet_amplitude, 0.06))
        
        self.pcode = 0
        return lp

    def log_likelihood ( self ):
        # \\ Gaussian likelihood        
        residsq = (self.model.evaluate ( self.wave )  -  self.flux)**2
        
        inner = -0.5*np.sum(residsq / self.u_flux**2 + np.log(2.*np.pi*self.u_flux**2) )    
        return inner

    def log_prob ( self, args ):
        self.model.set_arguments(args)
        lprior = self.log_prior ( )
        if not np.isfinite(lprior):
            return -np.inf
        
        return lprior + self.log_likelihood ( )