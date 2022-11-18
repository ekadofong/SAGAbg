import numpy as np
from scipy.stats import gaussian_kde
import pyneb as pn
from . import line_db, models

c_red = np.array([-1.857, 1.04])
c_blue  = np.array([-2.156, 1.509, -0.198, 0.011])  
b0 = 2.659  
def calzetti00 ( wave, unit='AA' ):
    '''
    Calzetti attenuation where
    A(lambda) = E(B-V) * ( k(lambda) + Rv )
    and the output of this function is k(lambda) 
    '''
    if unit == 'AA':
        conversion = 1e-4
    if isinstance( wave, float) or isinstance(wave, int):
        if wave >= 6300:
            return b0 * (c_red[0] + c_red[1]/wave) #+ Rv
        elif wave < 6300:
            return b0 * (c_blue[0] + c_blue[1]/wave + c_blue[2]/wave**2 + c_blue[3]/wave**3)
        
    # \\ otherwise do array math
    klambda = np.zeros_like(wave)
    wv_red = wave[wave>=6300.] * conversion
    klambda[wave >= 6300.] = b0 * (c_red[0] + c_red[1]/wv_red) #+ Rv
    wv_blu = wave[wave<6300.] * conversion
    klambda[wave < 6300.] = b0 * (c_blue[0] + c_blue[1]/wv_blu + c_blue[2]/wv_blu**2 + c_blue[3]/wv_blu**3) #+ Rv
    return klambda

def extinction_correction (wl, flux, Av, Rv=4.05, return_magcorr=False):
    magcorr = Av * ( calzetti00(wl)/Rv + 1.)
    if return_magcorr:
        return magcorr
    corr = 10.**(-0.4 * magcorr)
    return corr * flux

# fit from Kado-Fong+2022
logml = lambda gr: 1.65*gr - 0.66 
def CM_msun ( Mg, Mr, zp_g = 5.11 ):
    loglum_g = (Mg-zp_g)/-2.5
    logsmass = logml(Mg-Mr) + loglum_g    
    return logsmass

class LineRatio ( object ):
    def __init__ ( self, element, ionization, wl0, wl1, Rv=4.05 ):
        if element in ['H','He']:
            self.atom = pn.RecAtom ( element, ionization )
        else:
            self.atom = pn.Atom ( element, ionization )
            
        self.wl0 = wl0
        self.wl1 = wl1

        if isinstance(wl0, float):
            self.len0 = 1
            self.transition0 = [ self.atom.getTransition(wl0) ]
        else:
            self.len0 = len(wl0)
            self.transition0 = [ self.atom.getTransition ( x ) for x in wl0 ]
        if isinstance(wl1, float):
            self.len1 = 1
            self.transition1 = [ self.atom.getTransition(wl1) ]
        else:
            self.len1 = len(wl1)
            self.transition1 = [ self.atom.getTransition ( x ) for x in wl1 ]
                        
        self.Rv = Rv
    
    def introduce_extinction ( self, ebv, extinction_curve = 'calzetti'):
        '''
        Assuming
        [F0/F1]_obs = [F0/F1]10^(0.4 * E(B-V) * [k0 - k1]),
        return 10^(0.4 * E(B-V) * [k0 - k1])
        '''
        if extinction_curve == 'calzetti':
            k0 = calzetti00 ( self.wl0 )
            k1 = calzetti00 ( self.wl1 )
        kdiff = k0 - k1

        corr = 10.**(0.4*ebv*kdiff)
        
        return corr    
    
    def predict_intrinsicratio ( self, temperature, density ):
        '''
        Predict what the intrinsic line ratio of the two lines should
        be at a given temperature and density
        '''
        em0 = [ self.atom.getEmissivity ( temperature, density, *self.transition0[idx] ) for idx in range(self.len0) ]
        em0 = np.sum(em0)
        em1 = [ self.atom.getEmissivity ( temperature, density, *self.transition1[idx] ) for idx in range(self.len1) ]
        em1 = np.sum(em1)
        #if np.isnan(em0):
        #    print( f'Undefined emissivity for {self.element}{self.wl0}')            
        #if np.isnan(em1):
        #    print( f'Undefined emissivity for {self.element}{self.wl1}')
        #print(em0,em1)
        return em0/em1
    
    def predict ( self, temperature, density, Av ):
        ebv = Av / self.Rv
        intrinisc_lineratio = self.predict_intrinsicratio ( temperature, density )
        extinction_correction = self.introduce_extinction ( ebv )
        return extinction_correction * intrinisc_lineratio
    
class LineArray (object):
    def __init__ ( self, line_ratios=None ):
        if line_ratios is None:
            self.line_ratios = line_db.line_ratios
        self.n_ratios = len(self.line_ratios)
        self.build ()
        
    def build ( self ):
        self.rl_objects = []
        for lratio in self.line_ratios:
            self.rl_objects.append(LineRatio ( lratio[0], lratio[1], 
                                               line_db.line_wavelengths[lratio[2]],
                                               line_db.line_wavelengths[lratio[3]]))

    def load_litobservations ( self, xs, u_xs, fluxes, ground_key='F(Hbeta)', npull=1000 ):
        '''
        Import PDFs from the literature assuming errors are Gaussian
        '''
        self.gkde = []
        self.domain = []
        for idx in range(self.n_ratios):
            lrobs = xs[idx]
            u_lrobs = u_xs[idx]
            pdf = models.produce_gaussianfn ( (np.sqrt(2.*np.pi) * u_lrobs)**-1, lrobs, u_lrobs )            
            self.gkde.append(pdf)                        
            self.domain.append ( (lrobs - u_lrobs*3, lrobs+u_lrobs*3) )
          
        self.espec.obs_fluxes = np.zeros([npull, self.espec.model.n_emission])  
        for idx,key in enumerate(self.espec.model.emission_lines):
            if key not in fluxes.index:
                continue
            #la.espec.model.emission_lines
            flux = fluxes.loc[key] * fluxes.loc[ground_key]
            u_flux = fluxes.loc[f'u_{key}']**2 * fluxes.loc[ground_key]
            flux_pulls = np.random.normal(flux, u_flux, npull)
            self.espec.obs_fluxes[:,idx] = flux_pulls
    
    def load_observations ( self, espec ):
        '''
        Do a Gaussian KDE to do a density estimate of the 
        posterior sample drawn from line fitting
        '''
        self.espec = espec
        #self.obs_fluxes = espec.obs_fluxes
        self.gkde = []
        self.domain = []
        for lrargs in self.line_ratios:
            idx0 = self.espec.model.get_line_index(lrargs[2])
            idx1 = self.espec.model.get_line_index(lrargs[3])
            lrobs = espec.obs_fluxes[:,idx0]/espec.obs_fluxes[:,idx1]
            self.gkde.append(gaussian_kde ( lrobs ))   
            self.domain.append((lrobs.min(), min(100,lrobs.max())))
        
    def temperature_zones ( self, lineratio_tuple, temperature ):
        lowt = ['[NII]','[SII]','[OII]']
        hight= ['[OIII]']
        species = lineratio_tuple[2].strip('0123456789')
        if species in lowt:
            return 0.7 * temperature + 3000.
        elif species in hight:
            return temperature
        else:
            return temperature
            
    def predict ( self, args ):
        T,ne,Av = args
        prediction = np.zeros ( self.n_ratios, dtype=float )
        for idx,lr in enumerate(self.rl_objects):
            temperature = self.temperature_zones(self.line_ratios[idx], T)
            prediction[idx] = lr.predict ( temperature, ne, Av )
        return prediction
    
    def log_prior ( self, args ):
        T,ne,Av = args

        if (T<9e3) or (T>2e4):
            self.pcode = 0
            return -np.inf
        #elif (ne<1e-4) or (ne>1e4):
        #    self.pcode = 1
        #    return -np.inf        
        elif Av < 0:
            self.pcode = 2
            return -np.inf
        
        lp = np.log(models.gaussian ( np.log10(ne), (np.sqrt(2.*np.pi) * 0.5)**-1, 2, .5 ))
        
        if not np.isfinite(lp):
            self.pcode = 3
        else:             
            self.pcode = 0
        return lp
    
    def log_likelihood ( self, args ):
        prediction = self.predict ( args )
        lnL = 0.
        for idx in range(self.n_ratios):
            lnP = np.log(float(self.gkde[idx](prediction[idx])))
            if np.isnan(lnP):
                lnL = -np.inf
                return lnL
            lnL += lnP
        return lnL
    
    def log_prob ( self, args ):
        lnprior = self.log_prior ( args )
        if not np.isfinite(lnprior):
            return -np.inf
        return lnprior + self.log_likelihood ( args )
            
            