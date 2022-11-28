import numpy as np
from scipy.stats import gaussian_kde
from scipy import integrate
import pyneb as pn
from ekfstats import sampling
from . import line_db, models

temperature_zones = {'O3':'Toiii',
                     'O2':'Toii'}

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
            k0 = calzetti00 ( self.wl0[0] )
            k1 = calzetti00 ( self.wl1[0] ) # XXX assume that lines used 
            # \\ for line ratios together are also close in wavelength space.TODO: generalize
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

def index_on_string ( dictionary, indices, sep=',' ):
    values = []
    for key in indices.split(sep):
        cval = dictionary[key]
        if not isinstance(cval, list):
            cval = [cval]
        values.extend(cval)
    return values
    
class LineArray (object):
    def __init__ ( self, line_ratios=None, fit_ne=True ):
        if line_ratios is None:
            self.line_ratios = line_db.line_ratios
        else:
            self.line_ratios = line_ratios
        self.n_ratios = len(self.line_ratios)
        self.build ()
        self.fit_ne = fit_ne
        
    def build ( self ):
        self.rl_objects = []
        for lratio in self.line_ratios:
            self.rl_objects.append(LineRatio ( lratio[0], lratio[1], 
                                               index_on_string(line_db.line_wavelengths,lratio[2]),
                                               index_on_string(line_db.line_wavelengths,lratio[3])))

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
    
    # \\ XXX not currently in use, rejection sample line ratios instead of doing it
    # \\ with math
    def from_samples_load_observations ( self, espec, nsample=5000 ):
        self.espec = espec
        self.gkde = []
        self.domain = []
        for lrargs in self.line_ratios:
            names0 = [ f'flux_{x}' for x in lrargs[2].split(',') ]
            names1 = [ f'flux_{x}' for x in lrargs[3].split(',') ]
            
            numerator_flux = np.zeros(nsample)
            for name in names0:
                val = espec.psample[name]
                sample_n = sampling.rejection_sample_fromarray ( val[0], val[1], nsamp=nsample )
                numerator_flux += sample_n

            denominator_flux = np.zeros(nsample)
            for name in names1:
                val = espec.psample[name]
                sample_n = sampling.rejection_sample_fromarray ( val[0], val[1], nsamp=nsample )
                denominator_flux += sample_n

                
    def load_observations ( self, espec, npts=200 ):
        '''
        Estimate probability density functions of line ratios based off of 
        approximations to 
        '''
        self.espec = espec
        #fn = interp1d(val[0],val[1], bounds_error=False, fill_value=0.)
        self.gkde   = []
        self.domain = []
        for lrargs in self.line_ratios:
            names0 = [ f'flux_{x}' for x in lrargs[2].split(',') ]
            names1 = [ f'flux_{x}' for x in lrargs[3].split(',') ]
            
            numerator_flux = None
            numerator_pdf  = None
            for name in names0:
                ux,uy = sampling.upsample ( self.espec.psample[name][0], self.espec.psample[name][1] )
                if numerator_flux is None:
                    numerator_flux = ux
                    numerator_pdf  = uy
                else:
                    A,B = np.meshgrid ( numerator_flux, ux )
                    numerator_flux = (A + B).flatten()
                    pdfA,pdfB = np.meshgrid ( numerator_pdf, uy )
                    numerator_pdf = (pdfA*pdfB).flatten()
                    
            denominator_flux = None
            denominator_pdf  = None
            for name in names1:
                ux, uy = sampling.upsample ( self.espec.psample[name][0], self.espec.psample[name][1] )
                if denominator_flux is None:
                    denominator_flux = ux
                    denominator_pdf  = uy
                else:
                    A,B = np.meshgrid ( denominator_flux, ux )
                    denominator_flux = (A + B).flatten()
                    pdfA,pdfB = np.meshgrid ( denominator_pdf, uy )
                    denominator_pdf = (pdfA*pdfB).flatten()     
                    
            line_ratio  = sampling.cross_then_flat(numerator_flux, denominator_flux**-1)
            probdensity = sampling.cross_then_flat(numerator_pdf, denominator_pdf )
            
            xmin,xmax   = np.quantile ( line_ratio, [0.,1.] )
            xmin        = max(0., xmin)
            xmax        = min(10., xmax)
            domain      = np.linspace ( xmin, xmax, npts )   
            assns       = np.digitize ( line_ratio, domain )            
            pdf         = np.array([np.sum(probdensity[assns==x]) for x in np.arange(1, domain.shape[0]+1)])
            nrml        = np.trapz(pdf, domain)
            pdf        /= nrml
            interpfn = sampling.build_interpfn( domain, pdf )
            self.gkde.append ( interpfn )
            self.domain.append((xmin, min(100,xmax)))

    
    def DEP_load_observations ( self, espec ):
        '''
        Do a Gaussian KDE to do a density estimate of the 
        posterior sample drawn from line fitting
        '''
        self.espec = espec
        #self.obs_fluxes = espec.obs_fluxes
        self.gkde = []
        self.domain = []
        for lrargs in self.line_ratios:
            idx0 = [ self.espec.model.get_line_index(key) for key in lrargs[2].split(',') ]
            idx1 = [ self.espec.model.get_line_index(key) for key in lrargs[3].split(',') ]
            lrobs = espec.obs_fluxes[:,idx0].sum(axis=1)/espec.obs_fluxes[:,idx1].sum(axis=1)
            
            bw = 3.*lrobs.size**(-1./5.)
            #print(lrargs[0], lrobs.std(), bw)
            self.gkde.append(gaussian_kde ( lrobs, bw_method=bw ))   
            self.domain.append((lrobs.min(), min(100,lrobs.max())))
        
    def temperature_zones ( self, lineratio_tuple, temperature ):
        # XXX Andrews + 2013: this doesn't actually work?
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
        if self.fit_ne:
            Toiii, Toii, ne, Av, = args
        else:
            Toiii, Toii, Av = args
            ne=100.
        prediction = np.zeros ( self.n_ratios, dtype=float )
        for idx,lr in enumerate(self.rl_objects):
            species = ''.join([str(x) for x in self.line_ratios[idx][:2]])
            if species == 'O3':
                temperature = Toiii
            elif species == 'O2':
                temperature = Toii
            else:
                temperature = Toii # \\ assign N2 to be the OII temperature, for now
            #temperature = #self.temperature_zones(self.line_ratios[idx], T)
            prediction[idx] = lr.predict ( temperature, ne, Av )
        return prediction
    
    def log_prior ( self, args ):
        if self.fit_ne:
            Toiii, Toii, ne, Av = args
        else:
            Toiii, Toii, Av = args
            ne=100.
        m_ne = 2. # np.log10(100.)
        s_ne = 0.25
        #

        if (Toiii<7e3) or (Toiii>2e4):
            self.pcode = 0
            return -np.inf
        if (Toii<7e3) or (Toii>2e4):
            self.pcode = 0
            return -np.inf        
        #elif (ne<1e-4) or (ne>1e4):
        #    self.pcode = 1
        #    return -np.inf     #   
        elif Av < 0:
            self.pcode = 2
            return -np.inf 
        elif (ne < 100.) or (ne > 1000.):
            return -np.inf
        
        lp = np.log(models.gaussian ( np.log10(ne), (np.sqrt(2.*np.pi) * s_ne**2)**-1, m_ne, s_ne ))
        # \\ fuzzily enforce the T2-T3 relation
        # \\ based off of the relation from remeasuring metallicities of Andrews+2013
        lp += np.log(models.gaussian(Toii-Toiii, 'normalize', -2500., 1000.) )
        
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
                        
    def is_constrained ( self, lidx, alpha=0.995 ):
        xs = np.linspace(*self.domain[lidx], num=1000)
        cumtrapz = integrate.cumulative_trapezoid(self.gkde[lidx](xs),xs)
        midpts = 0.5*(xs[1:]+xs[:-1])
        u95 = np.interp(alpha, cumtrapz,midpts)
        l95 = np.interp(1.-alpha, cumtrapz, midpts)

        makes_constraint = u95 < line_db.line_ratio_theorymax[lidx]
        if line_db.line_ratio_theorymin[lidx] > 0.:
            makes_constraint |= l95 > line_db.line_ratio_theorymin[lidx]
        return makes_constraint, (l95, u95)           
            