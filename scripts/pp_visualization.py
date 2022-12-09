import numpy as np
import matplotlib.pyplot as plt
from SAGAbg import models
from ekfstats import sampling
import logistics
import do_bayesianfitlines

tdict = logistics.load_filters ()

def chain_from_kde ( kde_npz, npull=5000 ):
    with np.load(kde_npz) as xf:
        narguments = len(xf.keys())        
        fchain = np.zeros([npull, narguments])
        
        stddev_pull = sampling.rejection_sample_fromarray(xf['stddev_emission'][0],
                                                    xf['stddev_emission'][1],
                                                    nsamp=npull)

        for idx,key in enumerate(xf.keys()):         
            if 'flux' in key:
                flux_val = sampling.rejection_sample_fromarray(xf[key][0], xf[key][1], nsamp=npull )
                val = flux_val / (np.sqrt(2.*np.pi)*stddev_pull)
            elif 'wiggle' in key:
                x = xf[key] #np.random.normal(xf[key][2], std, 10000)
                y = np.array([0.025,0.16,.5,.84,.975])
                dy = np.diff(y)
                dx = np.diff(x)
                val = sampling.rejection_sample_fromarray(x[:-1]+dx/2., dy/dx, npull)
            else:
                val = sampling.rejection_sample_fromarray(xf[key][0], xf[key][1], nsamp=npull )
            fchain[:,idx] = val   
            
        elines = [em.split('_')[1] for em in list(xf.keys()) if 'flux' in em ]    
    return fchain, elines

def afterburner_viz ( row, fchain, emission_lines ):    
    wave,flux,_ = logistics.do_fluxcalibrate ( row, tdict, '/Users/kadofong/Dropbox/SAGA/')
    cl = models.CoordinatedLines ( z=row['SPEC_Z'], emission_lines=emission_lines )
    u_flux = cl.construct_specflux_uncertainties ( wave, flux )
    do_bayesianfitlines.qaviz(wave,flux,u_flux, fchain, cl)