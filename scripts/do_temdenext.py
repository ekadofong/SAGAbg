import os
import time
import glob
import argparse
import numpy as np
#from astropy import cosmology
#import pandas as pd
import emcee
import pyneb as pn
from SAGAbg import models, temdenext

import catalogs

def setup_run (lineratio_arrays, nwalkers, 
               Tlow=9000., Thigh=20000.,
               nelow=10., nehigh=1000.,
               Avlow=0., Avhigh=1. ):
    '''
    Initialize walkers over a uniform distribution of physically
    plausible HII region conditions:
    Te = [9000, 20000] K
    ne = [10, 1000] cc
    Av = [0 1]
    '''
    pl = [Tlow, nelow, Avlow]
    ph = [Thigh, nehigh, Avhigh]
    pinit = np.zeros([nwalkers, len(pl)])
    added = 0
    iters = 0
    while added < nwalkers:
        p0 = np.random.uniform(pl, ph, (nwalkers, len(pl)))
        lp = np.array([lineratio_arrays.log_prob(x) for x in p0])
        to_add = p0[np.isfinite(lp)][:(nwalkers-added)]
        pinit[added:(added+to_add.shape[0])] = to_add
        added += to_add.shape[0]
        iters += 1
        if iters > 10:
            break
    return pinit

def estimate_abundances ( la, fchain, species_d=None, npull=100 ):
    '''
    Estimate abundances from physical conditions inference
    '''
    if species_d is None:
        species_d = {'[OII]':('O',2),
                    '[OIII]':('O',3)}      
                 
    args = np.nanmedian(fchain,axis=0)
    
    # \\ establish baseline H indicator
    betaindex = la.espec.model.get_line_index('Hbeta')
    fhb_corr = temdenext.extinction_correction(la.espec.model.emission_lines['Hbeta'],
                                            la.espec.obs_fluxes[:,betaindex],
                                            args[-1])
    
    hrec = pn.RecAtom('H',1)
    cumulative_abundance_estimates = {}

    for species in species_d.keys():
        cumulative_abundance_estimates[species]  = np.zeros([npull,la.espec.obs_fluxes.shape[0]],dtype=float)
        
    # \\ estimate abundances using all lines from species
    for ix in range(npull):
        rint = np.random.randint(fchain.shape[0])
        args = fchain[rint]
                
        Te = la.temperature_zones ( species, args[0] )
        hbeta_emissivity = hrec.getEmissivity(Te, args[1], lev_i=4, lev_j=2) # XXX is this the right temperature?
        for species in species_d.keys():
            atom = pn.Atom(*species_d[species])
            lines = [ x for x in la.espec.model.emission_lines.keys() if species in x ]
            
            line_emissivity = 0.
            flux_corr = np.zeros(la.espec.obs_fluxes.shape[0], dtype=float)
            for idx,lname in enumerate(lines):
                Te = la.temperature_zones ( species, args[0] )
                ne = args[1]
                line = la.espec.model.emission_lines[lname]
                if isinstance(line, float):
                    line_emissivity += atom.getEmissivity(Te,ne,
                                                        *atom.getTransition(line))
                else:
                    for wl in line:                
                        line_emissivity += atom.getEmissivity(Te,ne,
                                                            *atom.getTransition(wl))
                lidx = la.espec.model.get_line_index ( lname )
                flux_corr += temdenext.extinction_correction (np.mean(la.espec.model.emission_lines[lname]), 
                                                            la.espec.obs_fluxes[:,lidx], args[-1] ) 
            intensity_ratio = flux_corr / fhb_corr
            emissivity_ratio = hbeta_emissivity / line_emissivity
            cumulative_abundance_estimates[species][ix] = emissivity_ratio * intensity_ratio
    return cumulative_abundance_estimates

def run ( lr_filename, z, nwalkers=12, nsteps=1000, discard=500, progress=True):
    '''
    Run inference
    '''
    #z = sbam.loc[wid, 'SPEC_Z']
    #fname = f'../local_data/SBAM/bayfit/{wid}/{wid}-chain.txt'
    cl = models.CoordinatedLines (z=z)
    espec = models.EmceeSpec ( cl )
    espec.load_posterior ( lr_filename )
        
    la = temdenext.LineArray ()
    la.load_observations(espec)
    
    p0 = setup_run ( la, nwalkers )
    ndim = p0.shape[1] 
    
    sampler = emcee.EnsembleSampler( nwalkers, ndim, la.log_prob, )
    out = sampler.run_mcmc( p0, nsteps, progress=progress )
    #kwargs = {'flat':True,'discard':100}
    fchain = sampler.get_chain ( flat=True, discard=discard )  
    condition_stats = np.quantile(fchain, [0.025,.16,.5,.84,0.975],axis=0 )
    # \\ save OH or save OII/OIII?  
    abundances = estimate_abundances ( la, fchain )
    oh = np.log10(abundances['[OII]'] + abundances['[OIII]'])+12.
    oiii = np.nanquantile(abundances['[OIII]'], [0.025,.16,.5,.84,0.975]) * 1e5 # abundances X 10^5
    oii = np.nanquantile(abundances['[OII]'], [0.025,.16,.5,.84,0.975]) * 1e5 # abundances X 10^5
    oh_q = np.nanquantile(oh, [0.025,.16,.5,.84,0.975])
    abundances_stats = np.array([oii,oiii,oh_q])
    np.savetxt ( lr_filename.replace('chain','conditions'), condition_stats ) 
    np.savetxt ( lr_filename.replace('chain','abundances'), abundances_stats )
    return la, sampler


def main (inputdir, dropbox_dir, start=0, end=-1, source='SBAM', clobber=False, verbose=True, **kwargs):
    '''
    Infer Te, ne, and Av of the galaxy spectrum
    based off of MCMC-inferred line ratios
    '''
    if source == 'SBAM':
        parent = catalogs.build_SBAM (dropbox_directory=dropbox_dir)
    elif source == 'SBAMsat':
        parent = catalogs.build_SBAM (dropbox_directory=dropbox_dir).query('p_sat_corrected==1')
        
    # \\ identify objects with line ratio measurements
    run_names = [ os.path.basename(x).split('-')[0] for x in glob.glob('%s/*/*chain.txt' % inputdir)]    
    parent = parent.reindex(run_names)
    if end == -1:
        end = None
        
    fulltime = time.time ()
    for name in parent.iloc[start:end].index:
        if verbose:
            print(f'beginning {name}')        
            start = time.time ()
        try:
            lr_filename = f'{inputdir}/{name}/{name}-chain.txt'
            if os.path.exists(lr_filename.replace('chain','abundances')) and not clobber:
                continue
            z = parent.loc[name, 'SPEC_Z']
            run (lr_filename, z, **kwargs )
        except Exception as e:
            print(f'{name} failed: {e}')
            continue
        
        if verbose:
            elapsed = time.time () - start
            overall = time.time () - fulltime
            print(f'{name} finished ({elapsed:.2f} sec laptime; {overall:.0f} sec total)')
           


if __name__ == '__main__':    
    parser = argparse.ArgumentParser ( prog='do_temdenext.py', description='ISM physical conditions inference')
    parser.add_argument ( '--input', '-i', action='store', default='../local_data/SBAM/bayfit/',
                          help='path to directory with SAGA spectra')
    parser.add_argument ( '--source', action='store', default='SBAM', )
    parser.add_argument ( '--clobber', action='store_true')

    parser.add_argument ( '--start', '-S', action='store', default=0, help='starting index')
    parser.add_argument ( '--end', '-E', action='store', default=-1, help='ending index')
    parser.add_argument ( '--dropbox_directory', '-d', action='store', default='/Users/kadofong/Dropbox/SAGA/',
                          help='path to directory with SAGA spectra')
    #parser.add_argument ( '--mpi', action='store_true' )
    args = parser.parse_args ()
    
    main ( args.input, args.dropbox_directory, start=int(args.start), end=int(args.end), source=args.source,
           clobber=args.clobber,)# multiprocess=args.mpi ) 
