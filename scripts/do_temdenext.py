import os
import time
import glob
import argparse
import numpy as np
from astropy import cosmology
#import pandas as pd
import emcee
import pyneb as pn
from ekfstats import sampling
from SAGAbg import models, temdenext, line_db

import catalogs
import logistics
tdict = logistics.load_filters()

# # # #
# \\ set up OIII 4363 ratio
O3 = pn.Atom('O',3)
tem_arr = np.logspace(3,7,100)
lrmod = O3.getEmissivity(tem_arr, 1e2, *O3.getTransition(4363.))
lrmod /= O3.getEmissivity(tem_arr, 1e2, *O3.getTransition(5007.))
interesting_limit = np.interp(2e4, tem_arr, lrmod)
# # # #

def setup_run (lineratio_arrays, nwalkers, 
               Tlow=7000., Thigh=20000.,
               nelow=10., nehigh=1000.,
               Avlow=0., Avhigh=1., fit_ne=True ):
    '''
    Initialize walkers over a uniform distribution of physically
    plausible HII region conditions:
    Te = [9000, 20000] K
    ne = [10, 1000] cc
    Av = [0 1]
    '''
    if fit_ne:    
        pl = [Tlow, Tlow, nelow, Avlow]
        ph = [Thigh,Thigh,nehigh,Avhigh]
    else:
        pl = [Tlow, Tlow, Avlow]
        ph = [Thigh, Thigh, Avhigh]
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
    lines_d = {'[OIII]':['[OIII]5007'], '[OII]':['[OII]3729']} 
    if not la.fit_ne:
        ne = 100. # XXX fix density?        
    
    # \\ establish baseline H indicator
    betaindex = la.espec.model.get_line_index('Hbeta')
    
    hrec = pn.RecAtom('H',1)
    cumulative_abundance_estimates = {}

    for species in species_d.keys():
        cumulative_abundance_estimates[species]  = np.zeros([npull,la.espec.obs_fluxes.shape[0]],dtype=float)
        
    # \\ estimate abundances using all lines from species
    for ix in range(npull):
        rint = np.random.randint(fchain.shape[0])
        if la.fit_ne:
            TeOIII, TeOII,ne, Av = fchain[rint] # pull conditions
        else:
            TeOIII, TeOII, Av = fchain[rint] # pull conditions
        
        fhb_corr = temdenext.extinction_correction(la.espec.model.emission_lines['Hbeta'],
                                                   la.espec.obs_fluxes[:,betaindex], Av)        
                

        hbeta_emissivity = hrec.getEmissivity(TeOII, ne, lev_i=4, lev_j=2) # XXX is this the right temperature?        
        for species in species_d.keys():
            if species=='[OIII]':
                Te = TeOIII
            elif species=='[OII]':
                Te = TeOII

            atom = pn.Atom(*species_d[species])
            lines = lines_d[species] 
            #[ x for x in la.espec.model.emission_lines.keys() if species in x ]
            
            line_emissivity = 0.
            flux_corr = np.zeros(la.espec.obs_fluxes.shape[0], dtype=float)
            for lname in lines:                            
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
                                                            la.espec.obs_fluxes[:,lidx], Av ) 
            intensity_ratio = flux_corr / fhb_corr
            emissivity_ratio = hbeta_emissivity / line_emissivity
            cumulative_abundance_estimates[species][ix] = emissivity_ratio * intensity_ratio
            
    oh = np.log10(cumulative_abundance_estimates['[OII]'] + cumulative_abundance_estimates['[OIII]'])+12.
    return oh, cumulative_abundance_estimates

def run ( lr_filename, row, nwalkers=12, nsteps=1000, discard=500, progress=True, fit_ne=False, require_detections=True,
         dropbox_directory=None,
         return_dataproducts=False, detection_cutoff=0.05, verbose='vv', setup_only=False):
    '''
    Run inference
    '''
    z=row['SPEC_Z']
    cl = models.CoordinatedLines (z=z)
    wave,flux,qC = logistics.do_fluxcalibrate ( row, tdict, dropbox_directory)
    if qC == 0:
        print('Flux calibration failed; skipping')
        return 1
    
    #u_flux = cl.construct_specflux_uncertainties ( wave, flux )    
    espec = models.EmceeSpec ( cl, wave, flux )
    espec.load_posterior ( lr_filename )
    
    # \\ delete red lines if they exceed max theoretical line ratio
    for line_name in line_db.remove_if_exceeds:
        if f'flux_{line_name}' not in espec.psample.keys():
            continue
        elif 'flux_[OII]3729' not in espec.psample.keys():
            print(f'deleting {line_name} - no strong line coverage')
            del espec.psample[f'flux_{line_name}'] 
            continue
        
        val = espec.psample[f'flux_{line_name}']
        num = np.trapz(val[0]*val[1],val[0])
        val = espec.psample['flux_[OII]3729']
        den = np.trapz(val[0]*val[1],val[0])
        if num/den > line_db.line_ratio_theorymax[line_name]:            
            print(f'deleting {line_name} due to unphysical line constraints (likely artifact)')
            del espec.psample[f'flux_{line_name}']    
    
    # \\ require at least one weak OII & OIII line to be detected
    # \\ where we define detection to be : see test_detection
    crucial_weaklines = ['[OIII]4363', '[OII]7320', '[OII]7330']
    pblank = np.zeros(len(crucial_weaklines),dtype=float)
    #for idx,slot in enumerate(crucial_weaklines):
    for idx,line_name in enumerate(crucial_weaklines):
        if f'flux_{line_name}' not in espec.psample.keys():
            pblank[idx] = np.NaN
            continue
        tdet = espec.test_detection( line_name ) 
        pblank[idx] = tdet
        got_det = tdet <= detection_cutoff                         
        if got_det:
            if verbose == 'vv':
                print(f"+ Detection of {line_name} ({tdet:.4f})")
            #is_detected[idx] = True
        else:
            if len(verbose) > 0:
                print(f"No detection of {line_name} ({tdet:.4f})")
              
    is_detected = pblank <= detection_cutoff
    bingpot = is_detected[0] or (is_detected[1] and is_detected[2])  # \\ XXX require OIII4363
    # \\ also run if the 4363 line ratio is better than the line ratio max
    # \\ at T(OIII)=20,000K
    if not bingpot:
        weak = sampling.rejection_sample_fromarray ( *espec.psample['flux_[OIII]4363'] )
        strong = sampling.rejection_sample_fromarray ( *espec.psample['flux_[OIII]5007'] )
        lrsamp = weak/strong
        #lrlim = np.quantile ( lrsamp, 0.95 )
        line_constraint = sampling.get_quantile_of_value(lrsamp, interesting_limit)
        if line_constraint > 0.95:
            bingpot = True
    
    with open(lr_filename.replace('bfit.npz','det.txt'), 'w') as f:
        print(pblank, file=f)
        
    # \\ run if we can detect ANY weak line OR if we get an interesting constraint off the
    # \\ upper limit on the line ratio
    if (not bingpot) and require_detections:
        with open(lr_filename.replace('bfit.npz', 'flag'), 'w') as f:
            print('failed line detection test', file=f)
            
        return_code = 3*10**len(crucial_weaklines)
        for didx in range(len(is_detected)):
            return_code += int(~is_detected[didx]) * 10**didx
        return_code = int(return_code)
        if setup_only:
            return return_code, None
        elif return_dataproducts:        
            return return_code,  None, None
        else:
            return return_code
     
    la = temdenext.LineArray (fit_ne=fit_ne)
    la.load_observations(espec)

    # \\ require constraints on line ratios to be better than 
    # \\ the physical range
    #for lidx in range(len(la.line_ratios)):
    #    is_constrained, _ = la.is_constrained ( lidx )
    #    if not is_constrained:
    #        print (f'{la.line_ratios[lidx][2]}/{la.line_ratios[lidx][3]} not constrained')
    #    if not is_constrained and require_detections:            
    #        with open(lr_filename.replace('bfit.npz', 'flag'), 'w') as f:
    #            print('failed line constraint test', file = f)
    #        if return_dataproducts:
    #            return 20+lidx,None,None
    #        else:
    #            return 20 + lidx
        
    p0 = setup_run ( la, nwalkers, fit_ne=fit_ne )
    ndim = p0.shape[1]
    
    if setup_only:
        print('Returning set-up only:')
        return la, p0     
    
    sampler = emcee.EnsembleSampler( nwalkers, ndim, la.log_prob, )
    out = sampler.run_mcmc( p0, nsteps, progress=progress )
    #kwargs = {'flat':True,'discard':100}
    fchain = sampler.get_chain ( flat=True, discard=discard )  
    condition_stats = np.quantile(fchain, [0.025,.16,.5,.84,0.975],axis=0 )
    
    # \\ save OH or save OII/OIII?  
    oh, abundances = estimate_abundances ( la, fchain )
    #oh = np.log10(abundances['[OII]'] + abundances['[OIII]'])+12.
    oiii = np.nanquantile(abundances['[OIII]'], [0.025,.16,.5,.84,0.975]) * 1e5 # abundances X 10^5
    oii = np.nanquantile(abundances['[OII]'], [0.025,.16,.5,.84,0.975]) * 1e5 # abundances X 10^5
    oh_q = np.nanquantile(oh, [0.025,.16,.5,.84,0.975])
    abundances_stats = np.array([oii,oiii,oh_q])
    np.savetxt ( lr_filename.replace('bfit.npz','conditions.txt'), condition_stats ) 
    np.savetxt ( lr_filename.replace('bfit.npz','abundances.txt'), abundances_stats )
    if return_dataproducts:
        return la, sampler, oh
    else:
        return 0 


def main (inputdir, dropbox_dir, start=0, end=-1, source='SBAM', require_detections=True, 
          clobber=False, verbose=True, barge=True, **kwargs):
    '''
    Infer Te, ne, and Av of the galaxy spectrum
    based off of MCMC-inferred line ratios
    '''    
    if source == 'SBAM':
        parent = catalogs.build_SBAM (dropbox_directory=dropbox_dir)
    elif source == 'SBAMz':
        parent = catalogs.build_SBAMz (dropbox_directory=dropbox_dir)
    elif source == 'SBAMsat':
        parent = catalogs.build_SBAM (dropbox_directory=dropbox_dir).query('p_sat_corrected==1')
    elif source == 'SBAMlm':
        import pandas as pd
        cosmo = cosmology.FlatLambdaCDM ( 70., 0.3 )
        sbam = catalogs.build_SBAM (dropbox_directory=dropbox_dir)
        distmod = cosmo.distmod(sbam['SPEC_Z'].values).value
        kcorr = pd.read_csv('../local_data/scratch/kcorrections.csv',index_col=0)
        Mr = sbam['r_mag'] - distmod - kcorr['K_r']
        Mg = sbam['g_mag'] - distmod - kcorr['K_g']
        logmstar = temdenext.CM_msun ( Mg, Mr )
    elif source == 'custom':
        names = open('/Users/kadofong/Downloads/names.txt','r').read().splitlines()
        parent = catalogs.build_SBAM ( dropbox_directory=dropbox_dir )
        parent = parent.reindex(names)
        
    # \\ identify objects with line ratio measurements
    #run_names = [ os.path.basename(x).split('-')[0] for x in glob.glob('%s/*/*bfit.npz' % inputdir)]    
    #parent = parent.reindex(run_names)
    if end == -1:
        end = None
        
    fulltime = time.time ()
    for name in parent.iloc[start:end].index:
        if verbose:
            print(f'beginning {name}')        
            start = time.time ()
        try:
            lr_filename = f'{inputdir}/{name[:2]}/{name}/{name}-bfit.npz'
            if not os.path.exists ( lr_filename ):
                print(f'{name} does not have line fit data. Skipping! ')
                continue
            if os.path.exists(lr_filename.replace('bfit.npz','abundances.txt')) and not clobber:
                print(f'{name} already run. Skipping...')
                continue
            elif os.path.exists(lr_filename.replace('bfit.npz', 'flag')) and not clobber and require_detections:
                print(f'{name} has already been shown to fail line detection tests')
                continue
            row = parent.loc[name]
            code = run (lr_filename, row, dropbox_directory=dropbox_dir, require_detections=require_detections, **kwargs )
            if code > 0:
                #print(f'{name} is missing meaningful line constraints')
                continue
        except Exception as e: 
            if barge:           
                print(f'{name} failed: {e}')
                continue
            else:
                raise Exception ( e )
        
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
    parser.add_argument ( '--delicate', action='store_true')
    parser.add_argument ( '--run_all', action='store_true')
    #parser.add_argument ( '--mpi', action='store_true' )
    args = parser.parse_args ()
    
    main ( args.input, args.dropbox_directory, start=int(args.start), end=int(args.end), source=args.source, barge=not bool(args.delicate),
           clobber=bool(args.clobber), require_detections=not bool(args.run_all))# multiprocess=args.mpi ) 
