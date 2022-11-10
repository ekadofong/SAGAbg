import argparse
import time
import os
import sys
#from multiprocessing import Pool, cpu_count
from schwimmbad import MPIPool
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import emcee
#from astropy import units as u
#from ekfparse import strings
from SAGAbg import line_fitting, models
from astropy import cosmology

import logistics, catalogs

os.environ["OMP_NUM_THREADS"] = "1"
cosmo = cosmology.FlatLambdaCDM(70.,0.3)
tdict = logistics.load_filters ()

def setup_run ( wave, flux, cl, stddev_em_init, stddev_abs_init, EW_init, p0_std, nwalkers ):
    # \\ initialize walkers
    ainit = np.zeros(cl.n_emission)
    balmer_lr = dict(zip(['Hbeta','Hgamma','Hdelta'],[2.86, 6.11,11.06]))
    oiii_lr = dict(zip(['[OIII]4959'],[2.98]))
    for idx,key in enumerate(cl.emission_lines.keys()):
        if key in balmer_lr.keys():
            ainit[idx] = ainit[list(cl.emission_lines.keys()).index("Halpha")] / (balmer_lr[key]*1.25)
        elif key in oiii_lr.keys():
            ainit[idx] = ainit[list(cl.emission_lines.keys()).index("[OIII]5007")] / (oiii_lr[key]*1.25)
        else:
            inline,_=line_fitting.get_lineblocs(wave,z=cl.z, lines=cl.emission_lines[key])
            ainit[idx] = np.nanmax(flux[inline])
        
    cinit = np.zeros(cl.n_continuum)
    for idx,key in enumerate(cl.continuum_windows.keys()):
        _,inbloc=line_fitting.get_lineblocs(wave,z=cl.z, lines=cl.continuum_windows[key])
        cinit[idx] = np.nanmedian(flux[inbloc])
    
    winit = np.random.uniform ( -0.1, 0.1, ainit.size )
    p0 = np.concatenate([ainit, winit, cinit, np.array([EW_init, stddev_em_init, stddev_abs_init])])
    p0 = p0[np.newaxis,:]
    p_init = np.random.normal(p0, p0_std*abs(p0), [nwalkers, p0.size])
     
    return p_init

def do_run (  wave, flux, z,
              nwalkers=80, nsteps=10000, p0_std = 0.1, stddev_em_init=2., stddev_abs_init=3., EW_init=-1.,
              progress=True, multiprocess=False ):    
    cl = models.CoordinatedLines (z=z)    
    u_flux = cl.construct_specflux_uncertainties ( wave, flux )
    
    p_init = setup_run ( wave, flux, cl, stddev_em_init, stddev_abs_init, EW_init, p0_std, nwalkers )
    ndim = p_init.shape[1] 
    
    # \\ run MCMC
    in_windows = u_flux>0.
    espec = models.EmceeSpec ( cl, wave[in_windows], flux[in_windows], u_flux[in_windows] )

    if multiprocess:
        #ncpu = cpu_count ()        
        with MPIPool () as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)            
                
            sampler = emcee.EnsembleSampler( nwalkers, ndim, espec.log_prob, pool=pool )
            sampler.run_mcmc(p_init, nsteps, progress=progress)    
    else:
        sampler = emcee.EnsembleSampler( nwalkers, ndim, espec.log_prob, )
        sampler.run_mcmc(p_init, nsteps, progress=progress)    
    return sampler, (cl, espec, p_init)

def qaviz ( wave,flux,u_flux, fchain, cl, fsize=3, npull=100 ):
    fig, axarr = plt.subplots(4,4,figsize=(3+fsize*1.5*4, fsize*3))
    f_axarr = axarr.flatten()
    for idx,pull in enumerate(np.random.randint ( 0, fchain.shape[0], npull )):
        cl.set_arguments ( fchain[pull] )
        for ax,key in zip(f_axarr, cl.emission_lines.keys()):
            _,inbloc = line_fitting.get_lineblocs ( wave, z=cl.z, lines=cl.emission_lines[key], window_width=80)
            if idx==0: 
                ax.errorbar(wave[inbloc], flux[inbloc],color='k',fmt='o', yerr=u_flux[inbloc],zorder=0, markersize=5 )
                ax.text (0.025, 0.975, key, transform=ax.transAxes, va='top', ha='left' )                

            #fmask = inbloc #cl.evaluate(wave[inbloc]) > 0.1
            ax.plot(wave[inbloc], cl.evaluate_no_emission(wave[inbloc]), color='b', ls='-', alpha=0.05)
            ax.plot(wave[inbloc], cl.evaluate(wave[inbloc]), color='r', ls='-', alpha=0.05)
            ax.axvline ( cl.emission_lines[key]*(1.+cl.z), color='grey', ls='--', zorder=0, lw=0.5)
    return fig, axarr
                

def do_work ( row, *args, savedir=None, makefig=True, dropbox_dir=None, nsteps=20000, nsave=5000,
             savefit=True, savefig=True, clobber=False, verbose=True, **kwargs ):
    if savedir is None:
        savedir = '../local_data/SBAM/bayfit'    
    if dropbox_dir is None:
        dropbox_dir = '/Users/kadofong/Dropbox/SAGA'   
        
    wid = row.name             
    if (not clobber) and os.path.exists ( f'{savedir}/{wid}/model_parameters.txt'):
        if verbose:
            print(f'{wid} already run. Skipping...')
        return None, None
    
    if not os.path.exists ( f'{savedir}/{wid}/' ):
        os.makedirs(f'{savedir}/{wid}/')
        
    z = row['SPEC_Z']
    wave, flux = logistics.do_fluxcalibrate ( row, tdict, dropbox_dir )         
    sampler, (cl, espec, p_init) = do_run ( wave, flux, z, *args, nsteps=nsteps, **kwargs)
    u_flux = cl.construct_specflux_uncertainties ( wave, flux )
    fchain = sampler.get_chain (flat=True, discard=nsteps - nsave )

    if makefig:         
        qaviz(wave, flux, u_flux, fchain, cl )
        if savefig:
            plt.savefig( f'{savedir}/{wid}/{wid}-QA.png' )
            plt.close ()
    if savefit:    
        np.savetxt ( f'{savedir}/{wid}/{wid}-chain.txt', fchain[-nsave:])
    return sampler
    
    
def main (dropbox_dir,*args, start=0, end=-1, nfig=10, verbose=True, savedir=None, source='SBAM', **kwargs):
    if source == 'SBAM':
        parent = catalogs.build_SBAM (dropbox_directory=dropbox_dir)
    elif source == 'SBAMsat':
        parent = catalogs.build_SBAM (dropbox_directory=dropbox_dir)
        parent = parent.query('p_sat_corrected==1') # \\ let's do testing on the satellite sample
    
    if end == -1:
        step = (parent.shape[0] - start) // nfig
        end = None
    else:
        step = (end - start) // nfig
    if step < 1:
        step = 1
    
    fulltime = time.time ()
    for idx,(name, row) in enumerate(parent.iloc[start:end].iterrows ()):
        if (idx/step).is_integer():
            makefig=True
        else:
            makefig=False
        if verbose:
            print(f'beginning {name}')        
            start = time.time ()
        try:
            do_work ( row, *args, makefig=makefig, dropbox_dir=dropbox_dir, savedir=savedir, **kwargs )
        except Exception as e:
            print(f'{name} failed: {e}')
            continue
        
        if verbose:
            elapsed = time.time () - start
            overall = time.time () - fulltime
            print(f'{name} finished ({elapsed:.2f} sec laptime; {overall:.0f} sec total)')
     

if __name__ == '__main__':    
    parser = argparse.ArgumentParser ( prog='do_fitlines.py', description='Full line fitting')
    parser.add_argument ( '--dropbox_directory', '-d', action='store', default='/Users/kadofong/Dropbox/SAGA/',
                          help='path to directory with SAGA spectra')
    parser.add_argument ( '--nfig', '-n', action='store', default=10, help='number of QA figures to generate' )
    parser.add_argument ( '--savedir', '-s', action='store', default='../local_data/SBAM/line_measurements',
                         help='path to output directory')
    parser.add_argument ( '--source', action='store', default='SBAM', )
    parser.add_argument ( '--clobber', action='store_true')
    #now = datetime.datetime.now()
    #parser.add_argument ( '--logfile', '-l', action='store', 
    #                     default=f'run_{now.year}{now.month:02d}{now.day:02d}.log' )
    parser.add_argument ( '--start', '-S', action='store', default=0, help='starting index')
    parser.add_argument ( '--end', '-E', action='store', default=0, help='ending index')
    args = parser.parse_args ()

    main ( args.dropbox_directory, nfig=int(args.nfig), start=int(args.start), end=int(args.end), source=args.source,
           clobber=args.clobber )
