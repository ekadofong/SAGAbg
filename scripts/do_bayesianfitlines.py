import argparse
import time
import os
import sys
#from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from schwimmbad import MPIPool
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
    oiii_lr = dict(zip(['[OIII]4959', '[OIII]4363'],[2.98, 6.25]))
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
    cinit[cinit<0.] = 1e-5
    
    winit = np.random.uniform ( -0.1, 0.1, ainit.size )
    p0 = np.concatenate([ainit, winit, cinit, np.array([EW_init, stddev_em_init, stddev_abs_init])])
    p0 = p0[np.newaxis,:]
    p_init = np.random.normal(p0, p0_std*abs(p0), [nwalkers, p0.size])
    
    # \\ add in our physical line constraints    
    for idx,key in enumerate(cl.emission_lines.keys()):
        if key == '[SII]6717':
            p_init[:,idx] = p_init[:,list(cl.emission_lines.keys()).index("[SII]6731")]*np.random.uniform(.67, 2.22, p_init.shape[0])
        elif key == '[OII]7330':            
            p_init[:,idx] = p_init[:,list(cl.emission_lines.keys()).index("[OII]7320")]*np.random.uniform(0.8, .81, p_init.shape[0])
        elif key == '[NII]6583':
            p_init[:,idx] = p_init[:,list(cl.emission_lines.keys()).index("[NII]6548")]*np.random.uniform(2.9, 3., p_init.shape[0])
            
    return p_init

def do_run (  wave, flux, u_flux=None, z=0.,
              nwalkers=100, nsteps=10000, p0_std = 0.1, stddev_em_init=2., stddev_abs_init=3., EW_init=-1.,
              progress=True, multiprocess=True, filename=None, dryrun=False ):   
    
    # \\ only include lines that are actually covered by the spectrum
    wvmin = wave.min()
    wvmax = wave.max()
    elines = {}
    for key in line_fitting.line_wavelengths:
        obswl = ((1.+z)*np.mean(line_fitting.line_wavelengths[key])) # \\ use mean wl for unresolved multiples
        if obswl < wvmin:
            continue
        elif obswl > wvmax:
            continue
        elines[key] = np.mean(line_fitting.line_wavelengths[key])
    # \\ propagate to absorption emission
    alines = {}
    clines = {}
    for key in line_fitting.BALMER_ABSORPTION:
        if key in elines.keys():
            alines[key] = elines[key]
    for key in line_fitting.CONTINUUM_TAGS:
        if key in elines.keys():
            clines[key] = elines[key]    
        
    cl = models.CoordinatedLines (z=z, emission_lines=elines, absorption_lines=alines, continuum_windows=clines)    
    if u_flux is None:
        u_flux = cl.construct_specflux_uncertainties ( wave, flux )
    
    p_init = setup_run ( wave, flux, cl, stddev_em_init, stddev_abs_init, EW_init, p0_std, nwalkers )
    ndim = p_init.shape[1] 
    
    # \\ run MCMC
    in_windows = u_flux>0.
    espec = models.EmceeSpec ( cl, wave[in_windows], flux[in_windows], u_flux[in_windows] )
    
    # \\ if dryrun
    if dryrun:
        return espec, p_init
    
    if filename is not None:
        backend = emcee.backends.HDFBackend ( filename )
        backend.reset ( nwalkers, ndim )
    else:
        backend = None
        
    if multiprocess:
        #ncpu = cpu_count ()        
        with MPIPool () as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)            
                
            sampler = emcee.EnsembleSampler( nwalkers, ndim, espec.log_prob, pool=pool, backend=backend )
            sampler.run_mcmc(p_init, nsteps, progress=progress)    
    else:        
        sampler = emcee.EnsembleSampler( nwalkers, ndim, espec.log_prob, backend=backend)
        sampler.run_mcmc(p_init, nsteps, progress=progress)            
    return sampler, (cl, espec, p_init)

def qaviz ( wave,flux,u_flux, fchain, cl, fsize=3, npull=100 ):
    fig, axarr = plt.subplots(4,4,figsize=(3+fsize*1.5*4, fsize*3))
    f_axarr = axarr.flatten()
    for idx,pull in enumerate(np.random.randint ( 0, fchain.shape[0], npull )):
        cl.set_arguments ( fchain[pull] )
        for ax,key in zip(f_axarr, cl.emission_lines.keys()):
            obswl = cl.emission_lines[key]*(1.+cl.z)
            if (obswl<wave.min()) or (obswl>wave.max()):
                continue
            inline,inbloc = line_fitting.get_lineblocs ( wave, z=cl.z, lines=cl.emission_lines[key], window_width=80)
            if idx==0: 
                ax.errorbar(wave[inbloc], flux[inbloc],color='k',fmt='o', yerr=u_flux[inbloc],zorder=0, markersize=5 )
                ax.text (0.025, 0.975, key, transform=ax.transAxes, va='top', ha='left' )                
            ax.plot(wave[inbloc], cl.evaluate_no_emission(wave[inbloc]), color='b', ls='-', alpha=0.05, zorder=0)
            ax.plot(wave[inbloc], cl.evaluate(wave[inbloc]), color='r', ls='-', alpha=0.05, zorder=0)
            ax.axvline ( cl.emission_lines[key]*(1.+cl.z), color='grey', ls='--', zorder=0, lw=0.5)
            
            if flux[inline].size == 0:
                mmax = abs(np.nanmax(flux[inbloc]))
            else:
                mmax = abs(np.nanmax(flux[inline]))
            mmax = max(10, mmax)
            mmin = max(0.,ax.get_ylim()[0])
            ax.set_ylim ( mmin, 1.75*mmax )
    return fig, axarr
                



def get_kde_domain (x, npts, covering_factor=1.5):
    xmin,xmed,xmax = np.quantile(x, [0.,.5,1.])
    start = xmed - (xmed-xmin)*covering_factor
    end = xmed + (xmax - xmed)*covering_factor
    domain = np.linspace(start, end, npts)    
    return domain

def approximate_kde ( cl, fchain, npts=100, covering_factor=1.5 ):
    '''
    Instead of saving chains, save an approximation of a gaussian kernel density estimate
    '''
    bw = 3.*fchain.shape[0]**(-1./5.)
    line_fluxes = models.get_linefluxes ( fchain, cl.n_emission )
    
    chain_args = cl.arguments
    arr_d = {}
    for idx in range(fchain.shape[1]):
        key = chain_args[idx]        
        if idx < cl.n_emission: # \\ also save line flux PDFs
            assert 'emission' in key
            gkde = gaussian_kde ( line_fluxes[:,idx], bw_method=bw )
            domain = get_kde_domain (line_fluxes[:,idx], npts, covering_factor)
            #arr_d[key] = np.quantile ( fchain[:,idx], [0.025,0.16,.5,.84,.975] )
            arr_d[key.replace('emission','flux')] = np.array([domain,gkde(domain)])
        elif 'wiggle' in key:
            arr_d[key] = np.quantile ( fchain[:,idx], [0.025,0.16,.5,.84,.975] )
        else:
            gkde = gaussian_kde ( fchain[:,idx], bw_method=bw )            
            domain = get_kde_domain (fchain[:,idx], npts, covering_factor)
            arr_d[key] = np.array([domain,gkde(domain)]) 
    return arr_d

def do_work ( row, *args, survey='SAGA', savedir=None, makefig=True, dropbox_dir=None, nsteps=20000, nsave=5000,
             savefit=True, savefig=True, clobber=False, verbose=True, save_sampler=False, **kwargs ):
    wid = row.name
    if survey == 'SAGA':
        if savedir is None:
            savedir = '../local_data/SBAM/bayfit'   
        # \\ split into subdirectories
        savedir = f'{savedir}/{wid[:2]}/' 
        if dropbox_dir is None:
            dropbox_dir = '/Users/kadofong/Dropbox/SAGA'   
    elif survey=='GAMA':
        if savedir is None:
            savedir = '../local_data/GSB/bayfit'
        if row['SPECID'][0] == 'G':
            subdir = row['SPECID'].split('_')[0]
        else:
            subdir = 'etc'
        savedir = f'{savedir}/{subdir}/'        
            
    
    if (not clobber) and os.path.exists ( f'{savedir}/{wid}/{wid}-bfit.npz'):
        if verbose:
            print(f'{wid} already run. Skipping...')
            #print(f'{savedir}/{wid}/{wid}-bfit.npz')
        return None, (None,None,None)
    
    if not os.path.exists ( f'{savedir}/{wid}/' ):
        os.makedirs(f'{savedir}/{wid}/')
       
    if survey == 'SAGA':
        z = row['SPEC_Z']
        wave, flux, qC = logistics.do_fluxcalibrate ( row, tdict, dropbox_dir )  
        u_flux = None #cl.construct_specflux_uncertainties ( wave, flux )
    elif survey == 'GAMA':
        z = row['Z']
        wave, flux, u_flux = logistics.load_gamaspec ( row['SPECID'].strip() )
        # XXX is it the unreasonably low (1e-19 reported = 1e-36 erg/s/cm^2/AA) uncertainty that 
        # are in the GAMA variance spectra?
        u_flux = None
        
    if save_sampler:
        sampler_backend = f'{savedir}/{wid}/{wid}-backend.h5'
    else:
        sampler_backend = None
    sampler, (cl, espec, p_init) = do_run ( wave, flux, u_flux, z, *args, nsteps=nsteps, filename=sampler_backend, **kwargs)
    if u_flux is None:
        u_flux = cl.construct_specflux_uncertainties ( wave, flux )
        print(u_flux)
    fchain = sampler.get_chain (flat=True, discard=nsteps - nsave )
    chain_approximation = approximate_kde ( cl, fchain )
    
    if makefig:         
        qaviz(wave, flux, u_flux, fchain, cl )
        if savefig:
            plt.savefig( f'{savedir}/{wid}/{wid}-QA.png' )
            plt.close ()
    if savefit:
        np.savez_compressed ( f'{savedir}/{wid}/{wid}-bfit.npz', **chain_approximation )
        #np.savetxt ( f'{savedir}/{wid}/{wid}-chain.txt', fchain[-nsave:])
        #with open(f'{savedir}/{wid}/{wid}-lines.txt', 'w') as f:
        #    for key in cl.emission_lines.keys():
        #        print(key, file=f)
                
    return sampler,  (cl, espec, p_init)        
    
def main (dropbox_dir,*args, start=0, end=-1, nfig=10, verbose=True, savedir=None, source='SBAM', wid=None, **kwargs):
    if source == 'SBAM':
        parent = catalogs.build_SBAM (dropbox_directory=dropbox_dir)
        survey = 'SAGA'
    elif source == 'SBAMsat':
        parent = catalogs.build_SBAM (dropbox_directory=dropbox_dir)
        parent = parent.query('p_sat_corrected==1') # \\ let's do testing on the satellite sample
        survey = 'SAGA'
    elif source == 'GSB':
        parent = catalogs.build_GSB ()
        survey = 'GAMA'
        
    
    if wid is not None:
        if wid.isdigit ():
            wid = int(wid)
        start = catalogs.get_index ( parent, wid )
        end = start + 1
        step = 1
        print(f'{wid} -> {start}')
    else:
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
            do_work ( row, *args, survey=survey, makefig=makefig, dropbox_dir=dropbox_dir, savedir=savedir, **kwargs )
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
    parser.add_argument ( '--savedir', '-s', action='store', default=None, #'../local_data/SBAM/bayfit',
                         help='path to output directory')
    parser.add_argument ( '--source', action='store', default='SBAM', )
    parser.add_argument ( '--clobber', action='store_true')
    #now = datetime.datetime.now()
    #parser.add_argument ( '--logfile', '-l', action='store', 
    #                     default=f'run_{now.year}{now.month:02d}{now.day:02d}.log' )
    parser.add_argument ( '--start', '-S', action='store', default=0, help='starting index')
    parser.add_argument ( '--end', '-E', action='store', default=-1, help='ending index')
    parser.add_argument ( '--serial', action='store_true' )
    parser.add_argument ( '--nsteps', action='store', default=20000 )
    parser.add_argument ( '--wid', action='store', default=None)
    parser.add_argument ( '--save_sampler', action='store_true' )
    args = parser.parse_args ()
    
    
    main ( args.dropbox_directory, 
           nfig=int(args.nfig), 
           start=int(args.start), 
           end=int(args.end), 
           source=args.source,
           savedir=args.savedir,
           clobber=args.clobber, 
           wid = args.wid,
           multiprocess=not args.serial, 
           nsteps=int(args.nsteps),
           save_sampler=args.save_sampler) 
