import datetime
import time
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from SAGAbg import line_fitting
import catalogs, logistics

tdict = logistics.load_filters ()

def do_work ( catrow, makefig=False, savefig=True, dropbox_dir=None, 
                savedir=None, clobber=False, verbose=False, savefit=True):
    if savedir is None:
        savedir = '../local_data/SBAM/line_measurements'
    if dropbox_dir is None:
        dropbox_dir = '/Users/kadofong/Dropbox/SAGA'
    
    z = catrow['SPEC_Z']
    wid = catrow.name
    
    if (not clobber) and os.path.exists ( f'{savedir}/{wid}/model_parameters.txt'):
        if verbose:
            print(f'{wid} already run. Skipping...')
        return None, None
    if not os.path.exists ( f'{savedir}/{wid}/' ):
        os.makedirs(f'{savedir}/{wid}/')
        
    wv, calibrated_spectrum = logistics.do_fluxcalibrate (catrow, tdict, dropbox_dir)
    
    mask = wv<=8000. # red cut-off
    wv = wv[mask]
    calibrated_spectrum = calibrated_spectrum[mask]    
    
    fitdata, fitinfo = line_fitting.fit ( wv, calibrated_spectrum, z=z, npull=10, add_absorption=True )
    emission_df, continuum_df, global_params = fitdata
    model_fit, indices = fitinfo    

    if makefig: 
        QAviz(wv, calibrated_spectrum, model_fit, z )
        if savefig:
            plt.savefig( f'{savedir}/{wid}/qa_lines.png' )
            plt.close ()
    if savefit:    
        emission_df.to_csv ( f'{savedir}/{wid}/emission.csv' )
        continuum_df.to_csv(f'{savedir}/{wid}/continuum.csv')
        global_params.to_csv(f'{savedir}/{wid}/global.csv')
        np.savetxt ( f'{savedir}/{wid}/model_parameters.txt', indices, fmt='%s')
    
    return fitdata, fitinfo
  
def QAviz (wv, flux, model_fit, z):
    fig, axarr = plt.subplots(3,4, figsize=(4*4,3*3))
    f_axarr = axarr.flatten()
    inlines, _ = line_fitting.get_lineblocs ( wv, z=z )
    for axidx, (key, val) in enumerate(line_fitting.line_wavelengths.items()):
        ax = f_axarr[axidx]
        inbloc = abs(wv - (val*(1.+z))) < (100./2.*(1.+z))
        

        ax.plot(wv[inbloc],flux[inbloc], color='k' )
        fm = np.nanmean(flux[inbloc&~inlines])
        fs = np.nanstd(flux[inbloc&~inlines])
        print(fm,fs)
        ax.axhline ( fm, color='b'  )
        ax.axhspan ( fm-fs, fm+fs, color='b' , alpha=0.1 )
        #ax.scatter(wv[inbloc],calibrated_spectrum[inbloc], color='k', s=3 )
        ax.plot(wv[inbloc], model_fit(wv)[inbloc], color='r', ls='-')
        ax.text ( 0.025, 0.975, key, transform=ax.transAxes, va='top', ha='left')
        ax.axvline ( val*(1.+z), color='pink', zorder=0)

        dw = 14./2.*(1.+z)
        ax.axvspan ( val*(1.+z) - dw, val*(1.+z) + dw, color='pink', alpha=0.25, zorder=0 )
    
        ax.set_xlabel(r'$\lambda\ (\rm \AA)$')
        ax.set_ylabel(r'''$f_{\lambda}(\lambda)\times 10^{17}$
(erg s$^{-1}$ cm$^{-2}$ $\rm \AA^{-1}$''')
    plt.tight_layout ()
    return fig, axarr

def main (dropbox_dir, start=0, end=0, nfig=10, verbose=True, savedir=None):
    parent = catalogs.build_SBAM (dropbox_directory=dropbox_dir)
    
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
            do_work ( row, makefig=makefig, dropbox_dir=dropbox_dir, savedir=savedir )
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
    #now = datetime.datetime.now()
    #parser.add_argument ( '--logfile', '-l', action='store', 
    #                     default=f'run_{now.year}{now.month:02d}{now.day:02d}.log' )
    parser.add_argument ( '--start', '-S', action='store', default=0, help='starting index')
    parser.add_argument ( '--end', '-E', action='store', default=0, help='ending index')
    args = parser.parse_args ()

    main ( args.dropbox_directory, nfig=args.nfig, start=int(args.start), end=int(args.end), )