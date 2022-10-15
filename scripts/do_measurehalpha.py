import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as co
from astropy import cosmology
from SAGAbg import line_fitting, SAGA_get_spectra

import logistics
import catalogs
import linesearch

cosmo = cosmology.FlatLambdaCDM( 70., 0.3 )
AAT_BREAK = 5790. # lambda @ blue -> red arm break in AAT

def model2luminosity ( ls_output, obj, tdict ):
    flux2lum = lambda flux: (flux * 4.*np.pi * cosmo.luminosity_distance ( obj['SPEC_Z'] )**2).to(u.erg/u.s)
    halpha_lineflux = ls_output[0][0,3] * 1e-17 * u.erg/u.s/u.cm**2
    continuum_specflux = ls_output[3].amplitude_1 *1e-17* u.erg/u.s/u.cm**2/u.AA

    Fha_direct = halpha_lineflux
    u_Fha_direct = ls_output[0][1,3] * 1e-17 * u.erg/u.s/u.cm**2
    Lha_direct = flux2lum(Fha_direct)
    u_Lha_direct = flux2lum(u_Fha_direct) # just F * const, so uncert = u_F * const
    
    zp = line_fitting.compute_zeropoint ( tdict['r'][:,0], tdict['r'][:,1] ) 
    
    # \\ f = 10^(-0.4*(r - zp)) = 10^x
    # \\ x = -0.4r + 0.4zp
    # \\ dx/dr = -0.4
    # \\ df / dr = df/dx * dx/dr
    # \\ df/dr = 10^x ln(10) * dx/dr
    # \\ df/dr = f * ln(10) * -0.4
    bbflux = 10.**((obj['r_mag'] - zp)/-2.5)* u.erg/u.s/u.cm**2/u.AA
    u_bbflux = abs(bbflux * np.log(10.) * -0.4 )*obj['r_err']
    # \\ FHA(EW) = EW * f_bb
    # \\ dFHA/dEW = f_bb
    # \\ dFHA/df_bb = EW
    ew = halpha_lineflux / continuum_specflux
    Fha_ew = ew * bbflux
    u_continuum_specflux = ls_output[1][2] * 1e-17 * u.erg/u.s/u.cm**2/u.AA
    u_ew = line_fitting.ew_uncertainty ( halpha_lineflux, continuum_specflux, u_Fha_direct, u_continuum_specflux )    
    u_Fha_ew = np.sqrt((u_ew*bbflux)**2 + (u_bbflux * ew)**2)
    Lha_ew = flux2lum ( Fha_ew )    
    u_Lha_ew = flux2lum ( u_Fha_ew )

    return (Lha_direct, u_Lha_direct), (Lha_ew, u_Lha_ew)


def dowork ( obj, tdict, dropbox_dir, savefig=False ):
    flux, wave, _, _ = SAGA_get_spectra.saga_get_spectrum(obj, dropbox_dir)
    finite_mask = np.isfinite(flux)
    flux = flux[finite_mask]
    wave = wave[finite_mask]

    _, qfactors = line_fitting.flux_calibrate( wave, flux, obj, tdict )

    if obj['TELNAME'] == 'AAT':
        fluxcal = np.where ( wave < AAT_BREAK, flux*qfactors[0], flux*qfactors[1])*1e17
    else:
        fluxcal = flux * np.nanmean(qfactors)*1e17
    ls_output = linesearch.fit ( wave, fluxcal, z=obj['SPEC_Z'], npull=50, savefig=savefig )
    (Lha_direct, u_Lha_direct), (Lha_ew, u_Lha_ew) = model2luminosity ( ls_output, obj, tdict )
    return (Lha_direct, u_Lha_direct), (Lha_ew, u_Lha_ew), ls_output
  
        

def main (dropbox_dir, savedir, verbose=True, clobber=False, nrun=None):
    if nrun is None:
        nrun = np.inf
    clean = catalogs.build_saga_catalog ().to_pandas ()
    low_mass = clean.query('cm_logmstar<9.')
    tdict = logistics.load_filters ()
    
    completed = 0
    with open(f'{savedir}/run.log', 'w') as f:
        for wordid in low_mass.index:
            # \\ check for rerun
            objdir = f'{savedir}/{wordid}/'
            if clobber:
                pass
            elif os.path.exists(f'{objdir}/{wordid}_fluxes.dat'):
                if verbose:
                    print(f'[main] {wordid} already run. Skipping...', file=f)
                continue
                    
            if not os.path.exists ( objdir ):
                os.makedirs ( objdir )
            
            try:
                (Lha_direct, u_Lha_direct), (Lha_ew, u_Lha_ew), ls_output = dowork ( clean.loc[wordid], tdict, dropbox_dir )    
            except Exception as e:
                print(f'[{wordid} failed with:] {e}', file=f)
                continue
                
            line_fluxes, u_fc, _, model_fit = ls_output
            
            ha_lum = np.array([[Lha_direct.value, Lha_ew.value],
                               [u_Lha_direct.value, u_Lha_ew.value]])
            np.savetxt ( f'{objdir}/{wordid}_haluminosity.dat', ha_lum )
            np.savetxt ( f'{objdir}/{wordid}_fluxes.dat', line_fluxes  )
            np.savetxt ( f'{objdir}/{wordid}_ucontinuum.dat', u_fc)
            np.savetxt ( f'{objdir}/{wordid}_lineparams.dat', model_fit.parameters  )   
            f.flush ()  
            completed += 1
            if completed >= nrun:
                break      
        
    
if __name__ == '__main__':
    dropbox_dir = sys.argv[1]
    savedir = sys.argv[2]
    if len(sys.argv) > 3:
        nrun = int(sys.argv[3])
    else:
        nrun = None
    main ( dropbox_dir, savedir, nrun = nrun)