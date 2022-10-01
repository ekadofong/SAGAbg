import numpy as np
#import os
import pandas as pd
#from astropy.io import ascii
#from astropy import table

from astropy import cosmology
#from astropy.table import Table
#from astropy.io import fits
#import matplotlib.pyplot as plt

import SAGA  
#from SAGA import ObjectCuts as C

#from easyquery import QueryMaker
#from easyquery import Query
#from SAGA.database import FitsTable 

from SAGAbg.utils import calc_kcor
from SAGAbg import SAGA_get_spectra
#import namer

cosmo = cosmology.FlatLambdaCDM(70.,0.3)

# \\ mock NB parameters
line_wavelengths = {'Halpha':6563.,'OIII4363':4363., 'Hbeta':4861., }
cont_wavelengths = {'Halpha':6650., 'OIII4363':4400., 'Hbeta':4880.,}

def build_saga_catalog ( local_dir='../local_data/', dropbox_directory = '/Users/kadofong/DropBox/SAGA/',
                          name_file='../local_data/naming/names.txt'):
    # SET UP SAGA STUFF    
    saga = SAGA.QuickStart(local_dir=local_dir,
                        shared_dir=dropbox_directory)
    names = np.genfromtxt(name_file, dtype=str)
    saga.database["combined_base"].remote.path = "https://drive.google.com/uc?export=download&id=1WnGUfDCZwXEUsy4zgGFR1ez3ZE5DFtsB&confirm=t&uuid=d0f82ed0-6db5-4ca0-bb8f-6c54d44a17db"
    saga.database["combined_base"].download(overwrite=False)

    base = saga.object_catalog.load_combined_base_catalog()
    base['wordid'] = names[:,1]
    #base = saga.host_catalog.construct_host_query("paper3").filter(base)

    cleaner = (base['REMOVE']==0)&base['is_galaxy']&(base['g_mag']<30.)&(base['r_mag']<30.)#&(base['ZQUALITY']>=3)
    clean = base[cleaner].copy()

    clean['selection'] = 0
    cuts = SAGA.objects.cuts
    SAGA.utils.fill_values_by_query(clean, cuts.main_targeting_cuts, {'selection':3})
    SAGA.utils.fill_values_by_query(clean, cuts.paper1_targeting_cut&~cuts.main_targeting_cuts, {'selection':2})
    SAGA.utils.fill_values_by_query(clean, ~cuts.main_targeting_cuts&~cuts.paper1_targeting_cut, {'selection':1})
    
    clean = estimate_stellarmass(clean)
    clean.add_index('wordid')
    return clean

def estimate_stellarmass (clean):
    kcorrect = calc_kcor.calc_kcor

    distmod = cosmo.distmod(clean['SPEC_Z']).value
    real_kcorrect_g = kcorrect ( 'g', clean['SPEC_Z'],'gr', clean['gr'])
    clean['Mg'] = clean['g_mag'] - distmod - real_kcorrect_g 
    clean["Kg"] = real_kcorrect_g
    real_kcorrect_r = kcorrect ( 'r', clean['SPEC_Z'], 'gr', clean['gr'])
    clean['Mr'] = clean['r_mag'] - distmod - real_kcorrect_r
    clean["Kr"] = real_kcorrect_r
    real_kcorrect_z = kcorrect ( 'z', clean['SPEC_Z'], 'rz', clean['rz'])
    clean['Mz'] = clean['z_mag'] - distmod - real_kcorrect_z 
    clean["Kz"] = real_kcorrect_z

    logml = 1.65 * (clean['Mg']-clean['Mr']) - 0.66
    clean['cm_logmstar'] = logml + (clean['Mg']-5.11)/-2.5    
    return clean
    
def measure_fake_NBflux ( wavelength, flux, ivar, line_wl, width=10.):
    in_transmission = abs(wavelength-line_wl) <= width
    line_flux = np.trapz(flux[in_transmission], wavelength[in_transmission])
    e_flux = np.sqrt(np.trapz(ivar[in_transmission], wavelength[in_transmission]))
    return line_flux, e_flux

def check_for_emission ( obj, dropbox_directory='/Users/kadofong/DropBox/SAGA/' ):
    flux, wave, ivar, _ = SAGA_get_spectra.saga_get_spectrum(obj, dropbox_directory)

    restwv = wave/(1.+obj['SPEC_Z'])
    arr = np.zeros(len(line_wavelengths))
    for idx,key in enumerate(line_wavelengths.keys()):
        nbflux,e_lineflux = measure_fake_NBflux ( restwv, flux, ivar, line_wavelengths[key] )
        contflux, e_contflux = measure_fake_NBflux ( restwv, flux, ivar, cont_wavelengths[key] )
        excess = (nbflux - contflux)/np.sqrt(e_lineflux**2 + e_contflux**2)
        arr[idx] = excess    
    return arr

def main ():
    clean = build_saga_catalog ()
    all_the_good_spectra = clean[(clean['ZQUALITY']>=3)&((clean['TELNAME']=='AAT')|(clean['TELNAME']=='MMT'))]
    line_df = pd.DataFrame ( index=all_the_good_spectra['wordid'], columns=line_wavelengths.keys() )
    for objname in line_df.index:
        try:
            line_df.loc[objname] = check_for_emission ( all_the_good_spectra.loc[objname] )
        except Exception as e:
            print (e)
            continue
    return line_df

if __name__ == '__main__':
    line_df = main ()
    line_df.to_csv('../local_data/scratch/line_df.csv')