#from turtle import width
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import cosmology
import SAGA  
from SAGAbg.utils import calc_kcor
from SAGAbg import SAGA_get_spectra
import fit_lines


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
    
def measure_fake_NBflux ( wavelength, flux, ivar, line_wl, width=10., continuum=0.):
    in_transmission = abs(wavelength-line_wl) <= (width/2.)
    if in_transmission.sum() == 0:
        return np.NaN, np.NaN
    centers = wavelength[in_transmission]
    bin_widths = np.diff(centers)
    bin_widths = np.insert (bin_widths, len(bin_widths), bin_widths[-1])
    line_flux = np.sum ( (flux[in_transmission] - continuum)*bin_widths )
    e_flux = np.sqrt(np.sum ( ivar[in_transmission]*bin_widths ))
    #line_flux = np.trapz(flux[in_transmission], wavelength[in_transmission])
    #e_flux = np.sqrt(np.trapz(ivar[in_transmission], wavelength[in_transmission]))
    #width = np.trapz(np.ones_like(wavelength[in_transmission]), wavelength[in_transmission])
    width = np.sum(bin_widths)
    return line_flux, e_flux

def check_for_emission ( obj, dropbox_directory='/Users/kadofong/DropBox/SAGA/', width=10. ):
    flux, wave, ivar, _ = SAGA_get_spectra.saga_get_spectrum(obj, dropbox_directory)
    
    restwv = wave/(1.+obj['SPEC_Z'])
    arr = np.zeros(len(line_wavelengths))
    u_arr = np.zeros_like(arr)
    for idx,key in enumerate(line_wavelengths.keys()):
        #contflux, e_contflux, width = measure_fake_NBflux ( restwv, flux, ivar, cont_wavelengths[key] )
        
        continuum_window = abs(restwv - cont_wavelengths[key]) <= (width/2.)
        mc = np.mean(flux[continuum_window])
        e_mc = np.std(flux[continuum_window])
        nbflux,e_lineflux = measure_fake_NBflux ( restwv, flux, ivar, line_wavelengths[key], continuum=mc, width=width )        
        #excess = (nbflux - contflux)/np.sqrt(e_lineflux**2 + e_contflux**2)
        u_nbflux = np.sqrt(e_lineflux**2 + e_mc**2)
        
        ew = nbflux / mc
        u_ew = np.sqrt(u_nbflux**2 + ew**2 * e_mc**2 )/mc
        
        #print(nbflux, u_nbflux, nbflux/u_nbflux)
        #print(ew, u_ew, ew/u_ew)
        arr[idx] = ew #excess    
        u_arr[idx] = u_ew
    return arr, u_arr

def main_fitline (dropbox_directory = '/Users/kadofong/DropBox/SAGA/'):
    clean = build_saga_catalog (dropbox_directory = dropbox_directory)
    all_the_good_spectra = clean[(clean['ZQUALITY']>=3)&((clean['TELNAME']=='AAT')|(clean['TELNAME']=='MMT'))]
    
    for objname in all_the_good_spectra['wordid']:
        arrpath = f'../local_data/line_fits/fit_{objname}.txt'
        if os.path.exists(arrpath):
            continue

        obj = all_the_good_spectra.loc[objname]
        linefit_info = fit_lines.singleton ( obj, dropbox_directory=dropbox_directory )
        np.savetxt ( arrpath, linefit_info )
        plt.close ()

def estimate_detections (intermediate_directory):
    import glob

    slist = np.array(glob.glob(intermediate_directory+'/*'))
    indices = np.array([ os.path.basename(x).split('.')[0].split('fit_')[1] for x in slist])

    linebloc = np.zeros([len(slist), 3, 7])
    for findex, filename in enumerate(slist):
        arr = np.genfromtxt(filename)
        linebloc[findex] = arr

    ew = linebloc[:,:,0] / linebloc[:,:,1]
    snr = linebloc[:,:,2] / linebloc[:,:,6]

    realdet = (snr[:,0]>1.)&(linebloc[:,0,4]>2.)&(ew[:,0]>0.)    
    return realdet, indices, linebloc, (ew,snr)

def main_fakeNB (line_df=None):
    clean = build_saga_catalog ()
    all_the_good_spectra = clean[(clean['ZQUALITY']>=3)&((clean['TELNAME']=='AAT')|(clean['TELNAME']=='MMT'))]
    if line_df is None:
        colnames = []
        for key in line_wavelengths.keys():
            colnames.extend ( [key, f'u_{key}'])
        line_df = pd.DataFrame ( index=all_the_good_spectra['wordid'], columns=colnames )
    
    for obj_index,objname in enumerate(line_df.index):
        if np.isfinite(line_df.iloc[obj_index]['Halpha']):
            continue
        try:
            ew, u_ew = check_for_emission ( all_the_good_spectra.loc[objname] )
            for idx,key in enumerate(line_wavelengths.keys()):
                line_df.loc[objname, key] = ew[idx]
                line_df.loc[objname, f'u_{key}'] = u_ew[idx]
            #print(objname)
        except Exception as e:
            print (e)
            continue
        if obj_index % 100 == 0:            
            line_df.to_csv ( '../local_data/scratch/line_df.csv' )
    line_df.to_csv('../local_data/scratch/line_df.csv')
    return line_df

if __name__ == '__main__':
    #line_df = main ()
    #line_df.to_csv('../local_data/scratch/line_df.csv')
    main_fitline ()