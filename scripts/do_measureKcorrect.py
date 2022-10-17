import sys
import numpy as np
import pandas as pd
from SAGAbg import line_fitting
import catalogs, logistics

def dowork ( obj, tdict, dropbox_dir ):
    wv, calibrated_spectrum = logistics.do_fluxcalibrate (obj, tdict, dropbox_dir)
    if calibrated_spectrum is None:
        return np.NaN, np.NaN
    kcorrections = np.zeros(2)
    for idx,filtername in enumerate('gr'):
            Kcorrect = line_fitting.compute_kcorrect ( wv, calibrated_spectrum, obj['SPEC_Z'], tdict[filtername] )
            kcorrections[idx] = Kcorrect        
    return kcorrections

def main (dropbox_dir):
    clean = catalogs.build_saga_catalog (dropbox_directory=dropbox_dir).to_pandas ()
    all_the_good_spectra = clean[(clean['ZQUALITY']>=3)&((clean['TELNAME']=='AAT')|(clean['TELNAME']=='MMT'))] 
    tdict = logistics.load_filters ()
    
    kcorr_df = pd.DataFrame ( index = clean.index, columns=['K_g','K_r'], dtype=float)
    for name in all_the_good_spectra.index:
        obj = clean.loc[name]
        kcorrections = dowork ( obj, tdict, dropbox_dir )
        kcorr_df.loc[name] = kcorrections
    return kcorr_df

if __name__ == '__main__':
    kdf = main ( sys.argv[1] )
    kdf.to_csv ( '../local_data/scratch/kcorrections.csv' )