import numpy as np
import pandas as pd
from astropy import cosmology, table
from astropy.io import fits
import SAGA  
from SAGAbg.utils import calc_kcor


cosmo = cosmology.FlatLambdaCDM(70.,0.3)

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
    
    #clean = estimate_stellarmass(clean)
    clean.add_index('wordid')
    return clean

def build_SBAM (*args, **kwargs):
    '''
    Load the SAGA Background AAT MMT sample
    '''
    clean = build_saga_catalog ( *args, **kwargs ).to_pandas ()
    subset = clean.query('(selection>=2)&((TELNAME=="AAT")|(TELNAME=="MMT"))&(ZQUALITY>=3)&(SPEC_Z<0.21)')
    #clean['logmstar'] = estimate_stellarmass ( clean )
    return subset

def build_SBAMz (*args, **kwargs):
    '''
    Load the SAGA Background AAT MMT sample
    '''
    clean = build_saga_catalog ( *args, **kwargs ).to_pandas ()
    subset = clean.query('(selection>=2)&((TELNAME=="AAT")|(TELNAME=="MMT"))&(ZQUALITY>=3)&(SPEC_Z<0.1)')
    return subset

# fit from Kado-Fong+2022
logml = lambda gr: 1.65*gr - 0.66 
def CM_msun ( Mg, Mr, zp_g = 5.11 ):
    loglum_g = (Mg-zp_g)/-2.5
    logsmass = logml(Mg-Mr) + loglum_g    
    return logsmass

def estimate_stellarmass ( clean, return_absmag=False, kcorr_file="../local_data/scratch/kcorrections.csv" ):
    distmod = cosmo.distmod(clean['SPEC_Z'].values).value    
    kcorr = pd.read_csv(kcorr_file,index_col=0)
    Mr = clean['r_mag'] - distmod - kcorr['K_r']
    Mg = clean['g_mag'] - distmod - kcorr['K_g']
    logmstar = CM_msun ( Mg, Mr )
    if return_absmag:
        df = pd.DataFrame ( index = logmstar.index, columns=['logmstar','Mg','Mr'])
        df['logmstar'] = logmstar
        df['Mr'] = Mr
        df['Mg'] = Mg
        return df.dropna()

    return logmstar.dropna()

def DEP_estimate_stellarmass (clean, distmod=None):
    kcorrect = calc_kcor.calc_kcor
    if distmod is None:
        distmod = cosmo.distmod ( clean['SPEC_Z'] ).value    
        
    real_kcorrect_g = kcorrect ( 'g', clean['SPEC_Z'],'gr', clean['gr'])
    clean['oc_Mg'] = clean['g_mag'] - distmod - real_kcorrect_g 
    clean["Kg"] = real_kcorrect_g
    real_kcorrect_r = kcorrect ( 'r', clean['SPEC_Z'], 'gr', clean['gr'])
    clean['oc_Mr'] = clean['r_mag'] - distmod - real_kcorrect_r
    clean["Kr"] = real_kcorrect_r
    real_kcorrect_z = kcorrect ( 'z', clean['SPEC_Z'], 'rz', clean['rz'])
    clean['oc_Mz'] = clean['z_mag'] - distmod - real_kcorrect_z 
    clean["Kz"] = real_kcorrect_z

    logml = 1.65 * (clean['oc_Mg']-clean['oc_Mr']) - 0.66
    clean['cm_logmstar'] = logml + (clean['oc_Mg']-5.11)/-2.5    
    return clean
    
def get_index ( catalog, indices ):
    if hasattr(catalog, 'index'):
        index = catalog.index
    else:
        index = catalog
    x = np.where(np.in1d(index, indices))[0]
    if len(x)==1:
        return int(x)
    else:
        return x
    
def build_GSB ( gdir = '../../gama/local_data/'):
    # \\ load catalogs from FITS
    gama_lines = table.Table(fits.getdata(f'{gdir}/GaussFitComplexv05.fits' ,1 )).to_pandas ()
    gama_spec = table.Table(fits.getdata(f'{gdir}/SpecAllv27.fits' ,1 )).to_pandas ().set_index('CATAID')
    gama_spec = gama_spec.query('IS_SBEST&IS_BEST')
    gama_smass = table.Table(fits.getdata(f'{gdir}/StellarMassesLambdarv20.fits' ,1 )).to_pandas ()
    
    # \\ line quality cuts
    gama_lines = gama_lines.set_index('CATAID')
    negative_haflux_err = gama_lines['HA_FLUX_ERR'] < 0.
    nonpositive_haflux  = gama_lines['HA_FLUX'] <= 0.
    gama_lines.loc[negative_haflux_err, 'HA_FLUX_ERR'] = np.NaN
    gama_lines.loc[nonpositive_haflux, 'HA_FLUX'] = np.NaN
    gama_lines = gama_lines.sort_values('HA_FLUX_ERR')
    keep = ~gama_lines.index.duplicated(keep='first')
    gama_lines = gama_lines.loc[keep]    
    # \\ only Halpha detections
    snr = gama_lines['HA_FLUX'] / gama_lines['HA_FLUX_ERR']
    gama_lines = gama_lines.loc[snr>5.]
    # \\ only include AAT & SDSS spectra
    surveys = gama_spec.set_index('SPECID').reindex(gama_lines['SPECID'])
    is_gama = surveys['SURVEY_CODE'] == 5
    is_sdss = surveys['SURVEY_CODE'] == 1
    gama_lines = gama_lines.loc[is_gama.values]    
    
    # \\ photometry/smass quality cuts
    qcut = gama_smass['nQ']>2
    qcut &= gama_smass['absmag_r'] < -10
    qcut &= gama_smass['Z'] > 0.001
    qcut &= gama_smass['Z'] < 0.5
    qcut &= gama_smass['absmag_r'] > -99
    gama_smass = gama_smass.loc[qcut].set_index("CATAID")    
    
    # \\ compute stellar masses in the same way as SBAM
    logmstar_gama = CM_msun ( gama_smass.reindex(gama_lines.index)['absmag_g'], gama_smass.reindex(gama_lines.index)['absmag_r'] )

    x = gama_smass.reindex(gama_lines.index)
    to_download = (logmstar_gama < 9.8)&(x['nQ']>2)&(x['Z']<=0.2)
    ll = x.loc[to_download].index
    
    # \\ join 
    adf = gama_smass.reindex(ll)[['Z','absmag_g','delabsmag_g', 'absmag_r','delabsmag_r']]
    bdf = gama_lines.reindex(ll)[[x for x in gama_lines if ('FLUX' in x) or ('EW' in x) or (x=='SPECID')]]                       
    df = adf.join(bdf)    
    
    return df