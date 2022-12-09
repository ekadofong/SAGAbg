import numpy as np
from astropy import cosmology
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
    
    clean = estimate_stellarmass(clean)
    clean.add_index('wordid')
    return clean

def build_SBAM (*args, **kwargs):
    '''
    Load the SAGA Background AAT MMT sample
    '''
    clean = build_saga_catalog ( *args, **kwargs ).to_pandas ()
    subset = clean.query('(selection>=2)&((TELNAME=="AAT")|(TELNAME=="MMT"))&(ZQUALITY>=3)&(SPEC_Z<0.21)')
    subset['number'] = np.arange(subset.shape[0])
    return subset

def build_SBAMz (*args, **kwargs):
    '''
    Load the SAGA Background AAT MMT sample
    '''
    clean = build_saga_catalog ( *args, **kwargs ).to_pandas ()
    subset = clean.query('(selection>=2)&((TELNAME=="AAT")|(TELNAME=="MMT"))&(ZQUALITY>=3)&(SPEC_Z<0.1)')
    return subset

def estimate_stellarmass (clean, distmod=None):
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