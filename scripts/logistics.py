import numpy as np

def load_filters ( filterset='DECam' ):
    if filterset=='DECam':
        tdict = {}
        for fname in 'grz':
            transmission = np.genfromtxt(f'../local_data/filter_curves/decam/CTIO_DECam.{fname}.dat')
            tdict[fname] = transmission
        return tdict
    elif filterset == 'SDSS':
        sloan_filters = {}
        for fname in 'gr':
            transmission = np.genfromtxt(f'../local_data/filter_curves/sloan/SLOAN_SDSS.{fname}.dat')
            sloan_filters[fname] = transmission  
        return sloan_filters          
