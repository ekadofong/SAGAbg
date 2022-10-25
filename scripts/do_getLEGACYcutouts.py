import os 
import subprocess
#from math import prod
#import sys
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
from astropy import table, coordinates
from astropy.io import fits

def load_surveybricks ( legacy_dir = None ):
    '''
    Load the region survey bricks, combine, and produce SkyCoord catalog
    '''
    if legacy_dir is None:
        legacy_dir = '../../LegacyImaging/catalogs/'
    survey_bricks_north = table.Table(fits.getdata(f"{legacy_dir}/survey-bricks-dr9-north.fits.gz", 1))
    survey_bricks_south = table.Table(fits.getdata(f"{legacy_dir}/survey-bricks-dr9-south.fits.gz", 1))

    survey_bricks_north['region'] = 'north'
    survey_bricks_south['region'] = 'south'
    
    survey_bricks = table.vstack ( [survey_bricks_north, survey_bricks_south] )
    brick_coords = coordinates.SkyCoord ( survey_bricks['ra'], survey_bricks['dec'], unit=('deg','deg') )    
    survey_bricks.add_index('brickname')
    return survey_bricks, brick_coords

def match_saga2bricks ( survey_bricks, brick_coords, saga_bg ):
    saga_coords = coordinates.SkyCoord ( saga_bg['RA'], saga_bg['DEC'], unit=('deg','deg') )
    matchindex, sep, _ = saga_coords.match_to_catalog_sky ( brick_coords )
    matching_bricks, nobj_per_brick = np.unique ( survey_bricks['brickname'][matchindex], return_counts=True )
    return matching_bricks, nobj_per_brick, matchindex


def download_brick ( row, dirstem = None, product_list=None, bands=None, savedir=None):
    if dirstem is None:
        dirstem = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/"
    if savedir is None:
        savedir = '../../LegacyImaging/cutouts/'
    if product_list is None:
        product_list = ['image','invar','maskbits','psfsize']
    if bands is None:
        bands = list('grz')        
    
    region = row['region']
    brickname = row['brickname']
    brickstem = row['brickname'][:3]
    directory = f'{dirstem}{region}/coadd/{brickstem}/{brickname}/'
    filestem = f'legacysurvey-{brickname}-'
    brickpath = f'{savedir}cutouts/{brickname}'
    if not os.path.exists(brickpath):
        os.mkdir ( brickpath )
    
    for pname in product_list:
        for band in bands:
            link = f'{directory}{filestem}{pname}-{band}.fits.fz'
            result = subprocess.run ( ['curl','-o',f'{savedir}{brickname}/{os.path.basename(link)}', link])
            
            
