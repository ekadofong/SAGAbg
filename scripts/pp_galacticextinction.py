import numpy as np
from astropy import table
from astropy import coordinates
from astropy.io import ascii as astro_ascii
import catalogs


def main ( sbam ):
    names = ['0_10000','10000_20000','20000_30000']
    ext_table = []
    for name in names:
        ctable = astro_ascii.read(f'../local_data/galactic_extinction/output/extinction_{name}.tbl')
        del ctable['objname']
        ext_table.append(ctable)    

    ge = table.vstack(ext_table).to_pandas ()
    ge_coords = coordinates.SkyCoord ( ge['ra'], ge['dec'], unit=('deg','deg') )
    sbam_coords = coordinates.SkyCoord ( sbam['RA'], sbam['DEC'], unit=('deg','deg') )

    d2d = sbam_coords.separation ( ge_coords )
    assert d2d.max().to('arcsec').value < 1. # make sure that the indexing is good
    ge.index = sbam.index
    
    ge.to_csv('../local_data/galactic_extinction/output/extinction_formatted.csv')
    
    
if __name__=='__main__':
    sbam = catalogs.build_SBAM ()
    main ( sbam )