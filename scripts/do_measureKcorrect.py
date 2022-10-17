import sys
import numpy as np
from SAGAbg import logistics


def dowork ( obj, tdict, dropbox_dir, savefig=False ):
    wv, calibrated_spectrum = logistics.do_fluxcalibrate (obj, tdict, dropbox_dir)
    