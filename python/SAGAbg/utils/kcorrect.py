import numpy as np
'''
Estimate of K correction given optical colors
following Chilingarian+2010x
'''
g_matrix = [[0., 0., 0., 0.],
            [-0.900332, 3.97338, 0.774394, -1.09389],
            [3.65877, -8.04213, 11.0321, 0.781176],
            [-16.7457, -31.1241, -17.5553, 0],
            [87.3565, 71.5801, 0., 0.],
            [-123.671, 0, 0, 0]] 
g_matrix = np.asarray(g_matrix) # uses g-r

r_matrix = [[0.,0.,0.,0.],
            [-1.61294, 3.81378, -3.56114, 2.47133],
            [9.13285, 9.85141, -5.1432, -7.02213],
            [-81.8341, -30.3631, 38.5052, 0.0],
            [250.732, -25.0159, 0.0, 0.0],
            [-215.377, 0.0, 0.0, 0.0]]
r_matrix = np.asarray(r_matrix) # uses g-r

z_matrix = [[0.,0.,0.,0.], 
            [-1.426,  3.08833, -0.726039, 1.06364],
            [2.9386, -8.48028,  -8.18852, -1.35281],
            [8.08986, 53.5534,   13.6829, 0],
            [-93.2991, 77.1975, 0, 0],
            [133.298, 0, 0, 0]]
z_matrix = np.asarray(z_matrix) # uses r-z

            
kcorr_d = {'g':g_matrix,'r':r_matrix, 'z':z_matrix}


def kcorrect ( z, color, bandpass, zlim=0.5 ):
    if bandpass in ['g','r']:
        print('color should be g-r')
    elif bandpass == 'z':
        print('color should be r-z')
    
    if isinstance(z, float):
        kcorr_est = 0.
    else:
        kcorr_est = np.zeros(z.shape)
    for x_idx in range(5):
        for y_idx in range(4):
            coeff = kcorr_d[bandpass][x_idx, y_idx]
            term = coeff * z**x_idx * color**y_idx
            kcorr_est += term
            
    kcorr_est[z>zlim] = np.NaN # \\ only applicable to low redshift objects
    return kcorr_est

# fit from Kado-Fong+2022
logml = lambda gr: 1.65*gr - 0.66 
def colormass_masses ( z, obsgr, Mr, zp_g = 5.11 ):
    kcorr_g = kcorrect ( z, obsgr, 'g' )
    kcorr_r = kcorrect ( z, obsgr, 'r' )
    gr_restframe = obsgr + kcorr_g - + kcorr_r
    M_g = Mr + gr_restframe
    
    loglum_g = (M_g-zp_g)/-2.5
    logsmass = logml(gr_restframe) + loglum_g    
    return logsmass