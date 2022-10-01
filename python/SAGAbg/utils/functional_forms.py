import numpy as np
from scipy import integrate 

def logschechter ( m, phi_ast, M_ast, alpha ):
    '''
    phi(M) = dN/d(log_10{M})
    '''
    nd = np.log(10.)*phi_ast *(m/M_ast)**(alpha+1.)*np.e**(-m/M_ast)
    return nd

def logschechter_alog ( logm, phi_ast, logM_ast, alpha ):
    '''
    phi(log_10{M}) = dN/d(log_10{M})
    '''
    t0 = np.log(10.)*phi_ast
    t1 = 10.**((logm - logM_ast)*(alpha+1.))
    t2 = np.e**(-10.**(logm-logM_ast))
    nd =t0*t1*t2
    return nd