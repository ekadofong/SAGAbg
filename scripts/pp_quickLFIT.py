import numpy as np
from scipy.optimize import minimize
import emcee

class LFitter ( object ):
    def __init__ ( self, mmin=None, mmax=None, bmin=None, bmax=None, nwalkers=32):
        self.mmin = mmin if (mmin is not None) else -np.inf
        self.mmax = mmax if (mmax is not None) else np.inf
        self.bmin = bmin if (bmin is not None) else -np.inf
        self.bmax = bmax if (bmax is not None) else np.inf
        self.nwalkers = nwalkers
        
    def log_likelihood(self, theta, x, y, yerr):
        m, b = theta
        model = m * x + b
        sigma2 = yerr**2 #+ model**2 * np.exp(2)
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    def log_prior(self, theta):
        m, b = theta
        if (self.mmin < m < self.mmax) and (self.bmin < b < self.bmax):
            return 0.0
        return -np.inf

    def log_probability(self, theta, x, y, yerr):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, x, y, yerr)
    
    def rms ( self, theta, x, y ):
        ypred = theta[0]*x + theta[1] 
        return np.sum(( y - ypred )**2)
    
    def lsquares ( self, x, y, yerr, initial):
        nll = lambda *args: -self.log_likelihood(*args)
        soln = minimize(nll, initial, args=(x, y, yerr))
        self.lsquares = soln 
        return soln.x

    def run (self, x,y,yerr, initial):
        nll = lambda *args: -self.log_likelihood(*args)    
        soln = minimize(nll, initial, args=(x, y, yerr))
        self.lsquares = soln        
        #pos = soln.x + np.random.randn(32, 2)
        pos = np.random.normal ( soln.x, abs(soln.x)*.1, [self.nwalkers,2] )
        
        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability, args=(x, y, yerr)
        )
        sampler.run_mcmc(pos, 1000, progress=True)
        self.sampler = sampler