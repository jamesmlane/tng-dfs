# ----------------------------------------------------------------------------
#
# TITLE - fitting.py
# AUTHOR - James Lane
# PROJECT - tng-dfs
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Routines for various types of fitting
'''
__author__ = "James Lane"

### Imports
import numpy as np
from . import densprofile as pdens
import scipy.stats
import multiprocessing
import emcee
import time
from . import util as putil

# ----------------------------------------------------------------------------

### Classes to hold information about fitting

# Super class

class Fit():
    '''Fit:

    Super class for all fitting information classes
    '''
    def __init__(self, *args, **kwargs):
        '''__init__:

        Initialize a Fit object
        '''
        if len(args) > 0:
            raise ValueError('Fit class does not take positional arguments')
        
        # Parameters set through kwargs
        self.nprocs = kwargs.get('nprocs', 1)
        self.nwalkers = kwargs.get('nwalkers', 100)
        self.nit = kwargs.get('nit', 1000)
        self.ncut = kwargs.get('ncut', 100)
        self.mcmc_progress = kwargs.get('mcmc_progress', True)
        self.opt_init = kwargs.get('opt_init', True)
        self.minimize_method = kwargs.get('minimize_method', 'Powell')

        # Parameters which will be set later
        self.chain = None
        self.sampler = None
        self.opt = None
    
    def _mcmc(self, loglike, init, loglike_args=[], loglike_kwargs={}):
        '''_mcmc:

        Run an MCMC fit in a manner common to all fits

        Args:

        Returns:
            chain (array): Array representing the MCMC chain
            sampler (emcee.EnsembleSampler): EnsembleSampler object
        '''
        # Parameters
        nparam = len(init)
        nwalkers = self.nwalkers
        nit = self.nit
        ncut = self.ncut
        progress = self.mcmc_progress
        usr_log_prior = self.usr_log_prior
        usr_log_prior_params = self.usr_log_prior_params
        
        def llfunc(params):
            '''llfunc:

            Log likelihood function for emcee which takes only the parameters
            '''
            return loglike(params, *loglike_args, **loglike_kwargs)
        
        # Optimize if desired
        if self.opt_init:
            res = self._minimize(loglike, init, loglike_args=loglike_args,
                loglike_kwargs=loglike_kwargs)
            init = res.x

        # Initialize the walkers
        init = np.asarray(init)
        mcmc_init = [init+1e-2*np.random.randn(nparam) for i in range(nwalkers)]
        t1 = time.time()
        with multiprocessing.Pool(processes=self.nprocs) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, nparam,
                llfunc, pool=pool)
            sampler.run_mcmc(mcmc_init, nit, progress=progress)
        t2 = time.time()
        
        # Cut the burn-in
        chain = sampler.get_chain(discard=ncut, flat=True)
        self.chain = chain
        self.sampler = sampler

        return chain, sampler

    def _minimize(self, loglike, init, loglike_args=[], loglike_kwargs={}):
        '''_minimize:

        Run a minimization fit in a manner common to all fits

        Args:

        Returns:
            res (scipy.optimize.OptimizeResult): OptimizeResult object
        '''
        def llfunc(params):
            '''llfunc:

            Log likelihood function for minimize which takes only the parameters
            '''
            return loglike(params, *loglike_args, **loglike_kwargs)
        res = scipy.optimize.minimize(llfunc, init, method='Nelder-Mead')
        self.opt = res
        return res

# Density profile fits

class DensityProfileFit(Fit):
    '''DensityProfileFit:

    Class to hold information about a density profile fit
    '''
    def __init__(self, densfunc, **kwargs):
        '''__init__:

        Initialize a DensityProfileFit object

        Args:
            densfunc (function): Function that returns the density profile. 
                Must be a DensityProfile object. Call signature is 
                densfunc(R, phi, z, params)
            usr_log_prior (function): Function that returns the log prior
                supplied by the user. Call signature is 
                usr_log_prior(densfunc, params, *usr_log_prior_params)
            
            
        '''
        super(DensityProfileFit,self).__init__([], **kwargs)
        if not isinstance(densfunc, pdens.DensityProfile):
            raise ValueError('densfunc must be a DensityProfile object')
        self.densfunc = densfunc
        self.usr_log_prior = kwargs.get('usr_log_prior', None)
        self.usr_log_prior_params = kwargs.get('usr_log_prior_params', [])
        self.effvol_params = kwargs.get('effvol_params', [])
        self.parts = kwargs.get('parts', False)
        self.n_params = densfunc.n_params
        self.n_hyperparams = 0

    def mcmc(self, init, R, phi, z, mass=None):
        '''mcmc:

        Use MCMC to fit a density profile. A wrapper for the Fit._mcmc method 
        specific to the likelihood framework for density profiles.

        Args:
            init (list): List of initial parameters for the density profile
            R, phi, z (array): Arrays of galactocentric cylindrical coordinates
                can be astropy quantities.
            mass (float): Mass of the tracer population. If None, the mass is
                set to 1. Can be an astropy quantity.
        
        Returns:
            chain (array): Array representing the MCMC chain
            sampler (emcee.EnsembleSampler): EnsembleSampler object
        '''
        R = putil.parse_astropy_quantity(R, 'kpc')
        phi = putil.parse_astropy_quantity(phi, 'rad')
        z = putil.parse_astropy_quantity(z, 'kpc')
        mass = putil.parse_astropy_quantity(mass, 'Msun')
        chain, sampler = self._mcmc(loglike_dens, init,
            loglike_args=[self.densfunc, R, phi, z],
            loglike_kwargs={'mass':mass,
                            'usr_log_prior':self.usr_log_prior,
                            'usr_log_prior_params':self.usr_log_prior_params,
                            'effvol_params':self.effvol_params,
                            'parts':self.parts})
        
        return chain, sampler
    
    def minimize(self, init, R, phi, z, mass=None, **kwargs):
        '''minimize:

        Use minimization to fit a density profile. A wrapper for the 
        Fit._minimize method specific to the likelihood framework for density 
        profiles.

        Args:
            R, phi, z (array): Arrays of galactocentric cylindrical coordinates
                can be astropy quantities.
            mass (float): Mass of the tracer population. If None, the mass is
                set to 1. Can be an astropy quantity.
        
        Returns:
            res (scipy.optimize.OptimizeResult): OptimizeResult object
        '''
        R = putil.parse_astropy_quantity(R, 'kpc')
        phi = putil.parse_astropy_quantity(phi, 'rad')
        z = putil.parse_astropy_quantity(z, 'kpc')
        mass = putil.parse_astropy_quantity(mass, 'Msun')
        res = self._minimize(mloglike_dens, init,
            loglike_args=[self.densfunc, R, phi, z],
            loglike_kwargs={'mass':mass,
                            'usr_log_prior':self.usr_log_prior,
                            'usr_log_prior_params':self.usr_log_prior_params,
                            'effvol_params':self.effvol_params,
                            'parts':self.parts})
        
        return res

# Binned density profile fits

class BinnedDensityProfileFit(Fit):
    '''BinnedDensityProfileFit:

    Class to hold information about a binned density profile fit
    '''
    def __init__(self, densfunc, **kwargs):
        '''__init__:

        Initialize a BinnedDensityProfileFit object
        '''
        super(BinnedDensityProfileFit,self).__init__()
        self.densfunc = densfunc
        if not isinstance(densfunc, pdens.DensityProfile):
            raise ValueError('densfunc must be a DensityProfile object')
        self.densfunc = densfunc
        self.usr_log_prior = kwargs.get('usr_log_prior', None)
        self.usr_log_prior_params = kwargs.get('usr_log_prior_params', [])
        self.effvol_params = kwargs.get('effvol_params', [])
        self.parts = kwargs.get('parts', False)
        self.marginalize_hyperparams = kwargs.get('marginalize_hyperparams',
            False)
        self.n_params = densfunc.n_params
        self.n_hyperparams = 1 # Just sigma for the Gaussian likelihood
    
    def mcmc(self, init, r, rho):
        '''mcmc:

        Use MCMC to fit a binned density profile. A wrapper for the Fit._mcmc
        method specific to the likelihood framework for binned density profiles.

        Args:
            r (array): Array of galactocentric spherical radii where the 
                density will be calculated for comparison with rho.
            rho (array): Array of binned density profile, calculated from the 
                data.
        
        Returns:
            chain (array): Array representing the MCMC chain
            sampler (emcee.EnsembleSampler): EnsembleSampler object
        '''
        # Insert hyperparameter into init if not already there.
        if len(init) == self.n_params or len(init) == self.n_params-1:
            init.append(1.)
        r = putil.parse_astropy_quantity(r, 'kpc')
        rho = putil.parse_astropy_quantity(rho, 'Msun/kpc^3')
        chain, sampler = self._mcmc(loglike_binned_dens, init,
            loglike_args=[self.densfunc, r, rho],
            loglike_kwargs={'usr_log_prior':self.usr_log_prior,
                            'usr_log_prior_params':self.usr_log_prior_params,
                            'effvol_params':self.effvol_params,
                            'parts':self.parts})
        if self.marginalize_hyperparams:
            chain = chain[:,:-int(self.n_hyperparams)]
        return chain, sampler
    
    def minimize(self, init, r, rho, **kwargs):
        '''minimize:

        Use minimization to fit a binned density profile. A wrapper for the 
        Fit._minimize method specific to the likelihood framework for binned 
        density profiles.

        Args:
            r (array): Array of galactocentric spherical radii where the 
                density will be calculated for comparison with rho.
            rho (array): Array of binned density profile, calculated from the 
                data.
        
        Returns:
            res (scipy.optimize.OptimizeResult): OptimizeResult object
        '''
        # Insert hyperparameter into init if not already there.
        if len(init) == self.n_params or len(init) == self.n_params-1:
            init.append(1.)
        r = putil.parse_astropy_quantity(r, 'kpc')
        rho = putil.parse_astropy_quantity(rho, 'Msun/kpc^3')
        res = self._minimize(mloglike_binned_dens, init,
            loglike_args=[self.densfunc, r, rho],
            loglike_kwargs={'usr_log_prior':self.usr_log_prior,
                            'usr_log_prior_params':self.usr_log_prior_params,
                            'effvol_params':self.effvol_params,
                            'parts':self.parts})
        if self.marginalize_hyperparams:
            res.x = res.x[:-int(self.n_hyperparams)]
        return res

# Distribution function fits

class DistributionFunctionFit(Fit):
    '''DistributionFunctionFit:

    Class to hold information about a distribution function fit
    '''
    def __init__(self):
        '''__init__:

        Initialize a DistributionFunctionFit object
        '''
        super(DistributionFunctionFit,self).__init__()


# ----------------------------------------------------------------------------

### Functions for fitting

# Density profiles

def mloglike_dens(*args,**kwargs):
    '''loglike_dens:
    
    Wrapper for the loglike_dens function that returns the negative log 
    likelihood        
    '''
    return -loglike_dens(*args,**kwargs)

def loglike_dens(params, densfunc, R, phi, z, mass=None, usr_log_prior=None, 
    usr_log_prior_params=None, effvol_params=[], parts=False):
    '''loglike_dens:
    
    Log likelihood function for fitting a density profile

    Args:
        params (list): List of parameters for the density profile
        densfunc (function): Function that returns the density profile
        R (array): Array of galactocentric cylindrical radii
        phi (array): Array of galactocentric cylindrical azimuths
        z (array): Array of galactocentric cylindrical heights
        usr_log_prior (function): Function that returns the log prior
            supplied by the user. Call signature is 
            usr_log_prior(densfunc, params, *usr_log_prior_params)

    Returns:
        loglike (float): Log likelihood of the density profile
        and if parts=True:
            logsumdens (float): Log of the sum of the density profile
            logeffvol (float): Log of the effective volume
            logprior (float): Log of the prior on the density profile
            usrlogprior (float): Log of the user supplied prior
    '''
    # Evaluate the domain prior
    if not domain_prior_dens(densfunc, params):
        return -np.inf
    # Evaluate the prior on the density profile
    logprior = logprior_dens(densfunc, params)
    # Evaluate any user supplied prior
    if callable(usr_log_prior):
        usrlogprior = usr_log_prior(densfunc, params, *usr_log_prior_params)
        if np.isinf(usrlogprior):
            return -np.inf
    else:
        usrlogprior = 0
    # Evaluate the density profile
    if mass is None:
        mass = 1.
    # else:
    #     assert len(params) == densfunc.n_params # Has amplitude
    #     _mass = mass/params[densfunc.param_names.index('amp')]
    logdens = np.log(mass*densfunc(R, phi, z, params))
    if np.any(np.isnan(logdens)):
        return -np.inf
    # Evaluate the effective volume
    effvol = densfunc.effective_volume(params, *effvol_params)/np.mean(mass)
    # logeffvol = np.log(effvol)
    # Evaluate the log likelihood
    loglike = np.sum(logdens) - effvol + logprior + usrlogprior
    if parts:
        return loglike, np.sum(logdens), effvol, logprior, usrlogprior
    else:
        return loglike

def logprior_dens(densfunc, params):
    '''logprior_dens:

    Prior on the density profile

    Args:
        densfunc (function): Function that returns the density profile
        params (list): List of parameters for the density profile
    
    Returns:
        logprior (float): Log prior on the density profile
    '''
    if isinstance(densfunc, pdens.TwoPowerSpherical):
        alpha,beta,a,_ = params
        # Place a log-uniform prior on alpha
        alpha_min = 0.001
        alpha_max = 1000.
        prior_alpha = scipy.stats.loguniform.pdf(alpha, alpha_min, alpha_max)
        # Place a log-uniform prior on beta
        beta_min = 0.001
        beta_max = 1000.
        prior_beta = scipy.stats.loguniform.pdf(beta, beta_min, beta_max)
        # Place a log-uniform prior on a
        a_min = 0.001
        a_max = 1000.
        prior_a = scipy.stats.loguniform.pdf(a, a_min, a_max)
        return np.log(prior_alpha*prior_beta*prior_a)
    if isinstance(densfunc, pdens.NFWSpherical):
        a,_ = params
        # Place a log-uniform prior on a
        a_min = 0.001
        a_max = 1000.
        prior_a = scipy.stats.loguniform.pdf(a, a_min, a_max)
        return np.log(prior_a)
    if isinstance(densfunc, pdens.SinglePowerCutoffSpherical):
        alpha,rc,_ = params
        # Place a log-uniform prior on alpha
        # alpha_min = 0.001
        # alpha_max = 1000.
        # prior_alpha = scipy.stats.loguniform.pdf(alpha, alpha_min, alpha_max)
        # Place a log-uniform prior on rc
        rc_min = 0.001
        rc_max = 100.
        prior_rc = scipy.stats.loguniform.pdf(rc, rc_min, rc_max)
        return np.log(prior_rc)
    if isinstance(densfunc, pdens.DoubleExponentialDisk):
        return 0.
        # hR,hz,_ = params
        # # Place a log-uniform prior on hR
        # hR_min = 0.001
        # hR_max = 100.
        # prior_hR = scipy.stats.loguniform.pdf(hR, hR_min, hR_max)
        # # Place a log-uniform prior on hz
        # hz_min = 0.001
        # hz_max = 100.
        # prior_hz = scipy.stats.loguniform.pdf(hz, hz_min, hz_max)
        # return np.log(prior_hR*prior_hz)
    # Handle composite density profiles recursively
    if isinstance(densfunc,pdens.CompositeDensityProfile):
        prior = 0.
        for i in range(densfunc.n_densprofiles):
            n_params = densfunc.densprofiles[i].n_params
            prior += logprior_dens(densfunc.densprofiles[i],
                params=params[i*n_params:(i+1)*n_params])
        return prior
    # if densfunc.__name__ == 'spherical':
    #     prior = scipy.stats.norm.pdf(params[0], loc=2.5, scale=1)
    #     return np.log(prior)
    return 0

def domain_prior_dens(densfunc, params):
    '''domain_prior_dens:

    Prior on the domain of the density profile

    Args:
        densfunc (function): Function that returns the density profile
        params (list): List of parameters for the density profile
    
    Returns:
        domain_prior (bool): True if the parameters are in the domain of the
            density profile
    '''
    if isinstance(densfunc, pdens.TwoPowerSpherical):
        alpha,beta,a,amp = params
        if alpha <= 0: return False
        if alpha >= beta: return False
        # if beta > 10: return False
        if a <= 0: return False
        if amp <= 0: return False
    if isinstance(densfunc, pdens.NFWSpherical):
        a,amp = params
        if a <= 0: return False
        if amp <= 0: return False
    if isinstance(densfunc, pdens.SinglePowerCutoffSpherical):
        alpha,rc,amp = params
        # if alpha <= 0: return False
        if rc <= 0: return False
        if amp <= 0: return False
    if isinstance(densfunc, pdens.DoubleExponentialDisk):
        hR,hz,amp = params
        if hR <= 0: return False
        if hz <= 0: return False
        if amp <= 0: return False
    # Handle composite density profiles recursively
    if isinstance(densfunc,pdens.CompositeDensityProfile):
        for i in range(densfunc.n_densprofiles):
            n_params = densfunc.densprofiles[i].n_params
            if not domain_prior_dens(densfunc.densprofiles[i], 
                params=params[i*n_params:(i+1)*n_params]):
                return False
    return True

def _multiprocessing_init_dens(_densfunc, _R, _phi, _z, _mass, 
    _usr_log_prior, _usr_log_prior_params, _effvol_params, _parts):
    '''_multiprocessing_init_dens:

    Initialize multiprocessing for loglike_dens (no underscores). Provides 
    global variable access for multiprocessing.
    '''
    global densfunc, R, phi, z, mass, usr_log_prior, usr_log_prior_params,\
        effvol_params, parts
    densfunc = _densfunc
    R = _R
    phi = _phi
    z = _z
    mass = _mass
    usr_log_prior = _usr_log_prior
    usr_log_prior_params  = _usr_log_prior_params
    effvol_params = _effvol_params
    parts = _parts

# Binned density profiles

def mloglike_binned_dens(*args,**kwargs):
    '''loglike_binned_dens:
    
    Wrapper for the loglike_binned_dens function that returns the negative log 
    likelihood        
    '''
    return -loglike_binned_dens(*args,**kwargs)

def loglike_binned_dens(params, densfunc, r, rho, usr_log_prior=None, 
    usr_log_prior_params=None, parts=False):
    '''loglike_binned_dens:
    
    Log likelihood function for fitting a binned density profile
    

    Args:
        params (list): List of parameters for the density profile
        densfunc (function): Function that returns the density profile
        r (array): Array of galactocentric spherical radii
        rho (array): Array of binned density profile
        usr_log_prior (function): Function that returns the log prior
            supplied by the user. Call signature is
            usr_log_prior(densfunc, params, *usr_log_prior_params)
    
    Returns:
        loglike (float): Log likelihood of the density profile
        and if parts=True:
            logdiffsum (float): sum of the squared differences between the
                density profile and the binned density profile
            logprior (float): Log of the prior on the density profile
            usrlogprior (float): Log of the user supplied prior
    '''
    # Extract hyperparameters
    # sigma = params[-1]
    # if sigma is None:
    #     sigma = 1.
    # params = params[:-1]
    # Evaluate the domain prior
    if not domain_prior_binned_dens(densfunc, params):
        return -np.inf
    # Evaluate the prior on the density profile
    logprior = logprior_binned_dens(densfunc, params)
    # Evaluate any user supplied prior
    if callable(usr_log_prior):
        usrlogprior = usr_log_prior(densfunc, params, *usr_log_prior_params)
        if np.isinf(usrlogprior):
            return -np.inf
    # Evaluate the density profile
    # dens = densfunc(r, 0., 0., params)
    # Evaluate the gaussian log-likelihood
    # logdiffsum = -0.5*np.sum((dens-rho)**2/sigma**2+np.log(2*np.pi*sigma**2))
    # Evaluate the objective function
    logdens = np.log(densfunc(r, 0., 0., params))
    logdiffsum = -np.sum(np.square(logdens-np.log(rho))) 
    loglike = logdiffsum + logprior + usrlogprior
    if parts:
        return loglike, logdiffsum, logprior, usrlogprior
    else:
        return loglike

def logprior_binned_dens(densfunc, params):
    '''logprior_binned_dens:

    Prior on the density profile

    Args:
        densfunc (function): Function that returns the density profile
        params (list): List of parameters for the density profile
    
    Returns:
        logprior (float): Log prior on the density profile
    '''
    if isinstance(densfunc, pdens.NFWSpherical):
        a,_ = params
        # Place a log-uniform prior on a
        a_min = 0.001
        a_max = 1000.
        prior_a = scipy.stats.loguniform.pdf(a, a_min, a_max)
        return np.log(prior_a)
    return 0

def domain_prior_binned_dens(densfunc, params):
    '''domain_prior_binned_dens:

    Prior on the domain of the density profile

    Args:
        densfunc (function): Function that returns the density profile
        params (list): List of parameters for the density profile
    
    Returns:
        domain_prior (bool): True if the parameters are in the domain of the
            density profile
    '''
    if isinstance(densfunc, pdens.NFWSpherical):
        a,A = params
        if a <= 0: return False
        if A <= 0: return False
    return True


# action distribution functions

def mloglike_fJ(*args,**kwargs):
    '''mloglike:

    Wrapper for the loglike function that returns the negative log likelihood
    '''
    return -loglike(*args,**kwargs)

def loglike_fJ():
    '''
    '''
    pass

def logprior_fJ():
    '''
    '''
    pass