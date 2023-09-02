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

# ----------------------------------------------------------------------------

# Classes to hold information about fitting

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
    effvol_params=None, parts=False):
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
            usr_log_prior(params, densfunc)
    '''
    # Evaluate the domain prior
    if not domain_prior_dens(densfunc, params):
        return -np.inf
    # Evaluate the prior on the density profile
    logprior = logprior_dens(densfunc, params)
    # Evaluate any user supplied prior
    if callable(usr_log_prior):
        usrlogprior = usr_log_prior(params, densfunc)
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
    # Evaluate the effective volume
    effvol = densfunc.effective_volume(params, *effvol_params)
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
    '''
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
    '''
    if isinstance(densfunc, pdens.TwoPowerSpherical):
        alpha,beta,a,A = params
        if alpha <= 0: return False
        if alpha >= beta: return False
        if a <= 0: return False
        if A <= 0: return False
    return True

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
    logdens = np.log(densfunc(r, 0., 0., params))
    # Evaluate the log likelihood
    logdiffsum = np.sum(np.square(logdens-np.log(rho))) 
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
    if isinstance(densfunc, pdens.NFWProfile):
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
    if isinstance(densfunc, pdens.NFWProfile):
        a,A = params
        if a <= 0: return False
        if A <= 0: return False


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