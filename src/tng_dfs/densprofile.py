# ----------------------------------------------------------------------------
#
# TITLE - densprofile.py
# AUTHOR - James Lane
# PROJECT - tng-dfs
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Routines and classes for density profiles
'''
__author__ = "James Lane"

### Imports
import numpy as np
from galpy import potential
from galpy import orbit
import astropy.units as apu
import scipy.interpolate
import scipy.integrate
import scipy.special
_ro = 8.275
_vo = 220.

# ----------------------------------------------------------------------------

### Superclasses for density profiles

class DensityProfile(object):
    '''DensityProfile:

    Superclass for density profiles
    '''
    def __init__(self):
        pass


class SphericalDensityProfile(DensityProfile):
    '''SphericalDensityProfile:

    Superclass for spherical density profiles
    '''
    def __init__(self):
        super(SphericalDensityProfile, self).__init__()

# ----------------------------------------------------------------------------

### Density profiles

class TwoPowerSpherical(SphericalDensityProfile):
    '''TwoPowerSpherical:

    Two power law density profile.

    The parameters of the profile are:
        alpha: Inner power law slope
        beta: Outer power law slope
        a: Scale radius [kpc, can be an astropy quantity]
        amp: Amplitude [Msun/kpc^3, can be an astropy quantity]
    '''
    def __init__(self):
        '''__init__:
        
        Initialize the density profile.
        '''
        super(TwoPowerSpherical, self).__init__()
        self.n_params = 4
        self.param_names = ['alpha', 'beta', 'a', 'amp']

    def __call__(self, R, phi, z, params):
        '''__call__:

        Evaluate the density profile

        Args:
            r (array): Array of galactocentric spherical radii in kpc
            A (float): Amplitude of the density profile
        '''
        alpha, beta, a, amp = self._parse_params(params)
        r = np.sqrt(R**2 + z**2)
        return amp*(r/a)**(-alpha)*(1+(r/a))**(alpha-beta)
    
    def _parse_params(self, params):
        '''_parse_params:

        Parse the parameters of the density profile. Can either be a list of 
        [alpha,beta,a] or [alpha,beta,a,A].

        Args:
            params (list): List of parameters for the density profile
        
        Returns:
            params
        '''
        if len(params) == 3:
            alpha, beta, a = params
            A = 1.0
        elif len(params) == 4:
            alpha, beta, a, A = params
        else:
            raise ValueError("params must have length 3 or 4")
        if isinstance(a, apu.Quantity):
            a = a.to(apu.kpc).value
        if isinstance(A, apu.Quantity):
            A = A.to(apu.Msun/apu.kpc**3).value
        return alpha, beta, a, A
    
    def effective_volume(self, params, rmin=0., rmax=np.inf, integrate=False):
        '''effective_volume:

        Integrate the density profile over all space. This is the effective
        volume of the density profile.

        Args:
            params (list): List of parameters for the density profile
            rmin (float): Minimum radius in kpc [default: 0.]
            rmax (float): Maximum radius in kpc [default: np.inf]
        
        Returns:
            effvol (float): Effective volume of the density profile
        '''
        if isinstance(rmin, apu.Quantity):
            rmin = rmin.to(apu.kpc).value
        if isinstance(rmax, apu.Quantity):
            rmax = rmax.to(apu.kpc).value
        return self.mass(rmax, params, integrate=integrate)\
            - self.mass(rmin, params, integrate=integrate)
    
    def mass(self, r, params, integrate=False):
        '''mass:

        Calculate the enclose mass of the density profile.

        Args:
            r (array): Array of galactocentric spherical radii in kpc
            params (list): List of parameters for the density profile
        
        Returns:
            mass (array): Array of enclosed masses in Msun
        '''
        alpha, beta, a, A = self._parse_params(params)
        if integrate:
            intfunc = lambda r: r**2*self(r, 0., 0., alpha=alpha, beta=beta, 
                a=a, A=A)
            return 4*np.pi*scipy.integrate.quad(intfunc, 0, r)[0]
        else:
            return 4*np.pi*A*(r**3)*(r/a)**(-alpha)*scipy.special.hyp2f1(
                3-alpha, beta-alpha, 4-alpha, -r/a)/(3-alpha)

    
# ----------------------------------------------------------------------------

# Sampling routines

def sample_densprofile(densfunc, params=None, n=1, rmin=0., rmax=np.inf, 
    xi_scale=1.0, ro=_ro, vo=_vo):
    '''sample_densprofile:

    Draw samples from a spherical density profile

    Args:
        densfunc (function): Function that returns the density profile. Can be
            a tng_dfs.densprofile.SphericalDensityProfile or 
            galpy.potential.Potential
        n (int): Number of samples to draw
        rmin (float): Minimum radius in kpc
        rmax (float): Maximum radius in kpc
        xi_scale (float): Scale factor for xi
        params (list): List of parameters for the density profile (only 
            for tng_dfs.densprofile.SphericalDensityProfile)
    '''
    rs = _sample_r(densfunc, n=n, rmin=rmin, rmax=rmax, xi_scale=xi_scale,
        params=params)
    phis = np.random.uniform(0, 2*np.pi, size=n)
    thetas = np.arccos(np.random.uniform(-1, 1, size=n))
    Rs = rs * np.sin(thetas)
    zs = rs * np.cos(thetas)
    zarr = np.zeros(n)
    vxvv = np.array([Rs/ro, zarr, zarr, zs/ro, zarr, phis]).T
    orbs = orbit.Orbit(vxvv=vxvv, ro=ro, vo=vo)
    return orbs

def _sample_r(densfunc, params=None, n=1, rmin=0., rmax=np.inf, xi_scale=1.0):
    '''_sample_r:

    Draw radial samples

    Args:
        see sample_densprofile
    
    Returns:
        r_samples (array): Array of sampled radii in kpc (no astropy units)
    '''
    # Make the cmf(xi) interpolator
    ximin = _RToxi(rmin, a=xi_scale)
    ximax = _RToxi(rmax, a=xi_scale)
    xis = np.arange(ximin, ximax, 1e-4)
    rs = _xiToR(xis, a=xi_scale)
    ms = _mass(densfunc, rs, params=params)
    mnorm = _mass(densfunc, rmax, params=params)
    # Total mass point
    if np.isinf(rmax):
        xis = np.append(xis, 1)
        ms = np.append(ms, 1)
    # Adjust masses for minimum radius
    if rmin > 0:
        ms -= _mass(densfunc, rmin, params=params)
        mnorm -= _mass(densfunc, rmin, params=params)
    ms /= mnorm
    # Make the interpolator
    xi_cmf_interpolator = scipy.interpolate.InterpolatedUnivariateSpline(ms, 
        xis, k=3)
    # Draw random mass fractions
    rand_mass_frac = np.random.uniform(size=n)
    xi_samples = xi_cmf_interpolator(rand_mass_frac)
    r_samples = _xiToR(xi_samples, a=xi_scale)
    return r_samples

def _mass(densfunc, r, params=None):
    '''_mass:

    Calculate the mass enclosed within a radius r for a density profile

    Args:
        densfunc (function): Function that returns the density profile
        r (float): Radius in kpc
        params (list): List of parameters for the density profile

    Returns:
        mass (float): Mass enclosed within r in Msun
    '''
    if isinstance(r, apu.Quantity):
        r = r.to(apu.kpc).value
    if isinstance(densfunc, potential.Potential):
        try:
            ms = potential.mass(densfunc, r*apu.kpc, use_physical=False)
        except (ValueError, TypeError):
            ms = np.array([potential.mass(densfunc, _r*apu.kpc, 
                                          use_physical=False) for _r in r])
    else:
        pass
    return ms

def _xiToR(xi, a=1):
    return a * np.divide((1.0 + xi), (1.0 - xi))

def _RToxi(r, a=1):
    out = np.divide((r / a - 1.0), (r / a + 1.0), where=True ^ np.isinf(r))
    if np.any(np.isinf(r)):
        if hasattr(r, "__len__"):
            out[np.isinf(r)] = 1.0
        else:
            return 1.0
    return out