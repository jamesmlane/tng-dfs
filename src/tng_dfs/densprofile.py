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
import warnings
from galpy import potential
from galpy import orbit
import astropy.units as apu
import astropy.cosmology
import astropy.constants
import scipy.interpolate
import scipy.integrate
import scipy.special
from . import util as putil

_HUBBLE_PARAM = 0.6774 # Planck 2015 value
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

    def _parse_R_phi_z_input(self, R, phi, z):
        '''_parse_R_phi_z_input:

        Parse input R, phi, z

        Args:
            R, phi, z (array): Arrays of galactocentric cylindrical radius, 
                azimuth, and height above the plane. Can be astropy quantities.
        
        Returns:
            R, phi, z (array): Same as input, but in kpc, radians, and kpc
        '''
        if isinstance(R, apu.Quantity):
            R = R.to(apu.kpc).value
        if isinstance(phi, apu.Quantity):
            phi = phi.to(apu.rad).value
        if isinstance(z, apu.Quantity):
            z = z.to(apu.kpc).value
        return R, phi, z

class SphericalDensityProfile(DensityProfile):
    '''SphericalDensityProfile:

    Superclass for spherical density profiles
    '''
    def __init__(self):
        super(SphericalDensityProfile, self).__init__()
    
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
        
        vol = self.mass(rmax, params, integrate=integrate)
        if rmin > 0.:
            vol -= self.mass(rmin, params, integrate=integrate)
        return vol

    def rforce(self, r, params):
        '''rforce:

        Calculate the radial force of the density profile.

        Args:
            r (array): Array of galactocentric spherical radii in kpc
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            rforce (array): Array of spherical radial force in km/s/Myr
        '''
        if isinstance(r, apu.Quantity):
            r = r.to(apu.kpc).value
        _mr2 = -self.mass(r, params, integrate=False)/r**2
        _mr2 *= (apu.Msun/apu.kpc**2)
        _rforce = _mr2*astropy.constants.G
        return _rforce.to(apu.km/apu.s/apu.Myr)

class AxisymmetricDensityProfile(DensityProfile):
    '''AxisymmetricDensityProfile:

    Superclass for axisymmetric density profiles
    '''
    def __init__(self):
        super(AxisymmetricDensityProfile, self).__init__()
    
    def effective_volume(self, params, rmin=0., rmax=np.inf, zmax=np.inf, 
        integrate=False):
        '''effective_volume:

        Integrate the density profile over all space. This is the effective
        volume of the density profile.

        Args:
            params (list): List of parameters for the density profile
            rmin (float): Minimum radius in kpc [default: 0.]
            rmax (float): Maximum radius in kpc [default: np.inf]
            zmax (float): Maximum height above the plane in kpc [default: np.inf]
        
        Returns:
            effvol (float): Effective volume of the density profile
        '''
        if isinstance(rmin, apu.Quantity):
            rmin = rmin.to(apu.kpc).value
        if isinstance(rmax, apu.Quantity):
            rmax = rmax.to(apu.kpc).value
        if isinstance(zmax, apu.Quantity):
            zmax = zmax.to(apu.kpc).value
        vol = self.mass(rmax, params, zmax=zmax, integrate=integrate)
        if rmin > 0.:
            vol -= self.mass(rmin, params, zmax=zmax, integrate=integrate)
        return vol

# ----------------------------------------------------------------------------

### Spherical density profiles

# Two Power Spherical

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
            R, phi, z (array): Arrays of galactocentric cylindrical radius, 
                azimuth, and height above the plane. Can be astropy quantities.
            params (list): List of parameters for the density profile, see
                class docstring.
            
        Returns:
            dens (array): Array of densities in Msun/kpc^3
        '''
        R, phi, z = self._parse_R_phi_z_input(R, phi, z)
        r = np.sqrt(R**2 + z**2)
        alpha, beta, a, amp = self._parse_params(params)
        return amp*(r/a)**(-alpha)*(1+(r/a))**(alpha-beta)
    
    def _parse_params(self, params):
        '''_parse_params:

        Parse the parameters of the density profile. Can either be a list of 
        [alpha,beta,a] or [alpha,beta,a,A].

        Args:
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            params (list): List of parameters for the density profile, see
                class docstring.
        '''
        if len(params) == 3:
            alpha, beta, a = params
            amp = 1.0
        elif len(params) == 4:
            alpha, beta, a, amp = params
        else:
            raise ValueError("params must have length 3 or 4")
        if isinstance(a, apu.Quantity):
            a = a.to(apu.kpc).value
        if isinstance(amp, apu.Quantity):
            amp = amp.to(apu.Msun/apu.kpc**3).value
        return alpha, beta, a, amp
    
    def mass(self, r, params, integrate=False):
        '''mass:

        Calculate the enclose mass of the density profile.

        Args:
            r (array): Array of galactocentric spherical radii in kpc, 
                can be astropy quantity.
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            mass (array): Array of enclosed masses in Msun
        '''
        if isinstance(r,apu.Quantity):
            r = r.to(apu.kpc).value
        alpha, beta, a, amp = self._parse_params(params)
        if integrate:
            intfunc = lambda r: r**2*self(r, 0., 0., params=params)
            return 4*np.pi*scipy.integrate.quad(intfunc, 0, r)[0]
        else:
            return 4*np.pi*amp*(r**3)*(r/a)**(-alpha)*scipy.special.hyp2f1(
                3-alpha, beta-alpha, 4-alpha, -r/a)/(3-alpha)
    
    def densfunc_to_pot(self, params, map_method='mass_at_a', densfunc=None,
        ro=_ro, vo=_vo, validate=True):
        '''densfunc_to_pot:

        Convert a pdens.TwoPowerSpherical instance to a 
        potential.TwoPowerSphericalPotential based on supplied params

        Args:
            params (list): List of parameters for the density profile, see
                class docstring.
            map_method (string): String representing the method used to 
                convert to galpy potential, see above [default 'mass_at_a']
            densfunc (callable): pdens.DensityProfile instance to convert
                [default pdens.TwoPowerSpherical()]
            ro, vo (floats): galpy distance and velocity scales [default 
                8.275, 220.]
            validate (bool): Check that the galpy potential and pdens 
                densfunc give the same enclosed mass, density, radial force.

        Returns:
            pot (potential.TwoPowerSphericalPotential) - output galpy potential
        '''
        if densfunc is None:
            densfunc = TwoPowerSpherical()
        alpha, beta, a, amp = densfunc._parse_params(params)

        if map_method == 'mass_at_a':
            _pot = potential.TwoPowerSphericalPotential(amp=1., a=a*apu.kpc, 
                alpha=alpha, beta=beta, ro=ro, vo=vo)
            _dmass = densfunc.mass(a*apu.kpc, params=params)
            _pmass = _pot.mass(a*apu.kpc, use_physical=True).to(apu.Msun).value
            _amp = _dmass/_pmass
            pot = potential.TwoPowerSphericalPotential(amp=_amp, a=a*apu.kpc, 
                alpha=alpha, beta=beta, ro=ro, vo=vo)
        
        if validate:
            tol = 1e-8
            rs = np.logspace(-1, 2, num=30)*apu.kpc
            # Mass
            gmass = pot.mass(rs).to(apu.Msun).value
            dmass = densfunc.mass(rs, params)
            assert np.all(np.abs((gmass-dmass)/dmass) < tol)
            gdens = pot.dens(rs, 0.).to(apu.Msun/apu.kpc**3).value
            ddens = densfunc(rs, 0., 0., params)
            assert np.all(np.abs((gdens-ddens)/ddens) < tol)
            gforce = pot.rforce(rs, 0.).to(apu.km/apu.s/apu.Myr).value
            dforce = densfunc.rforce(rs, params).value
            assert np.all(np.abs((gforce-dforce)/dforce) < tol)
            
        return pot
    
# NFW Spherical

class NFWSpherical(SphericalDensityProfile):
    '''NFWSpherical:

    Two power law density profile with alpha=1 and beta=3

    The parameters of the profile are:
        a: Scale radius [kpc, can be an astropy quantity]
        amp: Amplitude [Msun/kpc^3, can be an astropy quantity]
    '''
    def __init__(self):
        '''__init__:
        
        Initialize the density profile.
        '''
        super(NFWSpherical, self).__init__()
        self.n_params = 2
        self.param_names = ['a', 'amp']

    def __call__(self, R, phi, z, params):
        '''__call__:

        Evaluate the density profile

        Args:
            R, phi, z (array): Arrays of galactocentric cylindrical radius, 
                azimuth, and height above the plane. Can be astropy quantities.
            params (list): List of parameters for the density profile, see
                class docstring.
        '''
        R, phi, z = self._parse_R_phi_z_input(R, phi, z)
        r = np.sqrt(R**2 + z**2)
        a, amp = self._parse_params(params)
        alpha, beta = 1, 3
        return amp*(r/a)**(-alpha)*(1+(r/a))**(alpha-beta)
    
    def _parse_params(self, params):
        '''_parse_params:

        Parse the parameters of the density profile. Can either be a list of 
        [a,] or [a,A].

        Args:
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            params (list): List of parameters for the density profile, see
                class docstring.
        '''
        if len(params) == 1:
            a, = params
            amp = 1.0
        elif len(params) == 2:
            a, amp = params
        else:
            raise ValueError("params must have length 1 or 2")
        if isinstance(a, apu.Quantity):
            a = a.to(apu.kpc).value
        if isinstance(amp, apu.Quantity):
            amp = amp.to(apu.Msun/apu.kpc**3).value
        return a, amp
    
    def mass(self, r, params, integrate=False):
        '''mass:

        Calculate the enclose mass of the density profile.

        Args:
            r (array): Array of galactocentric spherical radii in kpc, 
                can be astropy quantity.
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            mass (array): Array of enclosed masses in Msun
        '''
        if isinstance(r,apu.Quantity):
            r = r.to(apu.kpc).value
        if integrate:
            intfunc = lambda r: r**2*self(r, 0., 0., params)
            return 4*np.pi*scipy.integrate.quad(intfunc, 0, r)[0]
        else:
            a, amp = self._parse_params(params)
            return 4*np.pi*amp*(a**3)*(np.log(1+r/a)-(r/a)/(1+r/a))
        
    def densfunc_to_pot(self, params, map_method='mass_at_a', densfunc=None,
        ro=_ro, vo=_vo, validate=True):
        '''densfunc_to_pot:

        Convert a pdens.NFWSpherical instance to a galpy potential.NFWPotential 
        instance based on supplied params. map_method options are:
        - 'mass_at_a': Determine the amplitude of the galpy potential by 
            comparing the mass enclosed at the scale radius, a.

        Args:
            params (list): List of parameters for the density profile, see
                class docstring.
            map_method (string): String representing the method used to 
                convert to galpy potential, see above [default 'mass_at_a']
            densfunc (callable): pdens.DensityProfile instance to convert
                [default pdens.NFWSpherical()]
            ro, vo (floats): galpy distance and velocity scales [default 
                8.275, 220.]
            validate (bool): Check that the galpy potential and pdens 
                densfunc give the same enclosed mass, density, radial force.

        Returns:
            pot (potential.NFWPotential): galpy NFW potential instance
        '''
        if densfunc is None:
            densfunc = NFWSpherical()
        a, amp = densfunc._parse_params(params)

        if map_method == 'mass_at_a':
            _pot = potential.NFWPotential(amp=1., a=a*apu.kpc, ro=ro, vo=vo)
            _dmass = densfunc.mass(a*apu.kpc, params=params)
            _pmass = _pot.mass(a*apu.kpc, use_physical=True).to(apu.Msun).value
            _amp = _dmass/_pmass
            pot = potential.NFWPotential(amp=_amp, a=a*apu.kpc, ro=ro, vo=vo)
        
        if validate:
            tol = 1e-8
            rs = np.logspace(-1, 2, num=30)*apu.kpc
            # Mass
            gmass = pot.mass(rs).to(apu.Msun).value
            dmass = densfunc.mass(rs, params)
            assert np.all(np.abs((gmass-dmass)/dmass) < tol)
            gdens = pot.dens(rs, 0.).to(apu.Msun/apu.kpc**3).value
            ddens = densfunc(rs, 0., 0., params)
            assert np.all(np.abs((gdens-ddens)/ddens) < tol)
            gforce = pot.rforce(rs, 0.).to(apu.km/apu.s/apu.Myr).value
            dforce = densfunc.rforce(rs, params).value
            assert np.all(np.abs((gforce-dforce)/dforce) < tol)
        
        return pot

# Broken Power Law Spherical

class BrokenPowerLawSpherical(SphericalDensityProfile):
    '''BrokenPowerLawSpherical:

    Broken power law density profile

    The parameters of the profile are:
        alpha1: Inner power law slope
        alpha2: Outer power law slope
        a1: Break radius [kpc, can be an astropy quantity]
        amp: Amplitude [Msun/kpc^3, can be an astropy quantity]
    '''
    def __init__(self):
        '''__init__:
        
        Initialize the density profile.
        '''
        super(BrokenPowerLawSpherical, self).__init__()
        self.n_params = 4
        self.param_names = ['alpha1', 'alpha2', 'r1', 'amp']

    def __call__(self, R, phi, z, params):
        '''__call__:

        Evaluate the density profile

        Args:
            R, phi, z (array): Arrays of galactocentric cylindrical radius, 
                azimuth, and height above the plane. Can be astropy quantities.
            params (list): List of parameters for the density profile, see
                class docstring.
        '''
        R, phi, z = self._parse_R_phi_z_input(R, phi, z)
        r = np.sqrt(R**2 + z**2)
        alpha1, alpha2, r1, amp = self._parse_params(params)
        inner_mask = r <= r1
        outer_mask = r > r1
        r1norm = r1**(alpha2-alpha1)
        dens = np.zeros_like(R)
        dens[inner_mask] = r[inner_mask]**(-alpha1)
        dens[outer_mask] = r1norm*r[outer_mask]**(-alpha2)
        return amp*dens
    
    def _parse_params(self, params):
        '''_parse_params:

        Parse the parameters of the density profile. Can either be a list of 
        [alpha1,alpha2,r1] or [alpha1,alpha2,r1,amp].

        Args:
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            params (list): List of parameters for the density profile, see
                class docstring.
        '''
        if len(params) == 3:
            alpha1, alpha2, r1 = params
            amp = 1.0
        elif len(params) == 4:
            alpha1, alpha2, r1, amp = params
        else:
            raise ValueError("params must have length 3 or 4")
        if isinstance(r1, apu.Quantity):
            r1 = r1.to(apu.kpc).value
        if isinstance(amp, apu.Quantity):
            amp = amp.to(apu.Msun/apu.kpc**3).value
        return alpha1, alpha2, r1, amp
    
    def mass(self, r, params, integrate=False):
        '''mass:

        Calculate the enclose mass of the density profile.

        Args:
            r (array): Array of galactocentric spherical radii in kpc, 
                can be astropy quantity.
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            mass (array): Array of enclosed masses in Msun
        '''
        if isinstance(r,apu.Quantity):
            r = r.to(apu.kpc).value
        alpha1, alpha2, r1, amp = self._parse_params(params)
        if integrate:
            intfunc = lambda r: r**2*self(r, 0., 0., params)
            return 4*np.pi*scipy.integrate.quad(intfunc, 0, r, points=(0.,r1))[0]
        else:
            r1norm = r1**(alpha2-alpha1)
            if r <= r1:
                _int = r**(3-alpha1)/(3-alpha1)
            else:
                _int = r1**(3-alpha1)/(3-alpha1)
                _int += r1norm*r**(3-alpha2)/(3-alpha2)
                _int -= r1norm*r1**(3-alpha2)/(3-alpha2)
            return 4*np.pi*amp*_int

# Double broken power law

class DoubleBrokenPowerLawSpherical(SphericalDensityProfile):
    '''DoubleBrokenPowerLawSpherical:

    Double broken power law density profile

    The parameters of the profile are:
        alpha1: Inner power law slope
        alpha2: Middle power law slope
        alpha3: Outer power law slope
        r1: Inner break radius [kpc, can be an astropy quantity]
        r2: Outer break radius [kpc, can be an astropy quantity]
        amp: Amplitude [Msun/kpc^3, can be an astropy quantity]
    '''
    def __init__(self):
        '''__init__:
        
        Initialize the density profile.
        '''
        super(DoubleBrokenPowerLawSpherical, self).__init__()
        self.n_params = 6
        self.param_names = ['alpha1', 'alpha2', 'alpha3', 'r1', 'r2', 'amp']

    def __call__(self, R, phi, z, params):
        '''__call__:

        Evaluate the density profile

        Args:
            R, phi, z (array): Arrays of galactocentric cylindrical radius, 
                azimuth, and height above the plane. Can be astropy quantities.
            params (list): List of parameters for the density profile, see
                class docstring.
        '''
        R, phi, z = self._parse_R_phi_z_input(R, phi, z)
        r = np.sqrt(R**2 + z**2)
        alpha1, alpha2, alpha3, r1, r2, amp = self._parse_params(params)
        inner_mask = r <= r1
        middle_mask = (r > r1) & (r <= r2)
        outer_mask = r > r2
        r1norm = r1**(alpha2-alpha1)
        r2norm = r2**(alpha3-alpha2)
        dens = np.zeros_like(R)
        dens[inner_mask] = r[inner_mask]**(-alpha1)
        dens[middle_mask] = r1norm*r[middle_mask]**(-alpha2)
        dens[outer_mask] = r1norm*r2norm*(r[outer_mask])**(-alpha3)
        return amp*dens
    
    def _parse_params(self, params):
        '''_parse_params:

        Parse the parameters of the density profile. Can either be a list of 
        [alpha1,alpha2,alpha3,r1,r2] or [alpha1,alpha2,alph3,r1,r2,amp].

        Args:
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            params (list): List of parameters for the density profile, see
                class docstring.
        '''
        if len(params) == 5:
            alpha1, alpha2, alpha3, r1, r2 = params
            amp = 1.0
        elif len(params) == 6:
            alpha1, alpha2, alpha3, r1, r2, amp = params
        else:
            raise ValueError("params must have length 5 or 6")
        if isinstance(r1, apu.Quantity):
            r1 = r1.to(apu.kpc).value
        if isinstance(amp, apu.Quantity):
            amp = amp.to(apu.Msun/apu.kpc**3).value
        return alpha1, alpha2, r1, amp
    
    def mass(self, r, params, integrate=False):
        '''mass:

        Calculate the enclose mass of the density profile.

        Args:
            r (array): Array of galactocentric spherical radii in kpc, 
                can be astropy quantity.
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            mass (array): Array of enclosed masses in Msun
        '''
        if isinstance(r, apu.Quantity):
            r = r.to(apu.kpc).value
        alpha1, alpha2, alpha3, r1, r2, amp = self._parse_params(params)
        if integrate:
            intfunc = lambda r: r**2*self(r, 0., 0., params)
            return 4*np.pi*scipy.integrate.quad(intfunc, 0, r, 
                points=(0.,r1, r2))[0]
        else:
            r1norm = r1**(alpha2-alpha1)
            r2norm = r2**(alpha3-alpha2)
            if r <= r1:
                _int = r**(3-alpha1)/(3-alpha1)
            elif r <= r2:
                _int = r1**(3-alpha1)/(3-alpha1)
                _int += r1norm*r**(3-alpha2)/(3-alpha2)
                _int -= r1norm*r1**(3-alpha2)/(3-alpha2)
            else:
                _int = r1**(3-alpha1)/(3-alpha1)
                _int += r1norm*r2**(3-alpha2)/(3-alpha2)
                _int -= r1norm*r1**(3-alpha2)/(3-alpha2)
                _int += r1norm*r2norm*r**(3-alpha3)/(3-alpha3)
                _int -= r1norm*r2norm*r2**(3-alpha3)/(3-alpha3)
            return 4*np.pi*amp*_int

class SinglePowerCutoffSpherical(SphericalDensityProfile):
    '''SinglePowerCutoffSpherical:

    Single power law density profile with an exponential cutoff.

    The parameters of the profile are:
        alpha: Power law index
        rc: Exponential scale length [kpc, can be an astropy quantity]
        amp: Amplitude [Msun/kpc^3, can be an astropy quantity]
    '''
    def __init__(self):
        '''__init__:

        Initialize the density profile.
        '''
        super(SinglePowerCutoffSpherical, self).__init__()
        self.n_params = 3
        self.param_names = ['alpha', 'rc', 'amp']

    def __call__(self, R, phi, z, params):
        '''__call__:

        Evaluate the density profile

        Args:
            R, phi, z (array): Arrays of galactocentric cylindrical radius, 
                azimuth, and height above the plane. Can be astropy quantities.
            params (list): List of parameters for the density profile, see
                class docstring.
            
        Returns:
            dens (array): Array of densities in Msun/kpc^3
        '''
        R, phi, z = self._parse_R_phi_z_input(R, phi, z)
        r = np.sqrt(R**2 + z**2)
        alpha, rc, amp = self._parse_params(params)
        return amp * (r ** (-alpha)) * np.exp(-(r / rc)**2)

    def _parse_params(self, params):
        '''_parse_params:

        Parse the parameters of the density profile.

        Args:
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            params (list): List of parameters for the density profile, see
                class docstring.
        '''
        if len(params) == 2:
            alpha, rc = params
            amp = 1.
        if len(params) == 3:
            alpha, rc, amp = params
        else:
            raise ValueError("params must have length 3")
        if isinstance(rc, apu.Quantity):
            rc = rc.to(apu.kpc).value
        if isinstance(amp, apu.Quantity):
            amp = amp.to(apu.Msun / apu.kpc ** 3).value
        return alpha, rc, amp

    def mass(self, r, params, integrate=False, _use_hypergeometric=True):
        '''mass:

        Calculate the enclosed mass of the density profile.

        Args:
            r (array): Array of galactocentric spherical radii in kpc, 
                can be astropy quantity.
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            mass (array): Array of enclosed masses in Msun
        '''
        # _use_hypergeometric = False # Better behaviour at alpha > 3
        if isinstance(r, apu.Quantity):
            r = r.to(apu.kpc).value
        _r_isarray = True
        try:
            len(r)
        except TypeError:
            r = float(r)
            r = np.atleast_1d(r)
            _r_isarray = False
        alpha, rc, amp = self._parse_params(params)
        if integrate:
            intfunc = lambda R: R ** 2 * self(R, 0., 0., params=params)
            return 4 * np.pi * scipy.integrate.quad(intfunc, 0, r)[0]
        else:
            if _use_hypergeometric:
                out = np.ones_like(r, dtype=float)
                mask = np.isinf(r)
                out[~mask] = (
                    2.0*np.pi*r[~mask]**(3.0-alpha)/(1.5-alpha/2.0)\
                    *scipy.special.hyp1f1(1.5-alpha/2.0, 2.5-alpha/2.0,
                        -((r[~mask]/rc)**2.0)))
                out[mask] = (2.0*np.pi*rc**(3.0-alpha)\
                    *scipy.special.gamma(1.5-alpha / 2.0))
                # For (r/rc) in the exponential rather than (r/rc)**2
                # out[~mask] = 4.0*np.pi * (r[~mask]**(3.0-alpha)) * \
                #     scipy.special.hyp1f1(3-alpha, 4-alpha, -(r[~mask]/rc)) / \
                #     (3-alpha)
                # out[mask] = 4.0*np.pi * (rc**(3.0-alpha)) * \
                #     scipy.special.gamma(3-alpha)
                if _r_isarray:
                    return amp*out
                else:
                    return amp*out[0]
            else:
                out = 2 * np.pi * (rc ** (3-alpha)) * \
                       scipy.special.gamma(1.5 - alpha/2.) * \
                       scipy.special.gammainc(1.5 - alpha/2., (r / rc)**2.)
                # For (r/rc) in the exponential rather than (r/rc)**2
                # out = 4 * np.pi * (rc ** (3-alpha)) * \
                #        scipy.special.gamma(3 - alpha) * \
                #        scipy.special.gammainc(3 - alpha, r / rc)
                if _r_isarray:
                    return amp*out
                else:
                    return amp*out[0]
    
    def densfunc_to_pot(self, params, map_method='mass_at_rc', densfunc=None,
        ro=_ro, vo=_vo, validate=True):
        '''densfunc_to_pot:

        Convert a pdens.SinglePowerCutoffSpherical instance to a 
        potential.PowerSphericalPotentialwCutoff based on supplied params.
        map_method options are:
        - 'mass_at_rc': Determine the amplitude of the galpy potential by 
            comparing the mass enclosed at the exponential scale radius, rc.

        Args:
            params (list): List of parameters for the density profile, see
                class docstring.
            map_method (string): String representing the method used to 
                convert to galpy potential, see above [default 'mass_at_rc']
            densfunc (callable): pdens.DensityProfile instance to convert
                [default pdens.TwoPowerSpherical()]
            ro, vo (floats): galpy distance and velocity scales [default 
                8.275, 220.]
            validate (bool): Check that the galpy potential and pdens 
                densfunc give the same enclosed mass, density, radial force.

        Returns:
            pot (potential.PowerSphericalPotentialwCutoff) - output galpy potential
        '''
        if densfunc is None:
            densfunc = SinglePowerCutoffSpherical()
        alpha, rc, amp = densfunc._parse_params(params)

        if map_method == 'mass_at_rc':
            _pot = potential.PowerSphericalPotentialwCutoff(amp=1., alpha=alpha, 
                rc=rc*apu.kpc, ro=ro, vo=vo)
            _dmass = densfunc.mass(rc*apu.kpc, params=params)
            _pmass = _pot.mass(rc*apu.kpc, use_physical=True).to(apu.Msun).value
            _amp = _dmass/_pmass
            pot = potential.PowerSphericalPotentialwCutoff(amp=_amp, 
                alpha=alpha, rc=rc*apu.kpc, ro=ro, vo=vo)
        
        if validate:
            tol = 1e-8
            rs = np.logspace(-1, 2, num=30)*apu.kpc
            # Mass
            gmass = pot.mass(rs).to(apu.Msun).value
            dmass = densfunc.mass(rs, params)
            assert np.all(np.abs((gmass-dmass)/dmass) < tol)
            gdens = pot.dens(rs, 0.).to(apu.Msun/apu.kpc**3).value
            ddens = densfunc(rs, 0., 0., params)
            assert np.all(np.abs((gdens-ddens)/ddens) < tol)
            gforce = pot.rforce(rs, 0.).to(apu.km/apu.s/apu.Myr).value
            dforce = densfunc.rforce(rs, params).value
            assert np.all(np.abs((gforce-dforce)/dforce) < tol)
        
        return pot

# ----------------------------------------------------------------------------

### Axisymmetric density profiles

# Double exponential disk

class DoubleExponentialDisk(AxisymmetricDensityProfile):
    '''DoubleExponentialDisk:

    Double exponential disk density profile

    The parameters of the profile are:
        hr: Scale radius [kpc, can be an astropy quantity]
        hz: Scale height [kpc, can be an astropy quantity]
        amp: Amplitude [Msun/kpc^3, can be an astropy quantity]
    '''

    def __init__(self, ):
        '''__init__:

        Initialize the density profile.
        '''
        super(DoubleExponentialDisk, self).__init__()
        self.n_params = 3
        self.param_names = ['hr', 'hz', 'amp']
    
    def __call__(self, R, phi, z, params):
        '''__call__:

        Evaluate the density profile

        Args:
            R, phi, z (array): Arrays of galactocentric cylindrical radius, 
                azimuth, and height above the x-y plane. Can be astropy quantities.
            params (list): List of parameters for the density profile, see
                class docstring.
            
        Returns:
            dens (array): Array of densities in Msun/kpc^3
        '''
        R, phi, z = self._parse_R_phi_z_input(R, phi, z)
        hr, hz, amp = self._parse_params(params)
        return amp*np.exp(-R/hr)*np.exp(-np.abs(z)/hz)
    
    def _parse_params(self, params):
        '''_parse_params:

        Parse the parameters of the density profile.

        Args:
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            params (list): List of parameters for the density profile, see
                class docstring.
        '''
        if len(params) == 2:
            hr, hz = params
            amp = 1.
        elif len(params) == 3:
            hr, hz, amp = params
        else:
            raise ValueError("params must have length 2 or 3")
        if isinstance(hr, apu.Quantity):
            hr = hr.to(apu.kpc).value
        if isinstance(hz, apu.Quantity):
            hz = hz.to(apu.kpc).value
        if isinstance(amp, apu.Quantity):
            amp = amp.to(apu.Msun/apu.kpc**3).value
        return hr, hz, amp
    
    def mass(self, R, params, z=np.inf, integrate=False):
        '''mass:

        Calculate the enclosed mass of the density profile. if z is not np.inf 
        then the slab mass is calculated, otherwise the cylindrical mass is 
        calculated.

        Args:
            R (array): Array of galactocentric cylindrical radii in kpc, 
                can be astropy quantity.
            params (list): List of parameters for the density profile, see
                class docstring.
            z (float): Height above the x-y plane in kpc, used to calculate
        
        Returns:
            mass (array): Array of enclosed masses in Msun
        '''
        if isinstance(R, apu.Quantity):
            R = R.to(apu.kpc).value
        if isinstance(z, apu.Quantity):
            z = z.to(apu.kpc).value
        hr, hz, amp = self._parse_params(params)
        if integrate:
            raise NotImplementedError("Integration not implemented")
            # intfunc = lambda r: r*self(r, 0., 0., params=params)
            # return 2*np.pi*scipy.integrate.quad(intfunc, 0, R)[0]
        else:
            return 4*np.pi*amp*hr*hz*(hr-np.exp(-R/hr)*(hr+R))*(1-np.exp(-np.abs(z)/hz))

# Miyamoto-Nagai

class MiyamotoNagai(AxisymmetricDensityProfile):
    '''MiyamotoNagai:

    Miyamoto-Nagai disk density profile

    The parameters of the profile are:
        a: Scale radius [kpc, can be an astropy quantity]
        b: Scale height [kpc, can be an astropy quantity]
        amp: Amplitude [Msun/kpc^3, can be an astropy quantity]
    '''

    def __init__(self, ):
        '''__init__:

        Initialize the density profile.
        '''
        super(MiyamotoNagai, self).__init__()
        self.n_params = 3
        self.param_names = ['a', 'b', 'amp']
    
    def __call__(self, R, phi, z, params):
        '''__call__:

        Evaluate the density profile

        Args:
            R, phi, z (array): Arrays of galactocentric cylindrical radius, 
                azimuth, and height above the x-y plane. Can be astropy quantities.
            params (list): List of parameters for the density profile, see
                class docstring.
            
        Returns:
            dens (array): Array of densities in Msun/kpc^3
        '''
        R, phi, z = self._parse_R_phi_z_input(R, phi, z)
        a, b, amp = self._parse_params(params)
        sqrtbz = np.sqrt(b**2.0 + z**2.0)
        asqrtbz = a + sqrtbz
        term1 = amp*b**2/(4*np.pi)
        term2 = a*R**2 + (a + 3*sqrtbz)*asqrtbz**2
        term3 = (R**2 + asqrtbz**2)**2.5*sqrtbz**3
        return term1*term2/term3

    def _parse_params(self, params):
        '''_parse_params:

        Parse the parameters of the density profile.

        Args:
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            params (list): List of parameters for the density profile, see
                class docstring.
        '''
        if len(params) == 2:
            a, b = params
            amp = 1.
        elif len(params) == 3:
            a, b, amp = params
        else:
            raise ValueError("params must have length 2 or 3")
        if isinstance(a, apu.Quantity):
            a = a.to(apu.kpc).value
        if isinstance(b, apu.Quantity):
            b = b.to(apu.kpc).value
        if isinstance(amp, apu.Quantity):
            amp = amp.to(apu.Msun).value
        return a, b, amp

    def Rforce(self, R, z, params):
        '''zforce:

        Calculate the Radial force of the density profile.

        Args:
            R, z (array): Arrays of galactocentric cylindrical radius 
                and height above the x-y plane. Can be astropy quantities.
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            Rforce (array): List of cylindrical radial forces in km/s/Myr. 
                Assuming amp in Msun, 
        '''
        R, _, z = self._parse_R_phi_z_input(R, 0., z)
        a, b, amp = self._parse_params(params)
        sqrtbz = np.sqrt(b**2.0 + z**2.0)
        asqrtbz = a + sqrtbz
        _mr2 = -amp*(R)/((asqrtbz**2 + R**2)**1.5)
        _mr2 *= (apu.Msun/apu.kpc**2)
        _Rforce = _mr2*astropy.constants.G
        return _Rforce.to(apu.km/apu.s/apu.Myr)

    def zforce(self, R, z, params):
        '''zforce:

        Calculate the vertical force of the density profile.

        Args:
            R, z (array): Arrays of galactocentric cylindrical radius 
                and height above the x-y plane. Can be astropy quantities.
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            zforce (array): List of cylindrical vertical forces in km/s/Myr
        '''
        R, _, z = self._parse_R_phi_z_input(R, 0., z)
        a, b, amp = self._parse_params(params)
        sqrtbz = np.sqrt(b**2.0 + z**2.0)
        asqrtbz = a + sqrtbz
        _mr2 = -amp*(z*asqrtbz)/(sqrtbz*(asqrtbz**2 + R**2)**1.5)
        _mr2 *= (apu.Msun/apu.kpc**2)
        _zforce = _mr2*astropy.constants.G
        return _zforce.to(apu.km/apu.s/apu.Myr)

    def mass(self, R, params, z=np.inf, integrate=False):
        '''mass:
        
        Calculate the enclosed mass of the density profile. z can be optionally 
        provided to calculate the slab mass, otherwise the mass is calculated 
        as if z is integrated from -inf to inf.
        
        Args:
            R (array): Array of galactocentric cylindrical radii in kpc, 
                can be astropy quantity.
            params (list): List of parameters for the density profile, see
                class docstring.
            z (array): Array of heights above the x-y plane in kpc, if None 
                then integrate from -inf to inf. Can be astropy quantity 
                [default: None] 
        
        Returns:
            mass (array) - Enclosed mass in Msun. 
        '''

        if isinstance(R, apu.Quantity):
            R = R.to(apu.kpc).value
        if isinstance(z, apu.Quantity):
            z = z.to(apu.kpc).value
        a, b, amp = self._parse_params(params)
        if integrate:
            if isinstance(R, (np.ndarray,list)):
                out = np.zeros_like(R)
                if not isinstance(z, (np.ndarray,list)):
                    z = np.ones_like(R)*z
                for i in range(len(R)):
                    out[i] = self.mass(R[i], params, z=z[i], integrate=True)
                return out
            if z is None:  # Within spherical shell
                raise NotImplementedError
                # def _integrand(theta):
                #     tz = R * np.cos(theta)
                #     tR = R * np.sin(theta)
                #     return self.rforce(tR, tz)*np.sin(theta)
                # return -(R**2.0)*scipy.integrate.quad(_integrand, 0.0, np.pi)[0]/2.0
            else:  # Within disk at <R, -z --> z
                if np.isinf(z):
                    _z = 10000*b
                else:
                    _z = z
                Rfunc = lambda x: self.Rforce(R, x, params=params)
                Rterm = -R*scipy.integrate.quad(Rfunc, -_z, _z)[0]/2.0
                zfunc = lambda x: self.zforce(x, _z, params=params)
                zterm = scipy.integrate.quad(zfunc, 0.0, R)[0]
                return Rterm + zterm
        else:
            raise NotImplementedError("mass not implemented for non-integrated case")

# ----------------------------------------------------------------------------

### Composite density profiles

class CompositeDensityProfile(DensityProfile):
    '''CompositeDensityProfile:

    Class to contain and handle more than one density profile.

    Args:
        densprofiles (list): List of density profiles to be contained in the 
            composite density profile. Should be instance of 
            DensityProfile() or one of its subclasses.
    '''

    def __init__(self, densprofiles):
        '''__init__:

        Initialize the composite density profile.
        '''
        super(CompositeDensityProfile, self).__init__()
        self.densprofiles = densprofiles
        self.n_densprofiles = len(densprofiles)
        self.n_params = np.sum([dens.n_params for dens in densprofiles])
        self.param_names = []
        for dens in densprofiles:
            self.param_names += dens.param_names
    
    def __call__(self, R, phi, z, params):
        '''__call__:
        
        Evaluate the density profile

        Args:
            R, phi, z (array): Arrays of galactocentric cylindrical radius, 
                azimuth, and height above the x-y plane. Can be astropy 
                quantities.
            params (list): List of parameters for the density profile, see
                class docstring.
        
        Returns:
            dens (array): Array of densities in Msun/kpc^3
        '''
        R, phi, z = self._parse_R_phi_z_input(R, phi, z)
        dens = np.zeros_like(R)
        for i, densprofile in enumerate(self.densprofiles):
            n_params = densprofile.n_params
            dens += densprofile(R, phi, z, params[i*n_params:(i+1)*n_params])
        return dens
    
    # def _parse_params(self, params):
    #     '''_parse_params:
        
    #     Parse the parameters of the density profile.
        
    #     Args:
    #         params (list): List of parameters for the density profile, see
    #             class docstring.
                
    #     Returns:
    #         params (list): List of parameters for the density profile, see
    #             class docstring.
    #     '''
    #     pass

    def mass(self, r, params, zmax=None, integrate=False):
        '''mass:
        
        Calculate the enclosed mass of the density profile. z can be optionally 
        provided to calculate the slab mass, otherwise the mass is calculated 
        as if z is integrated from -inf to inf.
        
        Args:
            R (array): Array of galactocentric cylindrical radii in kpc
            params (list): List of parameters for the density profile, see
                class docstring.
            z (array): Array of heights above the x-y plane in kpc, if None 
                then integrate from -inf to inf. [default: None] 
        '''
        mass = 0.
        for i, densprofile in enumerate(self.densprofiles):
            n_params = densprofile.n_params
            if isinstance(densprofile, SphericalDensityProfile):
                mass += densprofile.mass(r=r, 
                    params=params[i*n_params:(i+1)*n_params], 
                    integrate=integrate)
            if isinstance(densprofile, AxisymmetricDensityProfile):
                mass += densprofile.mass(R=r, 
                    params=params[i*n_params:(i+1)*n_params], 
                    zmax=zmax, integrate=integrate)
        return mass
    
    def effective_volume(self, params, rmin=0., rmax=np.inf, zmax=np.inf, 
        integrate=False):
        '''effective_volume:

        Integrate the density profile over all space. This is the effective
        volume of the density profile.

        Args:
            params (list): List of parameters for the density profile
            rmin (float): Minimum radius in kpc [default: 0.]
            rmax (float): Maximum radius in kpc [default: np.inf]
            zmax (float): Maximum height above the plane in kpc [default: np.inf]
        
        Returns:
            effvol (float): Effective volume of the density profile
        '''
        vol = 0.
        for i, densprofile in enumerate(self.densprofiles):
            n_params = densprofile.n_params
            if isinstance(densprofile, SphericalDensityProfile):
                _vol = densprofile.effective_volume(
                    params=params[i*n_params:(i+1)*n_params], 
                    rmin=rmin, rmax=rmax, integrate=integrate)
                vol += _vol
            if isinstance(densprofile, AxisymmetricDensityProfile):
                _vol = densprofile.effective_volume(
                    params=params[i*n_params:(i+1)*n_params], 
                    rmin=rmin, rmax=rmax, zmax=zmax, integrate=integrate)
                vol += _vol
        return vol



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

# ----------------------------------------------------------------------------

# Utilities

def calculate_critical_density(H0=_HUBBLE_PARAM, z=0, Om0=0.3, Ode0=0.7, 
    Ok0=0.0, Or0=0.0):
    '''calculate_critical_density:

    Calculate the critical density of the universe. Default parameters will 
    return the Planck 2015 value. Will also scan for defaults in Planck 2018 
    and WMAP9.

    Args:
        H0 (float): Hubble parameter in km/s/Mpc at z=0, can be astropy 
            quantity [default: 0.6774]
        z (float): Redshift [default: 0]
        Om0 (float): Matter density parameter at z=0 [default: 0.3]
        Ode0 (float): Dark energy density parameter at z=0 [default: 0.7]
        Ok0 (float): Curvature density parameter at z=0 [default: 0.0]

    Returns:
        rho_crit (float): Critical density of the universe as astropy quantity
            in Msun/kpc^3
    '''
    # Need to actually convert to
    if _HUBBLE_PARAM*100 == astropy.cosmology.Planck15.H0.value and z == 0.:
        rho_crit = astropy.cosmology.Planck15.critical_density0
    elif _HUBBLE_PARAM*100 == astropy.cosmology.Planck18.H0.value and z == 0.:
        rho_crit = astropy.cosmology.Planck18.critical_density0
    elif _HUBBLE_PARAM*100 == astropy.cosmology.WMAP9.H0.value and z == 0.:
        rho_crit = astropy.cosmology.WMAP9.critical_density0
    else:
        raise NotImplementedError("Need to implement this")
    return rho_crit.to(apu.Msun/apu.kpc**3)

def get_virial_radius(r, mass, rho_crit=None, overdensity_factor=200.):
    '''get_virial_radius:

    Determine the virial radius of a set of particles based on their radii and 
    masses, given some critical density and density factor

    Args:
        r (array): Array of radii in kpc, can be astropy quantity.
        mass (array): Array of masses in Msun, can be astropy quantity.
        rho_crit (float): Critical density of the universe in Msun/kpc^3. Can 
            be astropy quantity. If not supplied will be calculated using 
            calculate_critical_density() with default arguments.
        overdensity_factor (float): Overdensity factor [default: 200.]
    
    Returns:
        Rvir (float): Virial radius in kpc, astropy quantity
    '''
    # Parse for astropy quantities
    if isinstance(r, apu.Quantity):
        r = r.to(apu.kpc).value
    if isinstance(mass, apu.Quantity):
        mass = mass.to(apu.Msun).value
    if rho_crit is None:
        rho_crit = calculate_critical_density().to(apu.Msun/apu.kpc**3).value
    elif isinstance(rho_crit, apu.Quantity):
        rho_crit = rho_crit.to(apu.Msun/apu.kpc**3).value
    
    # Search for the virial radius using a binary search
    argsort = np.argsort(r)
    r = r[argsort]
    mass = mass[argsort]

    # Function to calculate the enclosed mass
    mean_dens = lambda R: np.sum(mass[r <= R])/((4./3.)*np.pi*R**3)

    # Test the outermost radius
    if mean_dens(r.max()) > overdensity_factor*rho_crit:
        warnings.warn('Most distant particle is already within the virial'
            ' radius, returning r.max()')
        return r.max()*apu.kpc

    # Binary search
    counter = 0
    while True:
        R = (r.max() - r.min())/2. + r.min()
        if mean_dens(R) > overdensity_factor*rho_crit:
            mask = (r >= R)
            r = r[mask]
            mass = mass[mask]
        else:
            mask = (r <= R)
            r = r[mask]
            mass = mass[mask]
        if len(r) == 1:
            return R*apu.kpc
        if counter > 100:
            raise RuntimeError("Binary search failed to converge")
        counter += 1
        
def get_NFW_init_params(densfunc, r, mass, rho_crit=None, overdensity_factor=200.,
    conc=10., ):
    '''get_NFW_init_params:

    Args:
        densfunc (function): Function that returns the density profile. Should 
            be NFW
        r (array): Array of radii in kpc, can be astropy quantity.
        mass (array): Array of masses in Msun, can be astropy quantity.
        rho_crit (float): Critical density of the universe in Msun/kpc^3. Can
            be astropy quantity. If not supplied will be calculated using
            calculate_critical_density() with default arguments.
        overdensity_factor (float): Overdensity factor for virial radius 
            [default: 200.]
        conc (float): Concentration parameter to use for calculating rs
            from virial radius. MW approx 10. [default: 10.]

    Returns:
        init_params (list): List of initial parameters [rs, amp] for the NFW
            profile
    '''
    r = putil.parse_astropy_quantity(r, 'kpc')
    mass = putil.parse_astropy_quantity(mass, 'Msun')

    # Get the virial radius
    rvir = get_virial_radius(r, mass, rho_crit=rho_crit,
        overdensity_factor=overdensity_factor)
    rvir = putil.parse_astropy_quantity(rvir, 'kpc')
    rs = rvir/conc

    # Get the amplitude
    amp = np.sum(mass[r < rvir]) / densfunc.effective_volume([rs,1.],
        rmin=0., rmax=rvir)
    
    return [rs, amp]

