# ----------------------------------------------------------------------------
#
# TITLE - kinematics.py
# AUTHOR - James Lane
# PROJECT - tng-dfs
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Functions and utilities for kinematics
'''
__author__ = "James Lane"

### Imports
import numpy as np
import copy
from galpy import potential,orbit
import astropy.units as apu

from . import util as putil

_cdict = putil.load_config_to_dict()
_keywords = ['RO','VO']
_ro,_vo = putil.parse_config_dict(_cdict,_keywords)

# ----------------------------------------------------------------------------

def calculate_spherical_jeans_quantities(orbs,pot=None,pe=None,r_range=[0,100],
    n_bin=10,norm_by_galpy_scale_units=False,calculate_pe_with_pot=False,
    adaptive_binning=False,bin_edge=None,
    rs_is_bin_mean_r=False,t=0.,ro=_ro,vo=_vo):
    '''calculate_spherical_jeans_quantities:
    
    Calculate the quantities used in the spherical Jeans equation.
    
    Args:
        orbs (Orbits) - Orbits object containing particles / kinematic sample
        pot (optional, Potential) - Potential object representing the 
            gravitational potential experienced by orbs
        pe (optional, array) - Potential energy of each particle in orbs, in
            units of km^2/s^2 [default: None]
        r_range (optional, list) - Range of radii to consider, in kpc 
            [default: [0,100]]
        n_bin (optional, int) - Number of bins to use in calculating Jeans
            equation, note derivative quantities will be calculated with 
            n_bin+1 bins [default: 10]
        norm_by_galpy_scale_units (optional, bool) - If True, normalize the
            Jeans equation by galpy scale units [default: False]
        calculate_pe_at_bin_cents (optional, bool) - If True, calculate the 
            potential at the bin centers, rather than the mean potential of the 
            orbs in the bin [default: False]
        adaptive_binning (optional, bool or dict) - If True/dict, use the 
            adaptive binning routine in calculating kinematic quantities. If 
            dict then use as keyword arguments to get_radius_binning
            [default: False]
        bin_edge (optional, np.ndarray) - If provided, use these bin edges
            instead of calculating them (note these will be the bin edges for 
            the derivative quantities) [default: None]
        rs_is_bin_mean_r (optional, bool) - If True, use the mean radius of the
            samples in each bin as the radius for the bin (rs in qs), 
            otherwise use the bin centers [default: False]
        t (optional, float) - Time at which to calculate the Jeans equation,
            used as an argument to the potential [default: None]
        ro (optional, float) - Distance scale in kpc [default: 8.275]
        vo (optional, float) - Velocity scale in km/s [default: 220.]
    
    Returns:
        qs (tuple) - Tuple of kinematic quantities used to calculate Jeans
            equation, output from calculate_spherical_jeans_quantities, 
            in order: dnuvr2dr,dphidr,nu,vr2,vp2,vt2,rs
    '''
    # Make sure we have a potential
    if pot is None:
        assert pe is not None,\
            "Must provide either a Potential or a potential energy array"
    else:
        pot = copy.deepcopy(pot)
        potential.turn_physical_on(pot,ro=ro,vo=vo)
    if calculate_pe_with_pot:
        assert pot is not None,\
            "Must provide a Potential if calculating potential from pot"

    # Handle orbits
    orbs = copy.deepcopy(orbs)
    orbs.turn_physical_on(ro=ro,vo=vo)

    ## Determine bins for kinematic properties
    # First need bins for derivatives, one more bin than for the data itself, 
    # since we're taking derivatives
    if adaptive_binning or isinstance(adaptive_binning,dict):
        # If adaptive binning is a dictionary, then use that as the keyword
        # arguments for the adaptive binning routine
        if isinstance(adaptive_binning,dict):
            kwargs = adaptive_binning
        else:
            kwargs = {}
        # Get the bins for derivatives
        dr_bin_edge,_ = get_radius_binning(orbs,**kwargs)
        n_dr_bin = len(dr_bin_edge)-1
        n_bin = n_dr_bin-1
    elif bin_edge is not None:
        dr_bin_edge = np.asarray(bin_edge)
        n_dr_bin = len(dr_bin_edge)-1
        n_bin = n_dr_bin-1
    else:
        n_dr_bin = n_bin+1
        dr_bin_edge = np.linspace(r_range[0],r_range[1],n_dr_bin+1)

    dr_bin_cents = (dr_bin_edge[1:]+dr_bin_edge[:-1])/2
    # dr_bin_delta = dr_bin_edge[1:]-dr_bin_edge[:-1]

    # One fewer bin for data, The edges are the derivative bin centers
    bin_edge = copy.deepcopy(dr_bin_cents)
    bin_cents = (bin_edge[1:]+bin_edge[:-1])/2
    # bin_delta = bin_edge[1:]-bin_edge[:-1]

    # Bin the data, derivative quantities first
    nuvr2 = np.zeros_like(dr_bin_cents)
    phi = np.zeros_like(dr_bin_cents)
    # Non-derivative quantities
    nu = np.zeros_like(bin_cents)
    vr2 = np.zeros_like(bin_cents)
    vt2 = np.zeros_like(bin_cents)
    vp2 = np.zeros_like(bin_cents)
    mean_r = np.zeros_like(bin_cents)

    # Handle potential energy
    rs = orbs.r(use_physical=True).to(apu.kpc).value
    if isinstance(pe,apu.Quantity):
        pe = pe.to(apu.km**2/apu.s**2).value
    if pot is not None:
        pe = potential.evaluatePotentials(pot,orbs.R(),orbs.z(),t=t,
            use_physical=True).to(apu.km**2/apu.s**2).value
        pe_bin_cents = potential.evaluatePotentials(pot,dr_bin_cents*apu.kpc,
            0*apu.kpc,t=t,use_physical=True).to(apu.km**2/apu.s**2).value

    # Derivative quantities
    for i in range(len(dr_bin_cents)):
        bin_mask = (rs>=dr_bin_edge[i]) & (rs<dr_bin_edge[i+1])
        n_in_bin = np.sum( bin_mask )
        bin_vol = 4*np.pi/3*(dr_bin_edge[i+1]**3-dr_bin_edge[i]**3)
        dr_nu = n_in_bin/bin_vol
        dr_vr2 = np.mean(orbs.vr(use_physical=True).to(apu.km/apu.s).value
            [bin_mask]**2.)
        if calculate_pe_with_pot:
            phi[i] = pe_bin_cents[i]
        else:
            phi[i] = np.mean(pe[bin_mask])
        nuvr2[i] = dr_nu*dr_vr2
    dphidr = np.diff(phi)/np.diff(dr_bin_cents)
    dnuvr2dr = np.diff(nuvr2)/np.diff(dr_bin_cents)

    # Non-derivative quantities
    for i in range(len(bin_cents)):
        bin_mask = (rs>=bin_edge[i]) & (rs<bin_edge[i+1])
        n_in_bin = np.sum( bin_mask )
        bin_vol = 4*np.pi/3*(bin_edge[i+1]**3-bin_edge[i]**3)
        nu[i] = n_in_bin/bin_vol
        vr2[i] = np.mean(orbs.vr(use_physical=True).to(apu.km/apu.s).value
            [bin_mask]**2.)
        vp2[i] = np.mean(orbs.vtheta(use_physical=True).to(apu.km/apu.s).value
            [bin_mask]**2.)
        vt2[i] = np.mean(orbs.vT(use_physical=True).to(apu.km/apu.s).value
            [bin_mask]**2.)
        mean_r[i] = np.mean(rs[bin_mask])
    
    # Normalize densities by number of orbits so they're proper number 
    # densities
    nu /= len(orbs)
    dnuvr2dr /= len(orbs)

    if norm_by_galpy_scale_units:
        nu = nu*(ro**3)
        vr2 = vr2/(vo**2)
        vp2 = vp2/(vo**2)
        vt2 = vt2/(vo**2)
        bin_cents = bin_cents/ro
        dphidr = dphidr*ro/(vo**2)
        dnuvr2dr = dnuvr2dr*(ro**4)/(vo**2)
    
    if rs_is_bin_mean_r:
        rs = mean_r
    else:
        rs = bin_cents

    return dnuvr2dr,dphidr,nu,vr2,vp2,vt2,rs

def calculate_spherical_jeans(orbs,pot=None,pe=None,n_bootstrap=1,
    r_range=[0,100],n_bin=10,norm_by_galpy_scale_units=False,
    norm_by_nuvr2_r=True,calculate_pe_with_pot=False,
    adaptive_binning=False,bin_edge=None,
    rs_is_bin_mean_r=False,return_kinematics=True,
    return_terms=False,t=0.,ro=_ro,vo=_vo):
    '''calculate_spherical_jeans:

    Calculate the spherical Jeans equation for a given kinematic sample

    Args:
        orbs (Orbits) - Orbits object containing particles / kinematic sample
        pot (optional, Potential) - Potential object representing the 
            gravitational potential experienced by orbs
        pe (optional, array) - Potential energy of each particle in orbs in
            units of km^2/s^2 [default: None]
        n_bootstrap (optional, int) - Number of bootstrap samples to calculate 
            the Jeans equation for, if 1, then don't bootstrap [default: 1]
        r_range (optional, list) - Range of radii to consider, in kpc 
            [default: [0,100]]
        n_bin (optional, int) - Number of bins to use in calculating Jeans
            equation, note derivative quantities will be calculated with 
            n_bin+1 bins [default: 10]
        norm_by_galpy_scale_units (optional, bool) - If True, normalize the
            Jeans equation by galpy scale units [default: False]
        norm_by_nuvr2_r (optional, bool) - If True, normalize the Jeans equation
            by nu*vr^2/r [default: True]
        calculate_pe_with_pot (optional, bool) - If True, calculate the 
            potential at the bin centers, rather than the mean potential of the 
            orbs in the bin [default: False]
        adaptive_binning (optional, bool or dict) - If True/dict, use the 
            adaptive binning routine in calculating kinematic quantities. If 
            dict then use as keyword arguments to get_radius_binning
            [default: False]
        bin_edge (optional, np.ndarray) - If provided, use these bin edges
            instead of calculating them (note these will be the bin edges for 
            the derivative quantities) [default: None]
        rs_is_bin_mean_r (optional, bool) - If True, use the mean radius of the
            samples in each bin as the radius for the bin (rs in qs), 
            otherwise use the bin centers [default: False]
        return_kinematics (optional, bool) - If True, return the kinematics
            used to calculate the Jeans equation [default: True]
        return_terms (optional, bool) - If True, return the individual terms
            of the Jeans equation [default: False]
        t (optional, float) - Time at which to calculate the Jeans equation,
            used as an argument to the potential [default: None]
        ro (optional, float) - Distance scale in kpc [default: 8.275]
        vo (optional, float) - Velocity scale in km/s [default: 220.]
    
    Returns:
        J (np.ndarray) - Jeans equation, may be normalized
        rs (np.ndarray) - Radii at which Jeans equation is calculated
        qs (tuple) - Tuple of kinematic quantities used to calculate Jeans
            equation, output from calculate_spherical_jeans_quantities, 
            in order: dnuvr2dr,dphidr,nu,vr2,vp2,vt2,rs
    '''
    # Compute the quantities for the spherical Jeans equation
    if n_bootstrap>1:
        qs = np.zeros((7,n_bootstrap,n_bin))
        for i in range(n_bootstrap):
            # Random bootstrap index
            indx = np.random.choice(np.arange(len(orbs),dtype=int),
                size=len(orbs),replace=True)
            if pe is not None:
                _pe = pe[indx]
            else:
                _pe = None
            _qs = calculate_spherical_jeans_quantities(orbs[indx],pot=pot,
                pe=_pe,r_range=r_range,n_bin=n_bin,
                norm_by_galpy_scale_units=norm_by_galpy_scale_units,
                calculate_pe_with_pot=calculate_pe_with_pot,
                adaptive_binning=adaptive_binning,bin_edge=bin_edge,
                rs_is_bin_mean_r=rs_is_bin_mean_r,t=t,ro=ro,vo=vo)
            qs[:,i,:] = _qs
    else:
        qs = calculate_spherical_jeans_quantities(orbs,pot=pot,pe=pe,
            r_range=r_range,n_bin=n_bin,
            norm_by_galpy_scale_units=norm_by_galpy_scale_units,
            calculate_pe_with_pot=calculate_pe_with_pot,
            adaptive_binning=adaptive_binning, bin_edge=bin_edge,
            rs_is_bin_mean_r=rs_is_bin_mean_r,t=t,ro=ro,vo=vo)

    dnuvr2dr,dphidr,nu,vr2,vp2,vt2,rs = qs

    # Compute the Jeans equation
    J1 = copy.deepcopy(dnuvr2dr)
    J2 = nu*(dphidr + (2*vr2-vp2-vt2)/rs) 
    J = J1 + J2

    # Normalize by nu*vr^2/r if desired. Note that this returns the same 
    # answer regardless of whether using physical or galpy units.
    if norm_by_nuvr2_r and not norm_by_galpy_scale_units:
        J1 = J1/(nu*vr2/rs)
        J2 = J2/(nu*vr2/rs)
        J = J/(nu*vr2/rs)

    if return_kinematics:
        if return_terms:
            return J,rs,qs,J1,J2
        else:
            return J,rs,qs
    else:
        if return_terms:
            return J,rs,J1,J2
        else:
            return J,rs
    
def calculate_weighted_average_J(J,rs,dens=None,qs=None,weights=None,
    handle_nans=False):
    '''calculate_weighted_average_J:

    Calculate the weighted average of the Jeans equation residual term + the 
    dispersion. By default will weight by density*rs^2, which is the mass of 
    the bin. Will get the density from qs if not provided. If weights are 
    provided, will use those instead.

    Args:
        J (np.ndarray) - Jeans equation residual term
        rs (np.ndarray) - Radii at which Jeans equation is calculated
        dens (optional, np.ndarray) - Density profile of the kinematic sample
            [default: None]
        qs (optional, tuple) - Tuple of kinematic quantities used to calculate 
            Jeans equation, output from calculate_spherical_jeans_quantities, 
            in order: dnuvr2dr,dphidr,nu,vr2,vp2,vt2,rs [default: None]
        weights (optional, np.ndarray) - Weights to use in calculating the
            weighted average [default: None]
        handle_nans (optional, bool) - If True, will handle nans in the Jeans
            equation by setting them to zero [default: False]
    
    Returns:
        J_avg (float) - Weighted average of the Jeans equation residual term
        J_dispersion (float) - Weighted average of the Jeans equation residual
            term dispersion
    '''
    if dens is None:
        if qs is None:
            raise Exception('Must provide either qs or dens')
        else:
            dens = qs[2]
    
    if weights is None:
        weights = dens*rs**2
    
    J_avg = np.average(J,weights=weights)
    J_dispersion = np.sqrt(np.average((J-J_avg)**2,weights=weights))
    
    return J_avg,J_dispersion

def beta_any_alpha_cuddeford91(r,ra=1.,alpha=0.,beta=None):
    '''beta_any_alpha:

    Calculate beta as a function of radius for any central alpha (2*beta).

    Equation 38 from Cuddeford (1991)

    Args:
        r (float or array): Radius
        ra (float): Scale radius
        alpha (float): Twice the central anisotropy parameter, default 0
        beta (optional, float): The anisotropy parameter at 

    Returns:
        beta (float or array): Anisotropy
    '''
    if beta is not None:
        alpha = 2*beta
    return (r**2-alpha*ra**2)/(r**2+ra**2)

def beta_ossipkov_merrit(r,ra=1.):
    '''beta_ossipkov_merrit:

    Calculate beta as a function of radius for the Ossipkov-Merrit DF

    Args:
        r (float or array): Radius
        ra (float): Scale radius

    Returns:
        beta (float or array): Anisotropy parameter
    '''
    return (r**2)/(r**2+ra**2)