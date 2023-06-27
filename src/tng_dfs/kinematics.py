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
from galpy import potential
import astropy.units as apu

from . import util as putil

_cdict = putil.load_config_to_dict()
_keywords = ['RO','VO']
_ro,_vo = putil.parse_config_dict(_cdict,_keywords)

# ----------------------------------------------------------------------------

def calculate_spherical_jeans_quantities(orbs,pot,r_range=[0,100],n_bin=10,
    norm_by_galpy_scale_units=False,calculate_pe_with_pot=False,ro=_ro,vo=_vo):
    '''calculate_spherical_jeans_quantities:
    
    Calculate the quantities used in the spherical Jeans equation.
    
    Args:
        orbs (Orbits) - Orbits object containing particles / kinematic sample
        pot (Potential) - Potential object representing the gravitational 
            potential experienced by orbs
        r_range (optional, list) - Range of radii to consider, in kpc 
            [default: [0,100]]
        n_bin (optional, int) - Number of bins to use in calculating Jeans
            equation, note derivative quantities will be calculated with 
            n_bin+1 bins [default: 10]
        norm_by_galpy_scale_units (optional, bool) - If True, normalize the
            Jeans equation by galpy scale units [default: False]
        calculate_pe_with_pot (optional, bool) - If True, calculate the 
            potential at the bin centers, rather than the mean potential of the 
            orbs in the bin [default: False]
        ro (optional, float) - Distance scale in kpc [default: 8.275]
        vo (optional, float) - Velocity scale in km/s [default: 220.]
    
    Returns:
        qs (tuple) - Tuple of kinematic quantities used to calculate Jeans
            equation, output from calculate_spherical_jeans_quantities, 
            in order: dnuvr2dr,dphidr,nu,vr2,vp2,vt2,rs
    '''
    orbs = copy.deepcopy(orbs)
    orbs.turn_physical_on(ro=ro,vo=vo)
    pot = copy.deepcopy(pot)
    pot.turn_physical_on(ro=ro,vo=vo)

    ## Determine bins for kinematic properties
    
    # First need bins for derivatives, one more bin than for the data itself, 
    # since we're taking derivatives
    n_dr_bin = n_bin+1
    dr_bin_edge = np.linspace(r_range[0],r_range[1],n_dr_bin+1)
    dr_bin_cents = (dr_bin_edge[1:]+dr_bin_edge[:-1])/2
    # dr_bin_delta = dr_bin_edge[1:]-dr_bin_edge[:-1]

    # One fewer bin for data, since we're taking derivatives. The edges are 
    # the derivative bin centers
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

    rs = orbs.r(use_physical=True).to(apu.kpc).value
    pe = potential.evaluatePotentials(pot,orbs.R(),orbs.z(),
        use_physical=True).to(apu.km**2/apu.s**2).value
    pe_bin_cents = potential.evaluatePotentials(pot,dr_bin_cents*apu.kpc,
        0*apu.kpc,use_physical=True).to(apu.km**2/apu.s**2).value

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

    return dnuvr2dr,dphidr,nu,vr2,vp2,vt2,bin_cents

def calculate_spherical_jeans(orbs,pot,r_range=[0,100],n_bin=10,
    norm_by_galpy_scale_units=False,norm_by_nuvr2_r=True,
    calculate_pe_with_pot=False,return_kinematics=True,ro=_ro,vo=_vo):
    '''calculate_spherical_jeans:

    Calculate the spherical Jeans equation for a given kinematic sample

    Args:
        orbs (Orbits) - Orbits object containing particles / kinematic sample
        pot (Potential) - Potential object representing the gravitational 
            potential experienced by orbs
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
        return_kinematics (optional, bool) - If True, return the kinematics
            used to calculate the Jeans equation [default: True]
        ro (optional, float) - Distance scale in kpc [default: 8.275]
        vo (optional, float) - Velocity scale in km/s [default: 220.]
    
    Returns:
        J (np.ndarray) - Jeans equation, may be normalized
        rs (np.ndarray) - Radii at which Jeans equation is calculated
        qs (tuple) - Tuple of kinematic quantities used to calculate Jeans
            equation, output from calculate_spherical_jeans_quantities, 
            in order: dnuvr2dr,dphidr,nu,vr2,vp2,vt2,rs
    '''
    # Compute the 
    qs = calculate_spherical_jeans_quantities(orbs,pot,r_range=r_range,
        n_bin=n_bin,norm_by_galpy_scale_units=norm_by_galpy_scale_units,
        calculate_pe_with_pot=calculate_pe_with_pot,ro=ro,vo=vo)

    dnuvr2dr,dphidr,nu,vr2,vp2,vt2,rs = qs

    # Compute the Jeans equation
    J = nu*(dphidr + (2*vr2-vp2-vt2)/rs) + dnuvr2dr

    # Normalize by nu*vr^2/r if desired. Note that this returns the same 
    # answer regardless of whether using physical or galpy units.
    if norm_by_nuvr2_r and not norm_by_galpy_scale_units:
        J = J/(nu*vr2/rs)

    if return_kinematics:
        return J,rs,qs
    else:
        return J,rs
    
def calculate_weighted_average_J(J,rs,dens=None,qs=None,weights=None):
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