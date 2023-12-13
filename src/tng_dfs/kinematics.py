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
from galpy import potential,orbit,df
import astropy.units as apu
import dill as pickle

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
        dr_bin_edge,_,_ = get_radius_binning(orbs,**kwargs)
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
    rs_is_bin_mean_r=True,return_kinematics=True,
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
    if adaptive_binning or isinstance(adaptive_binning,dict):
        # If adaptive binning is a dictionary, then use that as the keyword
        # arguments for the adaptive binning routine
        if isinstance(adaptive_binning,dict):
            kwargs = adaptive_binning
        else:
            kwargs = {}
        # Get the bins for derivatives
        dr_bin_edge,_,_ = get_radius_binning(orbs,**kwargs)
        n_dr_bin = len(dr_bin_edge)-1
        n_bin = n_dr_bin-1
        adaptive_binning = False
        bin_edge = dr_bin_edge

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
    
    if handle_nans:
        mask = np.isnan(J) | np.isinf(J) | np.isnan(weights) | np.isinf(weights)
        J[mask] = 0.
        weights[mask] = 0.

    J_avg = np.average(J,weights=weights)
    J_dispersion = np.sqrt(np.average((J-J_avg)**2,weights=weights))
    
    return J_avg,J_dispersion

def get_radius_binning(orbs, n=1000, rmin=0., rmax=np.inf,
    bin_mode='restrict number', max_bin_size=None, bin_equal_n=True, 
    end_mode='bin', bin_cents_mode='delta', delta_n=10, delta_r=1e-8):
    '''get_radius_binning:

    Bin up the samples according to radius so that there's a minimum number 
    of samples in each bin. The algorithm is as follows:

    Figure out the number of bins that satisfies the criterion that 
    each bin has at least min_n samples. Then, place the bin edges so that 
    they each contain the same number of samples.

    There are 3 possibilities for bin_mode:
    - If bin_mode is 'enforce size', then max_bin_size will be enforced. n will 
        be flexible and will be decreased if necessary to satisfy the 
        max_bin_size. bin_equal_n can be used.
    - If bin_mode is 'enforce numbers', then n will be enforced as the minimum
        number of samples per bin. bin_equal_n can be used.
    - If bin_mode is 'exact numbers', then n will be enforced as the exact
        number of samples per bin. max_bin_size, bin_equal_n, and end_mode
        will be ignored.

    Args:
        orbs (orbit.Orbit or np.array): Array of orbits or radii in kpc if array
        n (int): Minimum number of samples within each bin
        rmin (float): Minimum radius to consider kpc, can be Quantity
        rmax (float): Maximum radius to consider kpc, can be Quantity. If 
            np.inf, then just use the maximum radius in the sample.
        bin_mode (str) One of 'enforce size', 'enforce numbers', 
            'exact numbers'. See above for details.
        max_bin_size (float): Maximum size of a bin in kpc, can be Quantity
        bin_equal_n (bool): If True, then each bin will have ~ the same number
            of samples. If False, then the number of samples in each bin will
            be as close to min_n as possible.
        end_mode (str): Behaviour for the samples in the last bin. If 
            'bin', then last samples will be contained in their own bin. If 
            'join' then the samples in the last bin will be joined with the 
            previous bin. If float from 0 to 1 then will use 'join' if the 
            fraction of samples in the last bin compared w/ the previous bin is
            less than the float, otherwise will use 'bin'. If 'ignore', then
            the last samples will be ignored.
        bin_cents_mode (str): If 'mean' then compute the bin centers as the 
            mean of the samples in the bin. If 'median' then compute the bin
            centers as the median of the samples in the bin. If 'delta' then
            compute the bin centers as the delta between the bin edges. 
            [default 'delta']
        delta_n (int): Small number to decrease n by if the bin size is too
            large.
        delta_r (float): Small number to add to the bin edges to make sure that
            the bin edges are not exactly on top of a sample.
        
    '''
    ## Handle keyword collisions
    # specifically focus on end_mode? Maybe test

    # Check the model inputs
    assert bin_mode in ['enforce size','enforce numbers','exact numbers'], \
        'bin_mode must be one of "enforce size", "enforce numbers", "exact numbers"'
    assert end_mode in ['bin','join','ignore'] or isinstance(end_mode,float), \
        'last_samples must be one of "bin", "join", "ignore", or be a float'

    # Wrangle the numerical inputs
    if isinstance(orbs,orbit.Orbit):
        rs = orbs.r()
    else:
        rs = copy.deepcopy(orbs)
    if isinstance(rs,apu.quantity.Quantity):
        rs = rs.to(apu.kpc).value
    if isinstance(rmin,apu.quantity.Quantity):
        rmin = rmin.to(apu.kpc).value
    if isinstance(rmax,apu.quantity.Quantity):
        rmax = rmax.to(apu.kpc).value
    if isinstance(max_bin_size,apu.quantity.Quantity):
        max_bin_size = max_bin_size.to(apu.kpc).value
    n = int(n)
    
    # Mask out the radii that are outside the range
    mask = (rs > rmin) & (rs < rmax)
    rs = rs[mask]
    rs_sorted = np.sort(rs)

    # There will be a number of bins ~ number of floor(len(rs)/n)
    bin_edges = [rmin,]
    n_bins = int(np.floor(len(rs)/n))

    # First do the 'enforce size' algorithm
    if bin_mode == 'enforce size':
        bin_size_good = False
        while not bin_size_good:
            bin_edges = [rmin,]
            n_bins = int(np.floor(len(rs)/n))
            if bin_equal_n: # Increase n until it's possible to have equal n
                _n = n
                while len(rs)/_n > n_bins:
                    if len(rs)/(_n+1) <= n_bins:
                        break
                    _n += 1
            else: # Don't care, just use n
                _n = n
            for i in range(n_bins):
                new_bin_edge = rs_sorted[(i+1)*_n-1]+delta_r
                bin_edges.append(new_bin_edge)
            if np.any(np.diff(bin_edges) > max_bin_size):
                n -= delta_n
                if n <= 0:
                    raise ValueError('n has been decreased to 0')
            else:
                bin_size_good = True

    # Then do the 'enforce numbers' algorithm
    if bin_mode == 'enforce numbers':
        bin_edges = [rmin,]
        n_bins = int(np.floor(len(rs)/n))
        if bin_equal_n: # Increase n until it's possible to have equal n
            _n = n
            while len(rs)/_n > n_bins:
                _n += 1
        else: # Don't care, just use n
            _n = n
        for i in range(n_bins):
            new_bin_edge = rs_sorted[(i+1)*_n-1]+delta_r
            bin_edges.append(new_bin_edge)

    # Now do the 'exact numbers' algorithm
    if bin_mode == 'exact numbers':
        bin_edges = [rmin,]
        n_bins = int(np.floor(len(rs)/n))
        for i in range(n_bins):
            new_bin_edge = rs_sorted[(i+1)*n-1]+delta_r
            bin_edges.append(new_bin_edge)

    # Handling the last bin
    if isinstance(end_mode,float):
        if end_mode < 0 or end_mode > 1:
            raise ValueError('end_mode must be between 0 and 1')
        n_end = np.sum(rs > bin_edges[-1])
        n_final_bin = np.sum((rs > bin_edges[-2])&(rs <= bin_edges[-1]))
        if n_end/n_final_bin < end_mode:
            end_mode = 'join'
        else:
            end_mode = 'bin'
    if bin_mode == 'exact numbers': # Each bin has exactly n samples, no need to deal with ends
        pass
    elif end_mode == 'bin': # Add a bin edge at the end
        bin_edges.append(rs_sorted[-1]+delta_r)
    elif end_mode == 'join': # Move the last bin edge to the last sample
        bin_edges[-1] = rs_sorted[-1]+delta_r
    elif end_mode == 'ignore': # Ignore the final samples
        pass

    # Get the number of samples in each bin as a useful quantity
    n_samples = np.empty(len(bin_edges)-1)
    for i in range(len(bin_edges)-1):
        n_samples[i] = len(rs[(rs > bin_edges[i]) & (rs <= bin_edges[i+1])])
    
    bin_edges = np.asarray(bin_edges)

    # Compute the bin centers
    assert bin_cents_mode in ['mean','median','delta'], \
        'bin_cents_mode must be one of "mean", "median", "delta"'
    if bin_cents_mode == 'delta':
        bin_cents = (bin_edges[1:]+bin_edges[:-1])/2
    elif bin_cents_mode in ['mean','median']:
        bin_cents = np.empty(len(bin_edges)-1)
        for i in range(len(bin_edges)-1):
            mask = (rs > bin_edges[i]) & (rs <= bin_edges[i+1])
            if bin_cents_mode == 'mean':
                bin_cents[i] = np.mean(rs[mask])
            elif bin_cents_mode == 'median':
                bin_cents[i] = np.median(rs[mask])

    return bin_edges, bin_cents, n_samples


def calculate_Krot(orbs,masses,return_kappa=False):
    '''calculate_Krot:

    Calculate Krot, rotational support factor, by summing the mass-weighted 
    kappa values.

    Args:
        orbs (galpy.orbit.Orbit) - Orbits representing the motions of the 
            particles
        masses (np.array or float) - Masses
        return_kappa (optional, bool) - If True, return the individual kappa
            values [default: False]
    
    Returns:
        Krot (float) - Rotational support factor
    '''
    vphi = orbs.vT()
    vtot = (orbs.vR()**2.+orbs.vz()**2.+orbs.vT()**2.)**0.5
    if isinstance(vphi,apu.Quantity):
        vphi = vphi.to(apu.km/apu.s).value
    if isinstance(vtot,apu.Quantity):
        vtot = vtot.to(apu.km/apu.s).value
    
    kappa = vphi**2./vtot**2.
    Krot = np.sum(kappa*masses)/np.sum(masses)
    if return_kappa:
        return Krot,kappa
    else:
        return Krot

### Anisotropy computation & profiles ###

def compute_betas_bootstrap(orbs, bin_edges, n_bootstrap=10, 
    compute_betas_kwargs={}):
    '''compute_betas_bootstrap:

    Take a set of orbits and use bootstrapping to compute a distribution of 
    betas.

    Args:
        orbs (galpy.orbit.Orbit) - Orbits object 
        bin_edges (np.ndarray) - Radius bin edges used to bin data to compute 
            dispersions/mean-squares.
        n_bootstrap (int) - Number of bootstrap samples to take [default: 10]
        compute_betas_kwargs (dict) - Keyword arguments to pass to 
            compute_betas [default: {}]
    
    Returns:
        betas (np.ndarray) - Array of betas of shape (n_bootstrap,n_bin)
        if return_kinematics=True in compute_betas_kwargs: 
            [each array has shape (n_bootstrap,n_bin)]
            svr2/mvr2 (np.ndarray) - Radial velocity dispersion squared / 
                mean-square velocity
            svp2/mvp2 (np.ndarray) - Azimuthal velocity dispersion squared / 
                mean-square velocity
            svt2/mvt2 (np.ndarray) - Polar velocity dispersion squared / 
                mean-square velocity
    '''
    betas = np.zeros((n_bootstrap,len(bin_edges)-1))
    _return_kinematics = compute_betas_kwargs.get('return_kinematics',False)
    if _return_kinematics:
        bvr = np.zeros((n_bootstrap,len(bin_edges)-1))
        bvp = np.zeros((n_bootstrap,len(bin_edges)-1))
        bvt = np.zeros((n_bootstrap,len(bin_edges)-1))
    
    for i in range(n_bootstrap):
        indx = np.random.choice(np.arange(len(orbs),dtype=int), 
            size=len(orbs), replace=True)
        _orbs = orbs[indx]
        res = compute_betas(_orbs, bin_edges, **compute_betas_kwargs)
        if _return_kinematics:
            betas[i],bvr[i],bvp[i],bvt[i] = res
        else:
            betas[i] = res
    
    if _return_kinematics:
        return betas,bvr,bvp,bvt
    else:
        return betas

def compute_betas(orbs, bin_edges, use_dispersions=True, return_kinematics=False):
    '''compute_betas:

    General function to compute beta either from velocity dispersions or 
    from the mean-square velocities. Velocities are sourced from an orbit 
    object.

    Args:
        orbs (galpy.orbit.Orbit) - Orbits object 
        bin_edges (np.ndarray) - Radius bin edges used to bin data to compute 
            dispersions/mean-squares.
        use_dispersions (bool) - If True, use dispersions to compute beta,
            otherwise use mean-square velocities [default True]
        return_kinematics (bool) - If True, return the quantities used to 
            construct beta. If False, just return beta [default False]
        
    Returns:
        beta (np.ndarray) - Anisotropy parameter
        if return_kinematics:
            svr2/mvr2 (np.ndarray) - Radial velocity dispersion squared / 
                mean-square velocity
            svp2/mvp2 (np.ndarray) - Azimuthal velocity dispersion squared / 
                mean-square velocity
            svt2/mvt2 (np.ndarray) - Polar velocity dispersion squared / 
                mean-square velocity
    '''
    # Get the radii & velocities
    rs = orbs.r(use_physical=True).to(apu.kpc).value
    vr = orbs.vr(use_physical=True).to(apu.km/apu.s).value
    vp = orbs.vT(use_physical=True).to(apu.km/apu.s).value
    vt = orbs.vtheta(use_physical=True).to(apu.km/apu.s).value

    bvr = np.zeros(len(bin_edges)-1)
    bvp = np.zeros(len(bin_edges)-1)
    bvt = np.zeros(len(bin_edges)-1)

    for i in range(len(bin_edges)-1):
        bin_mask = (rs>=bin_edges[i]) & (rs<bin_edges[i+1])
        if use_dispersions:
            bvr[i] = np.std(vr[bin_mask])**2
            bvp[i] = np.std(vp[bin_mask])**2
            bvt[i] = np.std(vt[bin_mask])**2
        else:
            bvr[i] = np.mean(np.square(vr[bin_mask]))
            bvp[i] = np.mean(np.square(vp[bin_mask]))
            bvt[i] = np.mean(np.square(vt[bin_mask]))
    
    beta = 1 - (bvp + bvt)/(2*bvr)

    if return_kinematics:
        return beta,bvr,bvp,bvt
    else:
        return beta

def beta_constant(r, beta=0.):
    '''beta_constant: 
    
    Constant beta profile, included for completeness.
    
    Args:
        r (float or array): Radius
        beta (float): Anisotropy parameter
    
    Returns:
        beta (float or array): Anisotropy parameter
    '''
    if isinstance(r,(float,int)):
        return beta
    if isinstance(r,np.ndarray):
        return np.ones(len(r))*beta
    raise Exception('r must be float, int, or np.ndarray')

def beta_osipkov_merritt(r,ra=1.):
    '''beta_osipkov_merritt:

    Calculate beta as a function of radius for the Ossipkov-Merrit DF

    Args:
        r (float or array): Radius
        ra (float): Scale radius

    Returns:
        beta (float or array): Anisotropy parameter
    '''
    return (r**2)/(r**2+ra**2)

def beta_cuddeford91(r,ra=1.,alpha=0.):
    '''beta_cuddeford91:

    Calculate beta as a function of radius for the generalized anisotropic model 
    presented by Cuddeford (1991), specifically equation 38. 

    Args:
        r (float or array): Radius
        ra (float): Scale radius
        alpha (float): The negative of the central anisotropy.

    Returns:
        beta (float or array): Anisotropy
    '''
    return (r**2-alpha*ra**2)/(r**2+ra**2)

### Convenience functions ###

def _E_Enorm_Jz_Jcirc_bounds():
    '''_E_Enorm_Jz_Jcirc_bounds:

    Fetch the bounds for Jz_Jcirc vs E_Enorm plane for the standard kinematic 
    decomposition.

    Args:
        None
    
    Returns:
        Jz_Jcirc_halo_bound (float) - Jz/Jcirc boundary between halo and disk
        Jz_Jcirc_disk_bound (float) - Jz/Jcirc boundary between thin/thick disk
        Enorm_bulge_bound (float) - Enorm boundary between bulge and halo
    '''
    Jz_Jcirc_halo_bound = 0.5
    Jz_Jcirc_disk_bound = 0.8
    Enorm_bulge_bound = -0.75
    return Jz_Jcirc_halo_bound,Jz_Jcirc_disk_bound,Enorm_bulge_bound

def half_mass_radius(rs,masses):
    '''half_mass_radius:

    Compute the half-mass radius from supplied radii and masses

    Args:
        rs (np.array) - Radii
        masses (np.array) - Masses
    
    Returns:
        hm (float) - Half-mass radius
    '''
    if isinstance(rs,apu.quantity.Quantity):
        rs = rs.to(apu.kpc).value
    if isinstance(masses,apu.quantity.Quantity):
        masses = masses.to(apu.Msun).value
    # Get the half-mass radius
    rs_sorted = np.sort(rs)
    masses_sorted = masses[np.argsort(rs)]
    masses_cumsum = np.cumsum(masses_sorted)
    hm = masses_cumsum[-1]/2
    hm_indx = np.argmin(np.abs(masses_cumsum-hm))
    return rs_sorted[hm_indx]

### Sampling from rotating DFs ###

def rotate_df_samples(orbs, frot, chi):
    '''rotate_df_samples:
    
    Take a set of orbits and flip Lz for some of them such that the rotating 
    tanh kernel DF is satisfied.

    Args:
        orbs (Orbits) - Orbits object to rotate
        frot (float) - Fraction of orbits to rotate, must be between -1 and 1
        chi (float) - Characteristic angular momentum scale in kpc km/s
    
    Returns:
        orbs_rot (Orbits) - Orbits object with some orbits rotated such
    '''
    # Check input parameters
    assert frot>=-1. and frot<=1., "frot must be between -1 and 1"
    rotation_sign = np.sign(frot)
    assert chi > 0., "chi must be greater than 0"

    # Copy the orbits
    vxvv_rot = copy.deepcopy(orbs.vxvv)
    Lz = orbs.Lz(use_physical=True)
    if isinstance(Lz,apu.Quantity):
        Lz = Lz.to(apu.kpc*apu.km/apu.s).value

    # Probability of flipping the tangential velocity
    pflip = 1-df_rotation_function(rotation_sign*Lz, frot=np.abs(frot), chi=chi)
    _p = np.random.uniform(size=len(orbs))
    flip_mask = _p < pflip
    # Lz_to_be_flipped = rotation_sign*Lz < 0.
    # flip_mask = flip_probable # & Lz_to_be_flipped
    vxvv_rot[:,2][flip_mask] = -vxvv_rot[:,2][flip_mask]

    if orbs._roSet and orbs._voSet:
        return orbit.Orbit(vxvv=vxvv_rot, ro=orbs._ro, vo=orbs._vo)
    else:
        return orbit.Orbit(vxvv=vxvv_rot)

def df_rotation_function(Lz, frot=0., chi=1.):
    '''df_rotation_function:

    Function that expresses the rotating DF as a function of Lz.


    '''
    assert frot>=-1. and frot<=1., "frot must be between -1 and 1"
    if isinstance(Lz,apu.Quantity):
        Lz = Lz.to(apu.kpc*apu.km/apu.s).value
    if isinstance(chi,apu.Quantity):
        chi = chi.to(apu.kpc*apu.km/apu.s).value
    gLz = tanh_rotation_kernel(Lz, chi=chi)
    k = frot/2.
    return 1-k+k*gLz

def tanh_rotation_kernel(Lz, chi=1.):
    if isinstance(Lz,apu.Quantity):
        Lz = Lz.to(apu.kpc*apu.km/apu.s).value
    if isinstance(chi,apu.Quantity):
        chi = chi.to(apu.kpc*apu.km/apu.s).value
    return np.tanh(Lz/chi)

### Reconstructing DFs ###

def reconstruct_anisotropic_df(dfa, pot, denspot, dfa_kwargs=None, 
    validate=False):
    '''reconstruct_anisotropic_df:

    Re-build an anisotropic DF after loading it in. This navigates weird 
    issues with isinstance() failures after pickling/unpickling.

    Will only currently work with these DFs:
        - df.constantbetadf
        - df.osipkovmerrittdf

    Args:
        dfa (galpy.df object or str) - Anisotropic DF that needs to be 
            reconstructed, if string then load it as a filename.
        pot (galpy.potential object) - Potential to use in the DF
        denspot (galpy.potential object) - Potential to use in the DF for the 
            density
        dfa_kwargs (dict) - Dictionary of attributes for the new DF
        validate (bool) - If True, validate the reconstructed DF by comparing
            the density profile to the input density profile [default: False]
        
    Returns:
        dfac (galpy.df object) - Reconstructed DF
    '''
    if isinstance(dfa,str):
        with open(dfa,'rb') as f:
            dfa = pickle.load(f)
    if dfa_kwargs is None:
        dfa_kwargs = {}
    else:
        assert isinstance(dfa_kwargs,dict), 'dfa_kwargs must be a dictionary'
    
    # Ensure some universal kwargs are set in dfa_kwargs
    if 'ro' not in dfa_kwargs:
        dfa_kwargs['ro'] = dfa._ro
    if 'vo' not in dfa_kwargs:
        dfa_kwargs['vo'] = dfa._vo
    if 'rmax' not in dfa_kwargs:
        dfa_kwargs['rmax'] = dfa._rmax

    # Reconstruct based on DF type
    if dfa.__class__.__name__ == 'constantbetadf':
        if 'beta' not in dfa_kwargs:
            dfa_kwargs['beta'] = dfa._beta
        dfac = df.constantbetadf(pot=pot, denspot=denspot, **dfa_kwargs)
        dfac._fE_interp = dfa._fE_interp
    elif dfa.__class__.__name__ == 'osipkovmerrittdf':
        if 'ra' not in dfa_kwargs:
            dfa_kwargs['ra'] = dfa._ra
        dfac = df.osipkovmerrittdf(pot=pot, denspot=denspot, **dfa_kwargs)
        dfac._logfQ_interp = dfa._logfQ_interp
        if validate:
            Es = np.linspace(0.1, 1., 10)
            assert np.allclose(dfac._logfQ_interp(Es), dfa._logfQ_interp(Es)), \
                'logfQ not the same'
    else:
        raise Exception('DF type not recognized')

    if validate:
        rs = np.linspace(0.1,10.,10)
        assert np.allclose(dfac._pot(rs,0.), dfa._pot(rs,0.)), \
            'Potential not the same'
        assert np.allclose(dfac._denspot(rs,0.), dfa._denspot(rs,0.)), \
            'Density potential not the same'

    return dfac
    