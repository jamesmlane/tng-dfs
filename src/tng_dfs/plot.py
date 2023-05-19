# ----------------------------------------------------------------------------
#
# TITLE - tree.py
# AUTHOR - James Lane
# PROJECT - tng-dfs
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Utilities for merger trees
'''
__author__ = "James Lane"

### Imports
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from . import util as putil

# ----------------------------------------------------------------------------

def plot_all_merger_traces(tree,mlpids,snap_threshold=20,
    mass_ratio_threshold=0.1,mass_star_ratio_threshold=0.1,                       
    main_leaf_progenitor_id=None, subfind_id=None,mass=None,mass_star=None,
    snap=None):
    '''plot_all_merger_traces:

    Args:
        tree
        mlpids
        snap_threshold (int) - Threshold snapshot (default 20)
        mass_ratio_threshold (float) - Threshold mass ratio (default 0.1)
        mass_star_ratio_threshold (float) - Threshold stellar mass ratio
            (default 0.1)

    Returns:
        fig (matplotlib.pyplot.figure) - figure object
        axs (matplotlib.pyplot.axis) - axis object
    '''
    fontsize=10
    mass_color = 'Black'
    mass_star_color = 'DarkRed'

    # Load the data
    if main_leaf_progenitor_id is None:
        main_leaf_progenitor_id = tree.get_property('MainLeafProgenitorID')
    if subfind_id is None:
        subfind_id = tree.get_property('SubfindID')
    if mass is None:
        mass = tree.get_property('Mass')
    if mass_star is None:
        mass_star = tree.get_property('SubhaloMassType',ptype='stars')
    if snap is None:
        snap = tree.get_property('SnapNum')

    # Make figure and axes
    fig = plt.figure(figsize=(14,4))
    axs = fig.subplots(nrows=1,ncols=3,gridspec_kw={'width_ratios': [3,3,2]}, 
        sharey=True)

    # Determine the secondary to primary mass ratios
    mass = tree.get_property('Mass')
    mass_ratio = mass[~tree.main_branch_mask]/\
        mass[tree.main_branch_mask][tree.secondary_main_map]
    mass_star = tree.get_property('SubhaloMassType',ptype='stars')
    mass_star_ratio = mass_star[~tree.main_branch_mask]/\
        mass_star[tree.main_branch_mask][tree.secondary_main_map]
    snap = tree.get_property('SnapNum')
    secondary_branch_snap = snap[~tree.main_branch_mask]

    # Plot the secondary to primary mass ratios, all points
    axs[0].scatter(np.log10(mass_ratio), secondary_branch_snap,alpha=0.15,
        s=1,color='Black',zorder=1)
    axs[1].scatter(np.log10(mass_star_ratio), secondary_branch_snap,alpha=0.15,
        s=1,color='Black',zorder=1)

    # Show the major merger traces
    for i in range(len(mlpids)):
        mlpid = mlpids[i]
        branch_mask = main_leaf_progenitor_id[~tree.main_branch_mask] == mlpid
        axs[0].plot(np.log10(mass_ratio[branch_mask]), 
            secondary_branch_snap[branch_mask], linewidth=1, zorder=3,
            label=str(mlpid))
        axs[1].plot(np.log10(mass_star_ratio[branch_mask]), 
            secondary_branch_snap[branch_mask], linewidth=1, zorder=3,
            label=str(mlpid))

    # Plot the mass of the primary branch and show where mass increasing
    axs[2].plot(np.log10(mass[tree.main_branch_mask]),
        tree.main_branch_snap, linewidth=4., linestyle='solid', 
        color=mass_color)
    axs[2].plot(np.log10(
            mass[tree.main_branch_mask][tree.main_branch_mass_grow_mask]),
        tree.main_branch_snap[tree.main_branch_mass_grow_mask], linewidth=2., 
        linestyle='solid', color='DodgerBlue')
    axs[2].plot(np.log10(mass_star[tree.main_branch_mask]),
        tree.main_branch_snap, linewidth=4., linestyle='solid', 
        color=mass_star_color)
    axs[2].plot(np.log10(
            mass_star[tree.main_branch_mask][tree.main_branch_mass_grow_mask]),
        tree.main_branch_snap[tree.main_branch_mass_grow_mask], linewidth=2., 
        linestyle='solid', color='DarkOrange')

    # Show shadow boxes across all axes where the mass grows
    main_branch_mass_grow_fill_snap_cents = (
        tree.main_branch_snap[:-1]+tree.main_branch_snap[1:])/2.
    dz = np.diff(main_branch_mass_grow_fill_snap_cents)
    main_branch_mass_grow_fill_snap_edges = np.concatenate((
        np.atleast_1d(main_branch_mass_grow_fill_snap_cents[0] - dz[0]/2.),
        main_branch_mass_grow_fill_snap_cents,
        np.atleast_1d(main_branch_mass_grow_fill_snap_cents[-1] + dz[-1]/2.),
    ))
    #main_branch_mass_grow_fill_mask = np.zeros_like(
    #    main_branch_mass_grow_fill_z_cents, dtype=bool)
    main_branch_mass_grow_fill_mask = np.zeros_like(
        main_branch_mass_grow_fill_snap_edges, dtype=bool)
    main_branch_mass_grow_fill_mask[:] = True
    # main_branch_mass_star_grow_fill_mask = np.zeros_like(
    #     main_branch_mass_grow_fill_z_cents, dtype=bool)
    main_branch_mass_star_grow_fill_mask = np.zeros_like(
        main_branch_mass_grow_fill_snap_edges, dtype=bool)
    main_branch_mass_star_grow_fill_mask[:] = True
    
    main_branch_mass_grow_fill_mask[
        np.where(~tree.main_branch_mass_grow_mask)[0]] = False
    main_branch_mass_grow_fill_mask[
        np.where(~tree.main_branch_mass_grow_mask)[0]+1] = False
    
    main_branch_mass_star_grow_mask = tree.find_where_main_branch_mass_growing(
        mass_key={'key':'SubhaloMassType','ptype':'stars'}
    )
    main_branch_mass_star_grow_fill_mask[
        np.where(~main_branch_mass_star_grow_mask)[0]] = False
    main_branch_mass_star_grow_fill_mask[
        np.where(~main_branch_mass_star_grow_mask)[0]+1] = False
    
    for i in range(len(axs)):
        axs[i].fill_betweenx(1+main_branch_mass_grow_fill_snap_edges,
            axs[i].get_xlim()[0], axs[i].get_xlim()[1],
            where=~main_branch_mass_grow_fill_mask, alpha=0.1, 
            facecolor='Black', edgecolor='None')
        axs[i].fill_betweenx(1+main_branch_mass_grow_fill_snap_edges,
            axs[i].get_xlim()[0], axs[i].get_xlim()[1],
            where=~main_branch_mass_star_grow_fill_mask, alpha=0.1, 
            facecolor='Red', edgecolor='None')

    # Add lines
    axs[0].axhline(snap_threshold, linestyle='dashed', color='Grey',
        zorder=3)
    axs[0].axvline(np.log10(mass_ratio_threshold), linestyle='dashed', 
        color='Grey', zorder=3)
    axs[0].axvline(0, linestyle='dashed', color='Grey', linewidth=0.5, zorder=3)
    axs[1].axhline(snap_threshold, linestyle='dashed', color='Grey',
        zorder=3)
    axs[1].axvline(np.log10(mass_star_ratio_threshold), linestyle='dashed', 
        color='Grey', zorder=3)
    axs[1].axvline(0, linestyle='dashed', color='Grey', linewidth=0.5, zorder=3)
    axs[2].axhline(snap_threshold, linestyle='dashed', color='Grey',
        zorder=3)

    # Add text and scale axes
    axs[0].set_xlabel('Mass ratio', fontsize=fontsize)
    axs[0].set_ylabel('Snapshot', fontsize=fontsize)
    axs[1].set_xlabel('Stellar mass ratio', fontsize=fontsize)
    axs[2].set_xlabel(r'log($M_{\rm P}$)-10', fontsize=fontsize)
    fig.suptitle('Primary z0 SID '+str(tree.main_branch_ids[0]), 
        fontsize=fontsize)

    # Add exclusion rectangles
    select_rec = mpl.patches.Rectangle((np.log10(mass_ratio_threshold),
        snap_threshold),0-np.log10(mass_ratio_threshold),
        100-snap_threshold, alpha=0.2, color='Grey', zorder=3)
    axs[0].add_artist(select_rec)
    select_stars_rec = mpl.patches.Rectangle((np.log10(mass_star_ratio_threshold),
        snap_threshold),0-np.log10(mass_star_ratio_threshold),
        100-snap_threshold, alpha=0.2, color='Grey', zorder=3)
    axs[1].add_artist(select_stars_rec)

    # Add legend
    axs[0].legend(fontsize=12-len(mlpids)/2)
    # axs[1].plot(np.log10(mb_mass),mb_snap, color='DodgerBlue', 
    #             linewidth=3.)
    # axs[1].plot(np.log10(mb_mass)[where_mb_grow], mb_snap[where_mb_grow],
    #             color='Red', linewidth=1.)
    # axs[1].scatter(np.ones_like(mb_snap_grow)*np.min(np.log10(mb_mass))-0.25,
    #                 mb_snap_grow, color='Red', s=4)
    
    fig.subplots_adjust(wspace=0.025)

    return fig, axs
