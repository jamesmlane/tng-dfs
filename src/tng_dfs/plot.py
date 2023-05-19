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

def plot_merger_information(tree,mlpid,threshold_mass_ratio=0.1,
    threshold_mass_star_ratio=0.1, threshold_snap=20, 
    main_leaf_progenitor_id=None, subfind_id=None,mass=None,mass_star=None,
    snap=None):
    '''plot_merger_information:

    Args:
        tree (tng_dfs.tree.SublinkTree) - merger tree object
        mlpid (int) - main leaf progenitor id of the secondary branch to plot
        tree data (np.ndarray) - data from tree, will be loaded if not provided:
            - main_leaf_progenitor_id (MainLeafProgenitorID)
            - subfind_id (SubfindID)
            - mass (Mass)
            - mass_star (SubhaloMassType, ptype='stars')
            - snap (SnapNum)

    Returns:
        fig (matplotlib.pyplot.figure) - figure object
        ax (matplotlib.pyplot.axis) - axis object
    '''
    fontsize=10
    mass_color = 'Black'
    mass_star_color = 'Red'

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
    fig = plt.figure(figsize=(6,5))
    axs = fig.subplots(4,1,sharex=True)

    # Mask the secondary branch
    branch_mask = main_leaf_progenitor_id[~tree.main_branch_mask] == mlpid
    
    # Determine the mass ratio
    mass_ratio = mass[~tree.main_branch_mask][branch_mask]/\
        mass[tree.main_branch_mask][tree.secondary_main_map][branch_mask]
    
    # Determine redshift of maximum secondary mass, stellar mass
    z_tmax = putil.snapshot_to_redshift(
        snap[~tree.main_branch_mask][branch_mask]
        [np.argmax(mass[~tree.main_branch_mask][branch_mask])])
    z_tmax_star = putil.snapshot_to_redshift(
        snap[~tree.main_branch_mask][branch_mask]
        [np.argmax(mass_star[~tree.main_branch_mask][branch_mask])])

    # Plot the main branch
    axs[0].plot(1+putil.snapshot_to_redshift(snap[tree.main_branch_mask]),
                np.log10(mass[tree.main_branch_mask]), 
                linestyle='solid', color=mass_color)
    axs[0].plot(1+putil.snapshot_to_redshift(snap[tree.main_branch_mask]),
                np.log10(mass_star[tree.main_branch_mask]),
                linestyle='solid', color=mass_star_color)

    # Plot the secondary branch
    axs[0].plot(1+putil.snapshot_to_redshift(
            snap[~tree.main_branch_mask][branch_mask]),
        np.log10(mass[~tree.main_branch_mask][branch_mask]), 
        linestyle='dashed', color=mass_color)
    axs[0].plot(1+putil.snapshot_to_redshift(
            snap[~tree.main_branch_mask][branch_mask]),
        np.log10(mass_star[~tree.main_branch_mask][branch_mask]),
        linestyle='dashed', color=mass_star_color)
    
    # Plot the log mass ratio
    axs[1].plot(1+putil.snapshot_to_redshift(
            snap[~tree.main_branch_mask][branch_mask]),
        mass_ratio, color=mass_color
        )
    mass_ratio_star = mass_star[~tree.main_branch_mask][branch_mask]/\
        mass_star[tree.main_branch_mask][tree.secondary_main_map][branch_mask]
    axs[1].plot(1+putil.snapshot_to_redshift(
            snap[~tree.main_branch_mask][branch_mask]),
        mass_ratio_star, color=mass_star_color
        )
    
    # Plot the linear mass ratio
    axs[2].plot(1+putil.snapshot_to_redshift(
            snap[~tree.main_branch_mask][branch_mask]),
        mass_ratio, color=mass_color
        )
    mass_ratio_star = mass_star[~tree.main_branch_mask][branch_mask]/\
        mass_star[tree.main_branch_mask][tree.secondary_main_map][branch_mask]
    axs[2].plot(1+putil.snapshot_to_redshift(
            snap[~tree.main_branch_mask][branch_mask]),
        mass_ratio_star, color=mass_star_color
        )

    # Plot the mass of the primary, showing where the mass grows
    axs[3].plot(1+putil.snapshot_to_redshift(tree.main_branch_snap),
        np.log10(tree.main_branch_mass), color='Black', linewidth=3.
        )
    axs[3].plot(1+putil.snapshot_to_redshift(
            tree.main_branch_snap[tree.main_branch_mass_grow_mask]), 
        np.log10(tree.main_branch_mass[tree.main_branch_mass_grow_mask]), 
        color='DodgerBlue', linewidth=1.
        )
    # axs[3].scatter(1+putil.snapshot_to_redshift(
    #     tree.main_branch_snap[tree.main_branch_mass_grow_mask]),
    #     np.ones_like(tree.main_branch_snap
    #         [tree.main_branch_mass_grow_mask])*
    #         np.min(np.log10(tree.main_branch_mass
    #             [tree.main_branch_mass_grow_mask]))-0.25,
    #     color='DodgerBlue', marker='o', s=10, zorder=3
    #     )
    # axs[3].scatter(1+putil.snapshot_to_redshift(
    #     tree.main_branch_snap[~tree.main_branch_mass_grow_mask]),
    #     np.ones_like(tree.main_branch_snap
    #         [~tree.main_branch_mass_grow_mask])*
    #         np.min(np.log10(tree.main_branch_mass
    #             [tree.main_branch_mass_grow_mask]))-0.25,
    #     color='Red', marker='o', s=10, zorder=4 
    #     )
    
    # Show shadow boxes across all axes where the mass grows
    main_branch_mass_grow_fill_z_cents = (
        putil.snapshot_to_redshift(tree.main_branch_snap)[:-1]+
        putil.snapshot_to_redshift(tree.main_branch_snap)[1:])/2.
    dz = np.diff(main_branch_mass_grow_fill_z_cents)
    main_branch_mass_grow_fill_z_edges = np.concatenate((
        np.atleast_1d(main_branch_mass_grow_fill_z_cents[0] - dz[0]/2.),
        main_branch_mass_grow_fill_z_cents,
        np.atleast_1d(main_branch_mass_grow_fill_z_cents[-1] + dz[-1]/2.),
    ))
    #main_branch_mass_grow_fill_mask = np.zeros_like(
    #    main_branch_mass_grow_fill_z_cents, dtype=bool)
    main_branch_mass_grow_fill_mask = np.zeros_like(
        main_branch_mass_grow_fill_z_edges, dtype=bool)
    main_branch_mass_grow_fill_mask[:] = True
    # main_branch_mass_star_grow_fill_mask = np.zeros_like(
    #     main_branch_mass_grow_fill_z_cents, dtype=bool)
    main_branch_mass_star_grow_fill_mask = np.zeros_like(
        main_branch_mass_grow_fill_z_edges, dtype=bool)
    main_branch_mass_star_grow_fill_mask[:] = True
    
    main_branch_mass_grow_fill_mask[
        np.where(~tree.main_branch_mass_grow_mask)[0]] = False
    main_branch_mass_grow_fill_mask[
        np.where(~tree.main_branch_mass_grow_mask)[0]+1] = False

    # for i in range(len(main_branch_mass_grow_fill_mask)):
    #     if i < len(main_branch_mass_grow_fill_mask)-1:
    #         if not tree.main_branch_mass_grow_mask[i]:
    #             main_branch_mass_grow_fill_mask[i] = False
    #     if i > 0:
    #         if not tree.main_branch_mass_grow_mask[i]:
    #             main_branch_mass_grow_fill_mask[i] = False
    #     if not tree.main_branch_mass_grow_mask[i] or \
    #         not tree.main_branch_mass_grow_mask[i+1]:
    #         main_branch_mass_grow_fill_mask[i] = False
    
    main_branch_mass_star_grow_mask = tree.find_where_main_branch_mass_growing(
        mass_key={'key':'SubhaloMassType','ptype':'stars'}
    )
    main_branch_mass_star_grow_fill_mask[
        np.where(~main_branch_mass_star_grow_mask)[0]] = False
    main_branch_mass_star_grow_fill_mask[
        np.where(~main_branch_mass_star_grow_mask)[0]+1] = False

    # for i in range(len(main_branch_mass_star_grow_fill_mask)):
    #     if not main_branch_mass_star_grow_mask[i] or \
    #         not main_branch_mass_star_grow_mask[i+1]:
    #         main_branch_mass_star_grow_fill_mask[i] = False
    
    for i in range(len(axs)):
        axs[i].fill_between(1+main_branch_mass_grow_fill_z_edges,
            axs[i].get_ylim()[0], axs[i].get_ylim()[1], 
            where=~main_branch_mass_grow_fill_mask, alpha=0.1, 
            facecolor='Black', edgecolor='None')
        axs[i].fill_between(1+main_branch_mass_grow_fill_z_edges,
            axs[i].get_ylim()[0], axs[i].get_ylim()[1], 
            where=~main_branch_mass_star_grow_fill_mask, alpha=0.1, 
            facecolor='Red', edgecolor='None')
    
    # Add lines
    for k in range(len(axs)):
        axs[k].axvline(1+z_tmax, linestyle='dotted', color='Black', 
            linewidth=1.0, zorder=3)
        axs[k].axvline(1+z_tmax_star, linestyle='dotted', color='Red', 
            linewidth=1.0, zorder=3)
    axs[1].axhline(0.1, linestyle='dotted', color='Black', linewidth=1., 
        zorder=3)
    axs[2].axhline(0.1, linestyle='dotted', color='Black', linewidth=1.,
        zorder=3)
    axs[2].axhline(1., linestyle='solid', color='Black', linewidth=1.,
        zorder=3)

    # Add text and scale axes
    axs[0].set_ylabel(r'$\log_{10}(M)$-10', fontsize=fontsize)
    axs[0].set_xscale('log')
    axs[0].set_title('Primary z0 SID '+str(tree.main_branch_ids[0])+
                        ', Secondary MLPID '+str(mlpid), 
                        fontsize=fontsize)
    axs[0].set_xlim([-1,25])
    axs[1].set_ylabel(r'$M_{\rm s}/M_{\rm p}$', fontsize=fontsize)
    axs[1].set_yscale('log')
    axs[2].set_ylim(-0.1,1.1)
    axs[2].set_ylabel(r'$M_{\rm s}/M_{\rm p}$', fontsize=fontsize)
    axs[3].set_ylabel(r'$\log(M_{\rm p})-10$', fontsize=fontsize)
    axs[3].set_xlabel(r'$1+z$', fontsize=fontsize)
    
    fig.subplots_adjust(hspace=0.1)
    return fig,axs

