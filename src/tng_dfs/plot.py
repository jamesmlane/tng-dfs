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
import matplotlib
from galpy import potential
from astropy import units as apu
import scipy.stats

from . import util as putil
from . import kinematics as pkin

# ----------------------------------------------------------------------------

def plot_all_merger_traces(tree,mlpids,snap_threshold=20,
    mass_ratio_threshold=0.1,mass_star_ratio_threshold=0.1,                       
    main_leaf_progenitor_id=None, subfind_id=None,mass=None,mass_star=None,
    snap=None):
    '''plot_all_merger_traces:

    Args:
        tree (Tree) - Tree object
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
    mass_color = 'Navy'
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
        color=mass_color, label='DM')
    axs[2].plot(np.log10(
            mass[tree.main_branch_mask][tree.main_branch_mass_grow_mask]),
        tree.main_branch_snap[tree.main_branch_mass_grow_mask], linewidth=2., 
        linestyle='solid', color='DodgerBlue', label='DM growing')
    axs[2].plot(np.log10(mass_star[tree.main_branch_mask]),
        tree.main_branch_snap, linewidth=4., linestyle='solid', 
        color=mass_star_color, label='Stars')
    axs[2].plot(np.log10(
            mass_star[tree.main_branch_mask][tree.main_branch_mass_grow_mask]),
        tree.main_branch_snap[tree.main_branch_mass_grow_mask], linewidth=2., 
        linestyle='solid', color='DarkOrange', label='Stars growing')

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
    # select_rec = mpl.patches.Rectangle((np.log10(mass_ratio_threshold),
    #     snap_threshold),0-np.log10(mass_ratio_threshold),
    #     100-snap_threshold, alpha=0.2, color='Grey', zorder=3)
    # axs[0].add_artist(select_rec)
    # select_stars_rec = mpl.patches.Rectangle((np.log10(mass_star_ratio_threshold),
    #     snap_threshold),0-np.log10(mass_star_ratio_threshold),
    #     100-snap_threshold, alpha=0.2, color='Grey', zorder=3)
    # axs[1].add_artist(select_stars_rec)

    # Add legend
    axs[0].legend(fontsize=12-len(mlpids)/2)
    axs[2].legend(fontsize=8)
    # axs[1].plot(np.log10(mb_mass),mb_snap, color='DodgerBlue', 
    #             linewidth=3.)
    # axs[1].plot(np.log10(mb_mass)[where_mb_grow], mb_snap[where_mb_grow],
    #             color='Red', linewidth=1.)
    # axs[1].scatter(np.ones_like(mb_snap_grow)*np.min(np.log10(mb_mass))-0.25,
    #                 mb_snap_grow, color='Red', s=4)
    
    fig.subplots_adjust(wspace=0.025)

    return fig, axs

def plot_merger_information(tree,mlpid,threshold_mass_ratio=0.1,
    threshold_mass_star_ratio=0.1, threshold_snap=20, snap_mass_ratio=None,
    snap_mass_star_ratio=None,
    main_leaf_progenitor_id=None, subfind_id=None,mass=None,mass_star=None,
    snap=None):
    '''plot_merger_information:

    Plot the merger information for a given main leaf progenitor id.

    If both snapshot variables are None then will calculate the snapshots using 
    the maximum mass of each component.

    Args:
        tree (tng_dfs.tree.SublinkTree) - merger tree object
        mlpid (int) - main leaf progenitor id of the secondary branch to plot
        snap_mass_ratio (int) - Snapshot where the dark matter mass ratio was 
            recorded [default: None]
        snap_mass_star_ratio (int) - Snapshot where the stellar mass ratio was 
            recorded [default: None]
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
    if snap_mass_ratio is not None:
        z_tmax = putil.snapshot_to_redshift(snap_mass_ratio)
    else:
        z_tmax = None
    if snap_mass_star_ratio is not None:
        z_tmax_star = putil.snapshot_to_redshift(snap_mass_star_ratio)
    else:
        z_tmax_star = None
    
    if snap_mass_ratio is None and snap_mass_star_ratio is None:
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
        if z_tmax is not None:
            axs[k].axvline(1+z_tmax, linestyle='dotted', color='Black', 
                linewidth=1.0, zorder=3)
        if z_tmax_star is not None:
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

def plot_merger_scheme_comparison(data_1,data_2,plot_mlpids=False,
    log_mass_ratio=False):
    '''plot_merger_scheme_comparison:
    
    Compare the secondary/primary mass ratios of the mergers identified using 
    two different schemes, and possible two different sets of particles.
    
    Args:

    Returns:
        fig (matplotlib.figure.Figure): Figure object
        axs (list of matplotlib.axes._subplots.AxesSubplot): List of axes 
            objects    
    '''
    # Unpack data
    pz0id_1_raw, smlpid_1_raw, mratio_1_raw, mratio_snap_1_raw = data_1
    pz0id_2_raw, smlpid_2_raw, mratio_2_raw, mratio_snap_2_raw = data_2

    # Prepare figure and axes
    fig = plt.figure(figsize=(10,10))
    gs = mpl.gridspec.GridSpec(5,5)
    ax1 = fig.add_subplot(gs[2:,0:3])
    ax2 = fig.add_subplot(gs[1,0:3])
    ax3 = fig.add_subplot(gs[0,0:3])
    ax4 = fig.add_subplot(gs[2:,-2])
    ax5 = fig.add_subplot(gs[2:,-1])

    # Collate the results from the different trees
    smlpid_1 = np.concatenate(smlpid_1_raw)
    smlpid_2 = np.concatenate(smlpid_2_raw)
    smlpid_unique = np.unique(np.concatenate((smlpid_1,smlpid_2)))
    pz0id_1 = np.zeros(len(smlpid_1))
    pz0id_2 = np.zeros(len(smlpid_2))
    mratio_1 = np.concatenate(mratio_1_raw)
    mratio_2 = np.concatenate(mratio_2_raw)
    mratio_snap_1 = np.concatenate(mratio_snap_1_raw)
    mratio_snap_2 = np.concatenate(mratio_snap_2_raw)

    assert len(pz0id_1_raw) == len(pz0id_2_raw)
    for i in range(len(pz0id_1_raw)):
        pz0id_1[smlpid_1==smlpid_1_raw[i]] = pz0id_1_raw[i]
        pz0id_2[smlpid_2==smlpid_2_raw[i]] = pz0id_2_raw[i]

    # Get masks showing where major mergers are in common, and where they are unique
    # to one method
    common_mask_1 = np.in1d(smlpid_1,smlpid_2)
    common_mask_2 = np.in1d(smlpid_2,smlpid_1)
    # unique_mask_1 = np.in1d(smlpid_1,
    #     smlpid_unique[~np.in1d(smlpid_unique,smlpid_1)])
    # unique_mask_2 = np.in1d(smlpid_2,
    #     smlpid_unique[~np.in1d(smlpid_unique,smlpid_2)])

    # Plot the results
    ax1.scatter(mratio_1[common_mask_1],mratio_2[common_mask_2], s=24, 
        marker='o', edgecolors='Black', facecolors='Red')

    # Plot scatter of mass ratio vs snapshot of accretion in each method
    ax2.scatter(mratio_1[common_mask_1],mratio_snap_1[common_mask_1], s=24,
        marker='o', edgecolors='Black', facecolors='Red', zorder=3)
    ax2.scatter(mratio_1[~common_mask_1],mratio_snap_1[~common_mask_1], s=24,
        marker='s', edgecolors='Black', facecolors='LightGrey', zorder=2)

    ax4.scatter(mratio_snap_2[common_mask_2], mratio_2[common_mask_2], s=24,
        marker='o', edgecolors='Black', facecolors='Red', zorder=3)
    ax4.scatter(mratio_snap_2[~common_mask_2], mratio_2[~common_mask_2], s=24,
        marker='s', edgecolors='Black', facecolors='LightGrey', zorder=2)

    if plot_mlpids:
        for i in range(len(smlpid_1[common_mask_1])):
            ax1.text(mratio_1[common_mask_1][i],mratio_2[common_mask_2][i],
                str(smlpid_1[common_mask_1][i])[-5:],fontsize=3,color='Grey',
                zorder=4)
        for i in range(len(smlpid_1)):
            ax2.text(mratio_1[i],mratio_snap_1[i],str(smlpid_1[i])[-5:],
                fontsize=3,color='Grey',zorder=4)
        for i in range(len(smlpid_2)):
            ax4.text(mratio_snap_2[i],mratio_2[i],str(smlpid_2[i])[-5:],
                fontsize=3,color='Grey',zorder=4)

    # Plot the marginalized histograms for each method
    nbins = 5000
    hrange = [0.1,500]
    common_color = 'LightGrey'
    uncommon_color = 'Red'
    ax3.hist(mratio_1, bins=nbins, range=hrange, histtype='step', 
        color=common_color, linewidth=4, linestyle='solid', zorder=2)
    ax3.hist(mratio_1[common_mask_1], bins=nbins, range=hrange, histtype='step',
        color=uncommon_color, linewidth=2, linestyle='dashed', zorder=3)

    ax5.hist(mratio_2, bins=nbins, range=hrange, histtype='step', 
        color=common_color, linewidth=4, linestyle='solid', zorder=2, 
        orientation='horizontal')
    ax5.hist(mratio_2[common_mask_2], bins=nbins, range=hrange, histtype='step',
        color=uncommon_color, linewidth=2, linestyle='dashed', zorder=3,
        orientation='horizontal')

    # Fiduxial line, and a faint dashed line at mratio=1
    if log_mass_ratio:
        ax1.plot([0,10000],[0,10000], color='Black', linestyle='dashed')
        ax1.plot([0.,1.,1.],[1.,1.,0.], color='Grey', linestyle='dashed',
            linewidth=1.)
    else:
        ax1.plot([-1,2],[-1,2], color='Black', linestyle='dashed')
        ax1.plot([-1.,1.,1.],[1.,1.,-1.], color='Grey', linestyle='dashed',
            linewidth=1.)

    # Set limits and labels
    # mratio_max = np.max([np.max(mratio_1),np.max(mratio_2)])
    if log_mass_ratio:
        xlim = [0.04,1000]
        ylim = [0.04,1000]
        for ax in [ax1,ax2,ax3]:
            ax.set_xscale('log')
        for ax in [ax1,ax4,ax5]:
            ax.set_yscale('log')
    else:
        xlim = [-0.1,1.1]
        ylim = [-0.1,1.1]
    slim = [0,100]

    # show the points which are outside the limits, but which are common
    for i in range(len(mratio_1[common_mask_1])):
        if mratio_1[i] > xlim[1]:
            ax1.scatter(0.95*xlim[1], mratio_2[common_mask_2][i], s=24, 
                marker='o', edgecolor='Black', facecolor='Red')
            ax1.arrow(0.96*xlim[1], mratio_2[common_mask_2][i], 
                0.04*xlim[1], 0, length_includes_head=True, 
                head_width=0.02*xlim[1], color='Black')
            if plot_mlpids:
                ax1.text(0.95*xlim[1], mratio_2[common_mask_2][i],
                    str(smlpid_1[common_mask_1][i])[-5:],fontsize=3,color='Grey',
                    zorder=4)
        if mratio_2[i] > ylim[1]:
            ax1.scatter(mratio_1[common_mask_1][i], 0.95*ylim[1], s=24, 
                marker='o', edgecolor='Black', facecolor='Red')
            ax1.arrow(mratio_1[common_mask_1][i], 0.96*ylim[1], 
                0, 0.04*ylim[1], length_includes_head=True, 
                head_width=0.02*ylim[1], color='Black')
            if plot_mlpids:
                ax1.text(mratio_1[common_mask_1][i], 0.95*ylim[1],
                    str(smlpid_1[common_mask_1][i])[-5:],fontsize=3,color='Grey',
                    zorder=4)
    

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel(r'$M_{\rm ratio}$ (dark matter)')
    ax1.set_ylabel(r'$M_{\rm ratio}$ (stars)')

    # snap vs mratio for #1
    ax2.set_xlim(xlim)
    ax2.set_ylim(slim)
    ax2.set_ylabel(r'Snap $N_{\rm mratio}$')
    ax2.tick_params(labelbottom=False)

    # Marginalized histogram for #1
    ax3.set_xlim(xlim)
    ax3.set_ylabel(r'$N$')
    ax3.tick_params(labelbottom=False)

    # snap vs mratio for #2
    ax4.set_ylim(ylim)
    ax4.set_xlim(slim)
    ax4.set_xlabel(r'Snap $N_{\rm mratio}$')
    ax4.tick_params(labelleft=False)

    # Marginalized histogram for #2
    ax5.set_ylim(ylim)
    ax5.set_xlabel(r'$N$')
    ax5.tick_params(labelleft=False)

    return fig,[ax1,ax2,ax3,ax4,ax5]

def plot_jeans_diagnostics(Js,rs,qs,adf=None,r_range=None):
    '''plot_jeans_diagnostics:
    
    Plot many quantities related to the Jeans equations.

    Args:
        Js (array) - Array of Jeans equation terms, shape len(rs)
        rs (array) - Radii of bin centers where Js were calculated
        qs (array) - List of kinematic quantities from 
            kinematics.calculate_spherical_jeans_quantities()
        adf (galpy.df object) - Distribution function representing the sample 
            being plotted. Will be used to plot force, velocity dispersions, 
            density profile for comparison.
        r_range (list) - Radial range for plotting, only considered if adf 
            is not None, default None
    
    Returns:
        fig (matplotlib figure) - Figure
        axs (array of matplotlib axes) - Axes
    '''

    plot_jeans_sigmas = True
    data_color = 'Black'
    data_linewidth = 2.
    truth_color = 'Red'
    
    _has_adf = adf is not None
    if _has_adf:
        assert r_range is not None
        pot = adf._pot
        denspot = adf._denspot
    else:
        plot_jeans_sigmas = False

    percfunc =  lambda x: np.percentile(np.atleast_2d(x), [16,50,84], axis=0)

    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(nrows=4,ncols=3)
    axs = np.array([fig.add_subplot(gs[:2,0]),
                    fig.add_subplot(gs[0,1]),
                    fig.add_subplot(gs[1,1]),
                    fig.add_subplot(gs[:2,2]),
                    fig.add_subplot(gs[2:,0]),
                    fig.add_subplot(gs[2,1]),
                    fig.add_subplot(gs[3,1]),
                    fig.add_subplot(gs[2:,2])
                    ])
    # axs = fig.subplots(nrows=2,ncols=3).flatten()

    # J in the first panel
    lJ,mJ,uJ = percfunc(Js)
    axs[0].plot(rs, mJ, color=data_color, linewidth=data_linewidth)
    axs[0].fill_between(rs, lJ, uJ, color='Black', alpha=0.25)
    axs[0].axhline(0, color='Black', linestyle='--', linewidth=0.5)
    axs[0].set_xlim(0,50)
    axs[0].set_xlabel('r [kpc]')
    # Assume that J is normalized
    axs[0].set_ylabel(r'$J / (\nu \bar{v_{r}^{2}} / r)$')

    # Density in the second upper panel
    lnu,mnu,unu = percfunc(qs[2])
    axs[1].plot(rs, mnu, color=data_color, linewidth=data_linewidth)
    if _has_adf:
        denspot_dens = potential.evaluateDensities(denspot,rs*apu.kpc,0)
        denspot_norm = potential.mass(denspot,r_range[1]*apu.kpc)-\
                       potential.mass(denspot,r_range[0]*apu.kpc)
        denspot_dens = (denspot_dens/(denspot_norm)).to(apu.kpc**-3).value
        axs[1].plot(rs, denspot_dens,#*mnu[0]/denspot_dens[0], 
            color=truth_color, linestyle='--')
    axs[1].fill_between(rs, unu, lnu, color='Black', alpha=0.25)
    # axs[1].set_xlim(0,50)
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    # axs[1].set_xlabel(r'r [kpc]')
    axs[1].set_ylabel(r'$\nu$')

    # Delta density in the second lower panel
    if _has_adf:
        # dnu = (qs[2] - denspot_dens*mnu[0]/denspot_dens[0])/(denspot_dens*mnu[0]/denspot_dens[0])
        dnu = (qs[2]-denspot_dens)/denspot_dens
        ldnu,mdnu,udnu = percfunc(dnu)
        axs[2].plot(rs, mdnu, color=data_color, linewidth=data_linewidth)
        axs[2].fill_between(rs, udnu, ldnu, color='Black', alpha=0.25)
        axs[2].axhline(0, color='Black', linestyle='--', linewidth=0.5)
        axs[2].set_xscale('log')
        axs[2].set_xlabel(r'r [kpc]')
        axs[2].set_ylabel(r'$\Delta \nu$ [fractional]')

    # Beta in the third panel
    beta = 1 - (qs[4]+qs[5])/(2*qs[3])
    lbeta,mbeta,ubeta = percfunc(beta)
    axs[3].plot(rs, mbeta, color=data_color, linewidth=data_linewidth)
    axs[3].fill_between(rs, ubeta, lbeta, color='Black', alpha=0.25)
    axs[3].axhline(0, color='Black', linestyle='--', linewidth=0.5)
    axs[3].set_xlim(0,50)
    axs[3].set_xlabel(r'r [kpc]')
    axs[3].set_ylabel(r'$\beta$')

    # Radial velocity dispersions in the fourth panel, polar and azimuthal 
    # in the fifth upper/lower panels
    colors = ['DodgerBlue','Crimson','DarkOrange']
    v2_names = [r'$\bar{v_{r}^{2}}$',
                r'$\bar{v_{\phi}^{2}}$',
                r'$\bar{v_{\theta}^{2}}$',]
    for i in range(3):
        for j in range(3):
            lv2,mv2,uv2 = percfunc(qs[j+3])
            if i == j:
                axs[i+4].plot(rs, mv2, color=colors[j], 
                    linewidth=data_linewidth+2, zorder=2)
                axs[i+4].fill_between(rs, uv2, lv2, color=colors[i], alpha=0.25, 
                    zorder=1)
                if plot_jeans_sigmas:
                    mom = [0]*len(rs)
                    for k in range(len(rs)):
                        if i == 0:
                            mom[k] = adf.vmomentdensity(rs[k]*apu.kpc,2,0)
                        elif i in [1,2]:
                            mom[k] = adf.vmomentdensity(rs[k]*apu.kpc,0,2)/2
                        mom[k] /= adf.vmomentdensity(rs[k]*apu.kpc,0,0)
                        mom[k] = mom[k].to_value(apu.km**2/apu.s**2)
                    mom = np.asarray(mom)
                    axs[i+4].plot(rs, mom, color='Black', alpha=1.,
                        linestyle='--', linewidth=1., zorder=3)
            else:
                axs[i+4].plot(rs, mv2, color=colors[j], alpha=1., 
                    linestyle='--', linewidth=1., zorder=3)
        axs[i+4].set_xlim(0,50)
        if i in [0,2]:
            axs[i+4].set_xlabel(r'r [kpc]')
        axs[i+4].set_ylabel(v2_names[i])
        axs[i+4].set_yscale('log')
    
    # dphi/dr in the sixth panel
    ldphidr,mdphidr,udphidr = percfunc(qs[1])
    axs[7].plot(rs, mdphidr, color=data_color, linewidth=data_linewidth)
    axs[7].fill_between(rs, udphidr, ldphidr, color='Black', alpha=0.25)
    if _has_adf:
        negpf = -potential.evaluaterforces(pot,rs*apu.kpc,0).\
            to(apu.km**2/apu.s**2/apu.kpc).value
        axs[7].plot(rs, negpf, color=truth_color, linestyle='--')
    axs[7].set_xlim(0,50)
    axs[7].set_xlabel(r'r [kpc]')
    axs[7].set_ylabel(r'$\mathrm{d}\Phi/\mathrm{d}r$')
    axs[7].set_yscale('log')
    
    return fig,axs

def plot_dens_vdisp_beta(orbs,E=None,pe=None,fig=None,axs=None):
    '''plot_dens_vdisp_beta:

    Plots the density, velocity dispersions, anisotropy profile

    Args:
        orbs (list): list of orbits to plot
        E (float): total energy of the orbits in km^2/s^2
        pe (array): potential energy of the orbits in km^2/s^2
        fig,axs (matplotlib.pyplot figure and axes): if not None, plots on
            these axes
    
    Returns:
        fig,axs (matplotlib.pyplot figure and axes): figure and axes of the
            plot
    '''
    label_fs = 12
    v_colors = ['MediumBlue','Red','ForestGreen']
    v_names = [r'$r$',r'$\phi$',r'$\theta$']
    v_subscripts = [r'_{r}',r'_{\phi}',r'_{\theta}']
    _percentiles = [16,50,84]
    alpha_bg = 0.25
    plot_ELz = True
    _has_pe = pe is not None
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_bad(color='white')

    percfunc =  lambda x: np.percentile(np.atleast_2d(x), [16,50,84], axis=0)

    if fig is None or axs is None:
        fig = plt.figure(figsize=(9,18))
        # gso = fig.add_gridspec(1,4)
        # gs1 = gso[0,0].subgridspec(3,1,hspace=0.3)
        # gs2 = gso[0,1:].subgridspec(1,3,wspace=0.3)
        # gs3 = gso[1:,1].subgridspec(3,1,hspace=0.1)
        # gs4 = gso[1:,2].subgridspec(3,1,hspace=0.1)
        # gs5 = gso[1:,3].subgridspec(3,1,hspace=0.1)
        # ax1 = fig.add_subplot(gs1[0,0])
        # ax2 = fig.add_subplot(gs1[1,0])
        # ax3 = fig.add_subplot(gs1[2,0])
        # ax4 = fig.add_subplot(gs2[0,0])
        # ax5 = fig.add_subplot(gs2[0,1])
        # ax6 = fig.add_subplot(gs2[0,2])
        # c2_axs = [fig.add_subplot(gs3[i,0]) for i in range(3)]
        # c3_axs = [fig.add_subplot(gs4[i,0]) for i in range(3)]
        # c4_axs = [fig.add_subplot(gs5[i,0]) for i in range(3)]

        # gso = fig.add_gridspec(9,4)
        # gs1 = gso[:,0].subgridspec(3,1,hspace=0.3)
        # gs2 = gso[:3,1:].subgridspec(1,3,wspace=0.3)
        # gs3 = gso[3:,1].subgridspec(3,1,hspace=0.1)
        # gs4 = gso[3:,2].subgridspec(3,1,hspace=0.1)
        # gs5 = gso[3:,3].subgridspec(3,1,hspace=0.1)
        # ax1 = fig.add_subplot(gs1[0,0])
        # ax2 = fig.add_subplot(gs1[1,0])
        # ax3 = fig.add_subplot(gs1[2,0])
        # ax4 = fig.add_subplot(gs2[0,0])
        # ax5 = fig.add_subplot(gs2[0,1])
        # ax6 = fig.add_subplot(gs2[0,2])
        # c2_axs = [fig.add_subplot(gs3[i,0]) for i in range(3)]
        # c3_axs = [fig.add_subplot(gs4[i,0]) for i in range(3)]
        # c4_axs = [fig.add_subplot(gs5[i,0]) for i in range(3)]

        # axs = fig.subplots(4,4)
        # ax1 = axs[0,0] # First column, density
        # ax2 = axs[1,0] # First column, number count
        # ax3 = axs[2,0] # First column, beta
        # ax4 = axs[3,0] # First column, J residual
        # ax5 = axs[0,1]
        # ax6 = axs[0,2]
        # ax7 = axs[0,3]
        # c2_axs = [axs[i+1,1] for i in range(3)]
        # c3_axs = [axs[i+1,2] for i in range(3)]
        # c4_axs = [axs[i+1,3] for i in range(3)]
        # # axs[3,0].axis('off')

        axs = fig.subplots(nrows=6,ncols=3)
        ax1,ax2,ax3 = axs[0] # First row: density, number count, beta
        r2_axs = axs[1] # Second column: J, J1, J2
        ax7,ax8,ax9 = axs[2] # Third column: E-Lz, vr-vT, vR-vperp
        r4_axs = axs[3] # Fourth column: mean spherical velocities
        r5_axs = axs[4] # Fifth column: mean spherical velocity dispersions
        r6_axs = axs[5] # Sixth column: mean spherical velocity squares

    # Bin kinematic properties / bootstrapping
    nbin = 6
    nbs = 20
    r_min = 0
    r_max = 100
    bin_edges = np.linspace(r_min,r_max,num=nbin+1)
    bin_cents = (bin_edges[1:]+bin_edges[:-1])/2

    nu = np.zeros((nbs,nbin))
    counts = np.zeros((nbs,nbin))
    mv = np.zeros((3,nbs,nbin))
    v2 = np.zeros((3,nbs,nbin))
    sv = np.zeros((3,nbs,nbin))
    Js = np.zeros((3,nbs,nbin)) # total J, J term 1, J term 2
    wJs = np.zeros((2,nbs)) # weighted average J, weighted average square J
    for i in range(nbs):
        # Make the bootstrap sample
        _orbs_bs_indx = np.random.randint(0,len(orbs),len(orbs))
        _orbs_bs = orbs[_orbs_bs_indx]
        if _has_pe:
            _pe_bs = pe[_orbs_bs_indx]
        rs = _orbs_bs.r(use_physical=True).to(apu.kpc).value
        for j in range(len(bin_cents)):
            bin_mask = (rs>=bin_edges[j]) & (rs<bin_edges[j+1])
            n_in_bin = np.sum( bin_mask )
            bin_vol = 4*np.pi/3*(bin_edges[j+1]**3-bin_edges[j]**3)
            nu[i,j] = n_in_bin/bin_vol
            counts[i,j] = n_in_bin
            _vr = _orbs_bs.vr(use_physical=True).to(apu.km/apu.s).value[bin_mask]
            _vp = _orbs_bs.vT(use_physical=True).to(apu.km/apu.s).value[bin_mask]
            _vt = _orbs_bs.vtheta(use_physical=True).to(apu.km/apu.s).value[bin_mask]
            for k,v in enumerate([_vr,_vp,_vt]):
                mv[k,i,j] = np.mean(v)
                v2[k,i,j] = np.mean(v**2.)
                sv[k,i,j] = np.std(v)
        if _has_pe:
            _J,_rs,_qs,_J1,_J2 = pkin.calculate_spherical_jeans(_orbs_bs,
                pe=_pe_bs, n_bin=nbin, r_range=[r_min,r_max], 
                return_kinematics=True, return_terms=True)
            Js[0,i] = _J
            Js[1,i] = _J1
            Js[2,i] = _J2
            _wJ,_wJ2 = pkin.calculate_weighted_average_J(_J,_rs,dens=_qs[2])
            wJs[0,i] = _wJ
            wJs[1,i] = _wJ2

    beta = 1.-(v2[1]+v2[2])/(v2[0]*2.)

    ## First row, first panel: density
    lnu,mnu,hnu = np.percentile(nu,_percentiles,axis=0)
    ax1.plot(bin_cents, mnu, color='Black')
    ax1.fill_between(bin_cents, lnu, hnu, color='Black', alpha=0.2)
    ax1.set_xlabel(r'r [kpc]', fontsize=label_fs)
    ax1.set_ylabel(r'density [kpc$^{-3}$]', fontsize=label_fs)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlim(bin_cents[0]-1,120)
    # Add fiducials
    alphas = [1,2,3.01,4,5,6]
    # alpha_labels = [r'$\alpha=$'+str(a) for a in [1,2,3,4,5,6]]
    alpha_labels = [str(a) for a in [1,2,3,4,5,6]]
    denspot_rs = np.arange(np.min(bin_cents),100.,1.)
    for j in range(len(alphas)):
        denspot = potential.PowerSphericalPotential(alpha=alphas[j],ro=8,vo=220)
        fdens = potential.evaluateDensities(denspot, denspot_rs*apu.kpc, 0, 
            use_physical=True).value
        fdens *= (mnu[0]/fdens[0])
        ax1.plot(denspot_rs, fdens, color='Blue', linestyle='dashed', 
            alpha=0.5)
        ax1.annotate(alpha_labels[j], xy=(denspot_rs[-1],fdens[-1]*1.5),
            color='Blue', fontsize=8, va='center', alpha=1.0)
    
    ## First row, second panel: Number counts vs radius
    lnu,mnu,hnu = np.percentile(counts,_percentiles,axis=0)
    ax2.plot(bin_cents, mnu, marker='o', color='Black')
    ax2.fill_between(bin_cents, lnu, hnu, color='Black', alpha=0.2)
    ax2.set_xlabel(r'r [kpc]', fontsize=label_fs)
    ax2.set_ylabel(r'Number count', fontsize=label_fs)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim(bin_cents[0]-1,120)
    ax2.text(0.05, 0.3, r'$N_{tot} = $'+str(len(orbs)), transform=ax2.transAxes, 
        fontsize=label_fs-6)
    ax2.text(0.05, 0.2, (r'$N(r < '+str(r_max)+' \mathrm{kpc}) =$ '
        +str(len(orbs[orbs.r() < r_max*apu.kpc]))), fontsize=label_fs-6,
        transform=ax2.transAxes)
    ax2.text(0.05, 0.1, (r'$f(r < '+str(r_max)+' \mathrm{kpc}) =$ '
        +str(round(100*len(orbs[orbs.r() < r_max*apu.kpc])/len(orbs),2))+'\%'), 
        fontsize=label_fs-6, transform=ax2.transAxes)

    ## First row, third panel: beta
    lbeta,mbeta,hbeta = np.percentile(beta,_percentiles,axis=0)
    ax3.plot(bin_cents, mbeta, color='Black')
    ax3.fill_between(bin_cents, lbeta, hbeta, color='Black', alpha=0.2)
    ax3.set_xlabel(r'r [kpc]', fontsize=label_fs)
    ax3.set_ylabel(r'$\beta$', fontsize=label_fs)
    ax3.axhline(0, color='Black', linestyle='dashed')

    ## Second row, J total and individual terms
    if _has_pe:
        J_labels = [r'$J_{0}$',r'$J_{1}$',r'$J_{2}$']
        for i in range(3):
            lJ,mJ,hJ = np.percentile(Js[i],_percentiles,axis=0)
            r2_axs[i].plot(bin_cents, mJ, color='Black')
            r2_axs[i].fill_between(bin_cents, lJ, hJ, color='Black', alpha=0.2)
            r2_axs[i].set_xlabel(r'r [kpc]', fontsize=label_fs)
            r2_axs[i].set_ylabel(J_labels[i], fontsize=label_fs)
            r2_axs[i].axhline(0, color='Black', linestyle='dashed')
            r2_axs[i].set_xlim(bin_cents[0]-5,bin_cents[-1]+25)
        lwJ,mwJ,hwJ = np.percentile(wJs[0],_percentiles,axis=0)
        r2_axs[0].errorbar(bin_cents[-1]+10, mwJ, xerr=2, yerr=(hwJ-lwJ)/2, 
            color='DodgerBlue', capsize=0.)
        lwJ2,mwJ2,hwJ2 = np.percentile(wJs[1],_percentiles,axis=0)
        r2_axs[0].errorbar(bin_cents[-1]+20, mwJ2, xerr=2, yerr=(hwJ2-lwJ2)/2, 
            color='Crimson', capsize=0.)
    else:
        r2_axs[0].text(0.5, 0.5, 'No potential energy', 
            transform=r2_axs[0].transAxes, fontsize=label_fs-2, ha='center', 
            va='center')

    ## Third row, first panel: Optionally E-Lz
    if E is not None and plot_ELz:
        Lz = orbs.Lz(use_physical=True).to(apu.kpc*apu.km/apu.s).value
        if isinstance(E,apu.Quantity):
            E = E.to(apu.km**2/apu.s**2).value
        # Plot binned data
        Lz_range = [-3,3]
        E_range = [-5,-1]
        H, xedges, yedges = np.histogram2d(Lz/1e3, E/1e5, bins=[45,30],
            range=[Lz_range,E_range])
        H = np.rot90(H)
        H = np.flipud(H)
        Hmasked = np.ma.masked_where(H==0,H)
        cmap = plt.cm.get_cmap('viridis')
        cmap.set_bad(color='white')
        ax7.pcolormesh(xedges,yedges,Hmasked,cmap=cmap)        
        ax7.set_xlabel(r'Lz [$10^{3}$ kpc km/s]', fontsize=label_fs)
        ax7.set_ylabel(r'E [$10^{5}$ km$^{2}$/s$^{2}$]', fontsize=label_fs)
        # ax7.set_ylim(E_range[0],E_range[1])
        ax7.set_xlim(Lz_range[0],Lz_range[1])
        ax7.axvline(0, linestyle='dashed', linewidth=1., color='Grey')
    else:
        ax7.axis('off') 

    ## Third row, second panel: cylindrical vR vs vT
    vR = orbs.vr(use_physical=True).to(apu.km/apu.s).value
    vT = orbs.vT(use_physical=True).to(apu.km/apu.s).value
    # Plot binned data
    vR_range = [-400,400]
    vT_range = [-400,400]
    H, xedges, yedges = np.histogram2d(vR, vT, bins=[30,30],
        range=[vR_range,vT_range])
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0,H)
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_bad(color='white')
    ax8.pcolormesh(xedges,yedges,Hmasked,cmap=cmap)
    ax8.set_xlabel(r'$v_{R}$ [km/s]', fontsize=label_fs)
    ax8.set_ylabel(r'$v_{T}$ [km/s]', fontsize=label_fs)
    ax8.set_ylim(vR_range[0],vR_range[1])
    ax8.set_xlim(vT_range[0],vT_range[1])
    ax8.axhline(0, linestyle='dashed', linewidth=1., color='Grey')
    ax8.axvline(0, linestyle='dashed', linewidth=1., color='Grey')
    
    ## Third row, third panel panel: radial vs. perpendicular velocity
    vr = orbs.vr(use_physical=True).to(apu.km/apu.s).value
    vp = orbs.vT(use_physical=True).to(apu.km/apu.s).value
    vt = orbs.vtheta(use_physical=True).to(apu.km/apu.s).value
    vperp = np.sqrt(vp**2.+vt**2.)
    # Plot binned data
    vr_range = [-200,200]
    vperp_range = [0,400]
    H, xedges, yedges = np.histogram2d(vr, vperp, bins=[30,30],
        range=[vr_range,vperp_range])
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0,H)
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_bad(color='white')
    ax9.pcolormesh(xedges,yedges,Hmasked,cmap=cmap)
    ax9.set_xlabel(r'$v_{r}$ [km/s]', fontsize=label_fs)
    ax9.set_ylabel(r'$v_{\perp}$ [km/s]', fontsize=label_fs)
    ax9.set_ylim(vperp_range[0],vperp_range[1])
    ax9.set_xlim(vr_range[0],vr_range[1])
    ax9.axvline(0, linestyle='dashed', linewidth=1., color='Grey')

    ## Fourth row: mean velocity profiles
    for i in range(3):
        for j in range(3):
            lmv,mmv,hmv = np.percentile(mv[j],_percentiles,axis=0)
            if i == j:
                r4_axs[i].plot(bin_cents, mmv, color=v_colors[j], 
                    zorder=5)#, label=v_names[j])
                r4_axs[i].fill_between(bin_cents, lmv, hmv, color=v_colors[j], 
                    alpha=0.3, zorder=4)
            else:
                r4_axs[i].plot(bin_cents, mmv, color=v_colors[j], 
                    alpha=alpha_bg, zorder=3)#, label=v_names[j])
                # c3_axs[j].fill_between(bin_cents, lsv, hsv, color=v_colors[i],
                #     alpha=0.1, zorder=2)
        r4_axs[i].axhline(0., color='Black', linestyle='dashed', 
            linewidth=1, zorder=1)
        r4_axs[i].set_ylabel(r'$\bar{v'+v_subscripts[i]+'}$ [km/s]', 
            fontsize=label_fs)
        if i == 2:
            r4_axs[i].set_xlabel(r'r [kpc]', fontsize=label_fs)
        else:
            r4_axs[i].tick_params(labelbottom=False)
    for i in range(3):
        r4_axs[0].plot([],[],color=v_colors[i], label=r'$v'+v_subscripts[i]+'$')
    r4_axs[0].legend(loc='best', fontsize=12, frameon=False)

    # Third column: velocity dispersions / v2
    for i in range(3):
        for j in range(3):
            lsv,msv,hsv = np.percentile(sv[j],_percentiles,axis=0)
            if i == j:
                r5_axs[i].plot(bin_cents, msv, color=v_colors[j], 
                    label=v_names[j], zorder=5)
                r5_axs[i].fill_between(bin_cents, lsv, hsv, color=v_colors[j],
                    alpha=0.3, zorder=4)
            else:
                r5_axs[i].plot(bin_cents, msv, color=v_colors[j], 
                    label=v_names[j], alpha=alpha_bg, zorder=3)
                # r5_axs[j].fill_between(bin_cents, lsv, hsv, color=v_colors[i],
                #     alpha=0.1, zorder=2)
        r5_axs[i].set_ylabel(r'$\sigma'+v_subscripts[i]+'$ [km/s]', 
            fontsize=label_fs)
        # if i == 2:
        r5_axs[i].set_xlabel(r'r [kpc]', fontsize=label_fs)
        # else:
        #     r5_axs[i].tick_params(labelbottom=False)

    # Fourth column: velocity mean square
    v2_sqrt = True
    for i in range(3):
        for j in range(3):
            if v2_sqrt:
                lv2,mv2,hv2 = np.percentile(np.sqrt(v2[j]),_percentiles,axis=0)
            else:
                lv2,mv2,hv2 = np.percentile(v2[j],_percentiles,axis=0)
            if i == j:
                r6_axs[i].plot(bin_cents, mv2, color=v_colors[j], 
                    label=v_names[j], zorder=5)
                r6_axs[i].fill_between(bin_cents, lv2, hv2, color=v_colors[j],
                    alpha=0.3, zorder=4)
            else:
                r6_axs[i].plot(bin_cents, mv2, color=v_colors[j], 
                    label=v_names[j], alpha=alpha_bg, zorder=3)
                # r6_axs[j].fill_between(bin_cents, lv2, hv2, color=v_colors[i],
                #     alpha=0.1, zorder=3)
        if v2_sqrt:
            r6_axs[i].set_ylabel((r'$\bigg[ \bar{v^{2}'+v_subscripts[i]+
                r'} \bigg]^{1/2}$ [km/s]'), fontsize=label_fs)
        else:
            r6_axs[i].set_ylabel(r'$\bar{v^{2}'+v_subscripts[i]+r'}$ [km/s]', 
                fontsize=label_fs)
        # if i == 2:
        r6_axs[i].set_xlabel(r'r [kpc]', fontsize=label_fs)
        # else:
        #     r6_axs[i].tick_params(labelbottom=False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig,axs