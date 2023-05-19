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
import h5py

# ----------------------------------------------------------------------------

# Class for TNG Sublink Trees
class SublinkTree():
    '''SublinkTree:

    Class to handle Sublink trees
    '''

    ## init
    def __init__(self,filename):
        '''__init__:

        Initialize a SublinkTree class instance.

        Args:
            filename - Tree filename, must be hdf5 format

        Returns:
            None
        '''
        assert filename.endswith('.hdf5'), "Tree filename must be hdf5 format"
        self.filename = filename

        with h5py.File(self.filename,'r') as f:
            self.main_branch_ids = np.arange(f['SubhaloID'][0], 
                                             f['MainLeafProgenitorID'][0]+1)
            self.main_branch_mask = np.isin(f['SubhaloID'],self.main_branch_ids)
            self.main_branch_mass = f['Mass'][self.main_branch_mask]
            self.main_branch_snap = f['SnapNum'][self.main_branch_mask]

        # Will be initialized later if needed
        self.main_branch_mass_grow_mask = None
        self.secondary_main_map = None

    ## Methods

    def find_where_main_branch_mass_growing(self,mass_key=None,
        return_mask=False,_check=False):
        '''find_where_main_branch_mass_growing:
        
        Find where the main branch mass is growing. Mask returned ensures 
        that the mass is growing monotonically from point to point. So if the 
        mass decreases for a few snapshots, only include the snapshots where 
        it overtakes the previous maximum.

        For example if mass is:
        [1,3,5,7,9,6,8,10,12]

        the mask would be:
        [T,T,T,T,T,F,F,T,T]

        Skipping 6 and 8 because they are less than the previous maximum.
        
        Args:
            mass_key (str) - If not None, use this key to find the main branch. 
                Otherwise use self.main_branch_mass [default: None]
            return_mask (bool) - If True, return the mask instead of setting it. 
                If mass_key is not None, this is set to True [default: False]
            _check (bool) - If True, check that the main branch mass is
                monotonically increasing [default: False]
        
        Returns:
            main_branch_mass_grow_mask (np.array) - Mask for main branch so 
                that for example 
                self.main_branch_mass[self.main_branch_mass_grow_mask] is a 
                monotonically increasing sequence
        
        Sets:
            main_branch_mass_grow_mask (as above, if not return_mask)
        '''
        if mass_key is not None and return_mask is False:
            print('Setting return_mask=True because mass_key is not default')
            return_mask = True
        if mass_key is None:
            main_branch_mass = self.main_branch_mass
        else:
            main_branch_mass = self.get_property(mass_key)
            main_branch_mass = main_branch_mass[self.main_branch_mask]
        main_branch_mass_grow_rev_mask = np.zeros_like(main_branch_mass, 
            dtype=bool)
        main_branch_mass_rev = main_branch_mass[::-1]
        for j in range(len(main_branch_mass_rev)):
            if j == 0:
                main_branch_mass_grow_rev_mask[j] = True
                main_branch_mass_anchor = main_branch_mass_rev[j]
            elif main_branch_mass_rev[j] > main_branch_mass_anchor:
                    main_branch_mass_grow_rev_mask[j] = True
                    main_branch_mass_anchor = main_branch_mass_rev[j]
            else:
                main_branch_mass_grow_rev_mask[j] = False
        main_branch_mass_grow_mask = main_branch_mass_grow_rev_mask[::-1]
        if _check:
            # Diff is lt/eq 0 since present day is at beginning of array
            assert np.all(np.diff(
                main_branch_mass[main_branch_mass_grow_mask]) <= 0),\
                "Main branch mass is not monotonically increasing"
        if return_mask:
            print('Returning mask, not setting self.main_branch_mass_grow_mask')
            return main_branch_mass_grow_mask
        else:
            self.main_branch_mass_grow_mask = main_branch_mass_grow_mask

    def find_mapping_secondary_to_main_branch(self,**kwargs):
        '''find_mapping_secondary_to_main_branch:

        Find a mapping from all secondary subhalos at all snapshots (redshifts) 
        to the main branch at the same snapshot (redshift)

        Output mapping is same length as the sublink tree excluding the main 
        branch (i.e. applies to f['SubhaloID'][~self.main_branch_mask]) and 
        maps to a point on the main branch 
        (i.e. f['SubhaloID'][self.main_branch_mask])

        Example use is comparing the mass of a secondary branch subhalo with 
        the main branch subhalo mass at the same snapshot (redshift):

        f['Mass'][~self.main_branch_mask][i] 
        ^ mass of the secondary subhalo

        f['Mass'][self.main_branch_mask][self.secondary_main_map[i]] 
        ^ mass of the main branch subhalo at same redshift

        Args:
            None
        
        Returns:
            secondary_main_map (np.array) - Array of indices from secondary 
                branch subhalos to main branch subhalos, length is 
                np.sum(~self.main_branch_mask)
        
        Sets:
            secondary_main_map (np.array) - as above
        '''
        snap = self.get_property('SnapNum')
        secondary_branch_snap = snap[~self.main_branch_mask]
        secondary_main_map = _find_mapping_secondary_to_main_branch(
                self.main_branch_snap,secondary_branch_snap,**kwargs)
        self.secondary_main_map = secondary_main_map

    def find_major_mergers(self,scheme,scheme_kwargs={},
        check_descends_to_main=True):
        '''find_major_mergers:

        Find major mergers from this tree
        
        Args:
            scheme (string) - Scheme to use to identify major mergers.
                Options are:
                    - 'mratio_at_z_ensure_mgrow'
                    - 'mratio_at_tmax'
            scheme_kwargs (dict) - Keyword arguments passed to the scheme
                function
            check_descends_to_main (bool) - Check whether the secondary branch
                that defines the merger actually descends to the main branch 
                at some point (rather than another secondary branch), 
                this is recommended, default True
                    
        Returns:
            major_merger_mlpid (np.array) - MainLeafProgenitorID of the 
                secondary branch that merges into the main branch, constituting 
                a major merger.
            major_merger_mass_ratio (np.array) - Mass ratio of the secondary 
                branch to the main branch at the time of the merger.
        '''
        # Check which scheme to use
        if scheme == 'mratio_at_z_ensure_mgrow':
            _out = self._find_major_mergers_mratio_at_z_ensure_mgrow(
                **scheme_kwargs)
        elif scheme == 'mratio_at_tmax':
            _out = self._find_major_mergers_mratio_at_tmax(**scheme_kwargs)
        major_merger_mlpid,major_merger_mass_ratio,major_merger_mass_ratio_snap\
            = _out

        # Check whether the secondary branch that defines the merger actually
        # descends to the main branch at some point (rather than another
        # secondary branch), this is recommended.
        if check_descends_to_main:
            subhalo_id = self.get_property('SubhaloID')
            descendent_id = self.get_property('DescendantID')
            main_leaf_progenitor_id = self.get_property('MainLeafProgenitorID')
            main_branch_mask = self.main_branch_mask
            descends_to_main_mask = np.zeros(len(major_merger_mlpid),dtype=bool)
            for i,mlpid in enumerate(major_merger_mlpid):
                descends_to_main_mask[i] = self._check_mlpid_descends_to_main(
                    mlpid, main_leaf_progenitor_id,subhalo_id,descendent_id,
                    main_branch_mask)
            major_merger_mlpid = \
                major_merger_mlpid[descends_to_main_mask]
            major_merger_mass_ratio = \
                major_merger_mass_ratio[descends_to_main_mask]
            major_merger_mass_ratio_snap = \
                major_merger_mass_ratio_snap[descends_to_main_mask]

        return major_merger_mlpid,major_merger_mass_ratio,\
            major_merger_mass_ratio_snap

    def _find_major_mergers_mratio_at_z_ensure_mgrow(self, 
        mass_mratio_key='Mass', mass_ratio_threshold=0.1,snapnum_threshold=20):
        '''_find_major_mergers_mratio_at_z_ensure_mgrow:

        Routine to identify major mergers from the tree using the mass ratio of 
        the primary and secondary at any redshift (snapshot) while ensure the 
        mass is growing at that time.

        Used by find_major_mergers() with scheme='mratio_at_z_ensure_mgrow'

        Args:
            mass_mratio_key (str or dict) - Key for the tree dictionary to 
                get the mass used to calculate the mass ratio. For str/dict 
                definitions see get_property(). Default is 'Mass'
            mass_ratio_threshold (float) - Threshold for mass ratio of secondary
                to main branch subhalos, default 0.1
            snapnum_threshold (int) - Threshold for the snapshot number of the
                secondary branch subhalo, default 20
        
        Returns:
            major_merger_mlpid (np.array) - MainLeafProgenitorID of the 
                secondary branch that merges into the main branch, constituting 
                a major merger.
            major_merger_mass_ratio (np.array) - Mass ratio of the secondary 
                branch to the main branch at the time of the merger.
        '''
        mass = self.get_property(mass_mratio_key)
        snapnum = self.get_property('SnapNum')
        main_leaf_progenitor_id = self.get_property('MainLeafProgenitorID')
        main_branch_mask = self.main_branch_mask
        secondary_main_map = self.secondary_main_map
        assert secondary_main_map is not None, \
            'secondary_main_map is None, call find_mapping_secondary_to_main_branch()'
        main_branch_mass_grow_mask = self.main_branch_mass_grow_mask
        assert main_branch_mass_grow_mask is not None, \
            'main_branch_mass_grow_mask is None, call find_main_branch_mass_grow_mask()'

        # Make a mask based on mass ratio, snapshot number, and add in 
        # forced mass growth
        mass_ratio = mass[~main_branch_mask] /\
                     mass[main_branch_mask][secondary_main_map]
        major_mask = (mass_ratio > mass_ratio_threshold) &\
                     (mass_ratio < 1.0) &\
                     (snapnum[~main_branch_mask] > snapnum_threshold) &\
                     (main_branch_mass_grow_mask[secondary_main_map])
        major_merger_mlpid = main_leaf_progenitor_id[~main_branch_mask][major_mask]
        major_merger_mlpid_unique = np.unique(major_merger_mlpid)
        major_merger_mass_ratio_unique = np.zeros(
            len(major_merger_mlpid_unique))
        major_merger_mass_ratio_snap_unique = np.zeros(
            len(major_merger_mlpid_unique),dtype=int)
        # Get the largest mass ratio for each unique MainLeafProgenitorID
        for i,mlpid in enumerate(major_merger_mlpid_unique):
                mlpid_mask = major_merger_mlpid == mlpid
                indx = np.argmax(mass_ratio[major_mask][mlpid_mask])
                major_merger_mass_ratio_unique[i] = \
                    mass_ratio[major_mask][mlpid_mask][indx]
                major_merger_mass_ratio_snap_unique[i] = \
                    snapnum[~main_branch_mask][major_mask][mlpid_mask][indx]
                    
        return major_merger_mlpid_unique, major_merger_mass_ratio_unique,\
            major_merger_mass_ratio_snap_unique

    def _find_major_mergers_mratio_at_tmax(self, mass_tmax_key='Mass',
        mass_mratio_key=None, mass_ratio_threshold=0.1, snapnum_threshold=20):
        '''_find_major_mergers_mratio_at_tmax:

        Routine to identify major mergers from the tree using the mass ratio of 
        the primary and secondary at the time when the secondary achieves its 
        maximum mass.

        Used by find_major_mergers() with scheme='mratio_at_tmax'
        
        Args:
            mass_tmax_key (str or dict) - Key for the tree dictionary to
                get the mass used to calculate the time of maximum mass for 
                the secondary branches. For str/dict definitions see 
                get_property(). Default is 'Mass'
            mass_mratio_key (str or dict) - Key for the tree dictionary to 
                get the mass used to calculate the mass ratio. For str/dict 
                definitions see get_property(). Default is 'Mass' If None, 
                then use the same key as mass_tmax_key, default None
            mass_ratio_threshold (float) - Threshold for mass ratio of secondary
                to main branch subhalos, default 0.1
            snapnum_threshold (int) - Threshold for the snapshot number of the
                secondary branch subhalo, default 20
        
        Returns:
            major_merger_mlpid (np.array) - MainLeafProgenitorID of the 
                secondary branch that merges into the main branch, constituting 
                a major merger.
            major_merger_mass_ratio (np.array) - Mass ratio of the secondary 
                branch to the main branch at the time of the merger.
        '''
        # Get important properties
        mass_tmax = self.get_property(mass_tmax_key)
        if mass_mratio_key is None:
            mass_mratio_key = mass_tmax_key
        mass_mratio = self.get_property(mass_mratio_key)
        snapnum = self.get_property('SnapNum')
        main_leaf_progenitor_id = self.get_property('MainLeafProgenitorID')
        subhalo_id = self.get_property('SubhaloID')
        descendent_id = self.get_property('DescendantID')
        main_branch_mask = self.main_branch_mask
        secondary_main_map = self.secondary_main_map
        assert secondary_main_map is not None, \
            'secondary_main_map is None, call find_mapping_secondary_to_main_branch()'
        # main_branch_mass_grow_mask = self.main_branch_mass_grow_mask
        # assert main_branch_mass_grow_mask is not None, \
        #     'main_branch_mass_grow_mask is None, call find_main_branch_mass_grow_mask()'
        
        # First find all branches that merge to the main branch
        descend_mask = np.isin(descendent_id[~main_branch_mask],
            subhalo_id[main_branch_mask])
        descend_mlpid = main_leaf_progenitor_id[~main_branch_mask][descend_mask]
        
        # Loop over all branches that merge to the main branch
        mass_ratio_mask = np.zeros(len(descend_mlpid),dtype=bool)
        mass_ratio = np.zeros(len(descend_mlpid))
        mass_ratio_snapnum = np.zeros(len(descend_mlpid),dtype=int)
        for i,mlpid in enumerate(descend_mlpid):
            branch_mask = main_leaf_progenitor_id == mlpid
            branch_mass_max_ind = np.argmax(mass_tmax[branch_mask])
            branch_mass_max = mass_mratio[branch_mask][branch_mass_max_ind]
            main_mass_tmax = mass_mratio[main_branch_mask][secondary_main_map][branch_mask[~main_branch_mask]][branch_mass_max_ind]
            mass_ratio[i] = branch_mass_max / main_mass_tmax
            mass_ratio_snapnum[i] = snapnum[branch_mask][branch_mass_max_ind]
            if mass_ratio[i] > mass_ratio_threshold and\
                mass_ratio[i] < 1.0 and\
                snapnum[branch_mask][branch_mass_max_ind] > snapnum_threshold: # and\
                # main_branch_mass_grow_mask[secondary_main_map][branch_mass_max_ind]:
                mass_ratio_mask[i] = True

        major_merger_mass_ratio = mass_ratio[mass_ratio_mask]
        major_merger_mlpid = descend_mlpid[mass_ratio_mask]
        major_merger_mass_ratio_snapnum = mass_ratio_snapnum[mass_ratio_mask]

        return major_merger_mlpid, major_merger_mass_ratio,\
            major_merger_mass_ratio_snapnum

    def _check_mlpid_descends_to_main(self,mlpid,main_leaf_progenitor_id=None,
        subhalo_id=None,descendent_id=None,main_branch_mask=None):
        '''_check_mlpid_descends_to_main:

        Check whether a branch defined by a MainLeafProgenitorID descends onto 
        the main branch at some point, or onto another secondary_branch.

        Args:
        mlpid (int) - MainLeafProgenitorID of the branch to check
        subhalo_id (np.array) - SubhaloID of all subhalos in the tree, 
            len() is length of the whole tree including main branch.
        descendent_id (np.array) - DescendantID of all subhalos in the tree, 
            len() is length of the whole tree including main branch.
        main_leaf_progenitor_id (np.array) - MainLeafProgenitorID of all 
            subhalos in the tree, len() is length of the whole tree including 
            main branch.
        main_branch_mask (np.array) - Boolean mask for main branch subhalos,
            len() is length of the whole tree including main branch.
        '''
        if subhalo_id is None:
            subhalo_id = self.get_property('SubhaloID')
        if descendent_id is None:
            descendent_id = self.get_property('DescendantID')
        if main_leaf_progenitor_id is None:
            main_leaf_progenitor_id = self.get_property('MainLeafProgenitorID')
        if main_branch_mask is None:
            assert self.main_branch_mask is not None, \
                'main_branch_mask is None, call find_main_branch_mask()'
            main_branch_mask = self.main_branch_mask

        mlpid_mask = (main_leaf_progenitor_id == mlpid)
        mlpid_descends_to_main = np.any(np.isin(descendent_id[mlpid_mask],
            subhalo_id[main_branch_mask]))
        return mlpid_descends_to_main

    ## Getters

    def get_property(self,arg,**kwargs):
        '''get_property:
        
        The forward-facing function for getting any quantity from the tree file 
        using the _get_property() function. Allows flexibility of providing 
        a single input argument.
        
        Args:
            arg (str or dict) - If str, then is the key to access the property 
                in the hdf5 file. If dict, then is a dictionary of keyword
                arguments to pass to _get_property().
            kwargs (dict) - Dictionary of keyword arguments to pass to 
                _get_property(). Only used if arg is a str.

        Returns:
            output (unknown) - Output property
        '''
        if isinstance(arg,str):
            return self._get_property(arg,**kwargs)
        elif isinstance(arg,dict):
            return self._get_property(**arg)
        else:
            raise ValueError('arg must be str or dict')

    def _get_property(self,key,ptype=None,numpy_wrap=True):
        '''_get_property:

        Generic wrapper for getting any quantity from the tree file.

        Args:
            key (str) - Key to access the property
            ptype (str) - Particle type to access the property [default None]
            numpy_wrap (bool) - Cast output as a numpy array [default True]
        
        Returns:
            output (unknown) - Output property
        '''
        with h5py.File(self.filename,'r') as f:
            assert key in f.keys(), 'Key not found in hdf5 file'
            output = f[key]
            if ptype is not None:
                indx = self._ptype_to_indx(ptype)
                output = output[:,indx]
            if numpy_wrap:
                output = np.asarray(output)
        return output

    def _ptype_to_indx(self,ptype):
        '''_ptype_to_ind: Query the standard relationship between named 
        particle types and index of 6D array for some fields.'''
        if str(ptype).lower() in ['parttype0','gas','cells']:
            return 0
        elif str(ptype).lower() in ['parttype1','dm','darkmatter']:
            return 1
        elif str(ptype).lower() in ['parttype2']:
            return 2
        elif str(ptype).lower() in ['parttype3','tracers','tracer']:
            return 3
        elif str(ptype).lower() in ['parttype4','star','stars','stellar',
                                    'wind']:
            return 4
        elif str(ptype).lower() in ['parttype5','bh','bhs','blackhole',
                                    'blackholes']:
            return 5
        else:
            raise ValueError('ptype not understood')

def _find_mapping_secondary_to_main_branch(main_branch_snap,
        secondary_branch_snap,_check=False):
    '''_find_mapping_secondary_to_main_branch:

    Find a mapping from all secondary subhalos at all snapshots (redshifts) 
    to the main branch at the same snapshot (redshift)

    Output mapping is same length as the sublink tree excluding the main 
    branch (i.e. applies to f['SubhaloID'][~main_branch_mask]) and 
    maps to a point on the main branch 
    (i.e. f['SubhaloID'][main_branch_mask])

    Example use is comparing the mass of a secondary branch subhalo with 
    the main branch subhalo mass at the same snapshot (redshift):

    f['Mass'][~main_branch_mask][i] 
    ^ mass of the secondary subhalo

    f['Mass'][main_branch_mask][secondary_main_map[i]] 
    ^ mass of the main branch subhalo at same redshift

    Args:
        main_branch_snap (np.array) - Snapshot number of main branch 
            subhalos
        secondary_branch_snap (np.array) - Snapshot number of secondary
            branch subhalos
        _check (bool) - If True, check that the mapping is correct
    
    Returns:
        secondary_main_map (np.array) - Array of indices from secondary 
            branch subhalos to main branch subhalos, length is 
            np.sum(~main_branch_mask)
    '''
    # Efficiently search with sorted arrays
    main_branch_snap_indx = np.argsort(main_branch_snap)
    main_branch_snap_sorted = main_branch_snap[main_branch_snap_indx]
    main_branch_snap_sorted_indx = np.searchsorted(main_branch_snap_sorted,
        secondary_branch_snap)
    secondary_main_map = np.take(main_branch_snap_indx, 
        main_branch_snap_sorted_indx, mode='clip')
    if not np.all(main_branch_snap[secondary_main_map] ==\
                  secondary_branch_snap):
        unique_failed_snaps = np.unique(secondary_branch_snap[
            main_branch_snap[secondary_main_map] != secondary_branch_snap])
        print('Warning: detected mismatch between main and secondary branch '
              'for snapshots: ',unique_failed_snaps, ' - if this is for small '
              'numbered snapshots then it is likely due to the main branch '
              'not reaching snap 0')
    if _check: # This can fail if the main branch doesn't get to snap 0
        assert np.all(main_branch_snap[secondary_main_map] ==\
            secondary_branch_snap)
    return secondary_main_map

# def _find_major_mergers_mratio_at_z_ensure_mgrow(mass,snapnum,main_branch_mask,
#         secondary_main_map,main_branch_mass_grow_mask,mass_ratio_threshold=0.1,
#         snapnum_threshold=20):
#     '''_find_major_mergers_mratio_at_z_ensure_mgrow:

#     Routine to identify major mergers from the tree using the mass ratio of 
#     the mergers at any redshift (snapshot) while ensure the mass is growing 
#     at that time.

#     Used by find_major_mergers() with scheme='mratio_at_z_ensure_mgrow'

#     Args:
#         mass (np.array) - Mass of subhalos, len() is length of tree, including 
#             the main branch
#         snapnum (np.array) - Snapshot number of subhalos, len() is length of
#             tree, including the main branch
#         main_branch_mask (np.array) - Boolean mask for main branch subhalos, 
#             len() is length of tree
#         secondary_main_map (np.array) - Mapping from secondary branches to 
#             main branches at the same redshift, len() is length of 
#             tree[~main_branch_mask]
#         main_branch_mass_grow_mask (np.array) - Boolean mask for main branch
#             to identify where the mass is monotonically increasing, len() is 
#             length of tree[main_branch_mask]
#         mass_ratio_threshold (float) - Threshold for mass ratio of secondary
#             to main branch subhalos, default 0.1
#         snapnum_threshold (int) - Threshold for the snapshot number of the
#             secondary branch subhalo, default 20
    
#     Returns:
#         major_merger_mlpid (np.array) - MainLeafProgenitorID of the 
#             secondary branch that merges into the main branch, constituting a 
#             major merger.
#     '''

#     mass_ratio = mass[~main_branch_mask] /\
#                  mass[~main_branch_mask][secondary_main_map]
#     major_mask = (mass_ratio > mass_ratio_threshold) &\
#                  (mass_ratio < 1.0) &\
#                  (snapnum[~main_branch_mask] > snapnum_threshold) &\
#                  (main_branch_mass_grow_mask[secondary_main_map])
    

                 

def get_mapping_sb_to_mb(mb_snap,sb_snap):
    '''get_mapping_sb_to_mb
    
    Get mapping from secondary branches to main branches at same redshift.
    
    sb_mb_map (output) is the same length as the sublink tree 
    excluding the main branch (i.e. ~ismb), and each element indexes to the 
    element of the main branch at the same redshift / snapshot.
    
    So for example if you were interested in the mass of a secondary subhalo i 
    compared with the mass of the main branch at the same redshift you could do:
    f['Mass'][~ismb][i] <- mass of the secondary subhalo
    f['Mass'][ismb][mb_sb_map[i]] <- mass of the main branch subhalo at same redshift
    
    Args:
        mb_snap (np.array) - Array of main branch snapshots
        sb_snap (np.array) - Array of secondary branch snapshots
    
    Returns:
        sb_mb_map (np.array) - Array of indices from secondary branch subhalos 
            to main branch subhalos 
    '''
    # First sort the main branch snapshots so they can be searched efficiently
    mb_snap_indx = np.argsort(mb_snap)
    mb_snap_sorted = mb_snap[mb_snap_indx]
    mb_snap_sorted_indx = np.searchsorted(mb_snap_sorted,sb_snap)
    # Undo the sorting in the resulting index list
    sb_mb_map = np.take(mb_snap_indx, mb_snap_sorted_indx, mode='clip')
    return sb_mb_map

def get_mask_from_particle_ids(primary_id,secondary_id):
    '''get_mask_from_particle_ids

    Get a mask for a set of particle IDs based on a second set of particle IDs.

    Args:
        primary_id (np.array) - Array of particle IDs representing the primary 
            galaxy, which will be masked such that only particles from the 
            secondary galaxy are included
        secondary_particle_ids (np.array) - Array of particle IDs representing
            the secondary galaxy. This is the merger remnant.

    Returns:
        mask (np.array) - Mask for primary_id such that only particles from the
            secondary galaxy are included
    '''
    # First sort the primary particle IDs so they can be searched efficiently
    primary_id_indx = np.argsort(primary_id)
    primary_id_sorted = primary_id[primary_id_indx]
    primary_id_sorted_indx = np.searchsorted(primary_id_sorted,secondary_id)
    # Undo the sorting in the resulting index list
    mask = np.take(primary_id_indx, primary_id_sorted_indx, mode='clip')
    return mask

def plot_mass_primary_secondary_branch(tree):
    '''plot_mass_primary_secondary_branch:

    Plot the masses of a secondary branch and compare with the primary branch

    Args:
        tree (dict) - Tree dictionary
    '''
    