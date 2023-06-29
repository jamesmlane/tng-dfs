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
import warnings
import h5py
import scipy.interpolate
import os

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
            self.main_branch_star_mass = (f['SubhaloMassType'][:,4]
                [self.main_branch_mask])
            self.main_branch_snap = f['SnapNum'][self.main_branch_mask]
            self.main_branch_mlpid = f['MainLeafProgenitorID'][0]

        # Will be initialized later if needed
        self.main_branch_mass_grow_mask = None
        self.secondary_main_map = None
        self.find_mapping_secondary_to_main_branch()
        self.all_main_map = None
        self.find_mapping_all_to_main_branch()

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
            warnings.warn(
                'Setting return_mask=True because mass_key is not default')
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
            warnings.warn(
                'Returning mask, not setting self.main_branch_mass_grow_mask')
            return main_branch_mass_grow_mask
        else:
            self.main_branch_mass_grow_mask = main_branch_mass_grow_mask

    def get_main_branch_mass_growing_interpolator(self,mass_key=None):
        '''get_main_branch_mass_growing_interpolator:
        
        Get an interpolator for the main branch mass that is monotonically
        increasing. In gaps between the previous maximum and a new maximum, 
        linearly interpolate the mass.

        Args:
            mass_key (str) - If not None, use this key get the main branch 
                mass and accompanying mass. Otherwise use the default mass
                (dark matter), default None

        Returns:
            interp (scipy.interpolate.interp1d) - Interpolator for the main 
                branch mass that is monotonically increasing
        '''
        if mass_key is not None:
            main_branch_mass_growing_mask = \
                self.find_where_main_branch_mass_growing(mass_key=mass_key,
                return_mask=False,_check=True)
            mass = self.get_property(mass_key)[self.main_branch_mask]
        else:
            assert self.main_branch_mass_grow_mask is not None
            main_branch_mass_growing_mask = self.main_branch_mass_grow_mask
            mass = self.main_branch_mass
        
        return scipy.interpolate.interp1d(
            self.main_branch_snap[main_branch_mass_growing_mask],
            mass[main_branch_mass_growing_mask],
            kind = 'linear', bounds_error=False, fill_value='extrapolate')
        
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

    def find_mapping_all_to_main_branch(self):
        '''find_mapping_all_to_main_branch:
        
        Find a mapping from all subhalos (not just secondaries) to the main 
        branch. Very similar to secondary_main_map, but also includes the 
        main branch. So if the main branch is the first N elements of the 
        tree, for example, (which it should always be?) then the all_main_map
        is just:
        
        np.concatenate([np.arange(N),secondary_main_map]))
        
        But it will be checked to ensure that the mapping is correct

        Args:
            None
        
        Returns:
            all_main_map (np.array) - Array of indices from all subhalos to 
                main branch subhalos, length is len(self)
        '''
        if self.secondary_main_map is None:
            raise RuntimeError('''Must run 
                find_mapping_secondary_to_main_branch() first''')
        all_main_map =  np.concatenate([np.where(self.main_branch_mask)[0],
            self.secondary_main_map])
        self.all_main_map = all_main_map

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
            return_as_class (bool) - Return as a list of TreeMajorMerger
                objects, default False
                    
        Returns:
            major_merger_mlpid (np.array) - MainLeafProgenitorID of the 
                secondary branch that merges into the main branch, constituting 
                a major merger.
            major_merger_mass_ratio (np.array) - Mass ratio of the secondary 
                branch to the main branch at the time of the merger.
            major_merger_mass_ratio_snap (np.array) - Snapshot (redshift) of
                the merger.
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

    # def _find_major_mergers_mratio_at_z_ensure_mgrow(self, 
    #     mass_mratio_key='Mass', mass_ratio_threshold=0.1,snapnum_threshold=20):
    #     '''_find_major_mergers_mratio_at_z_ensure_mgrow:

    #     Routine to identify major mergers from the tree using the mass ratio of 
    #     the primary and secondary at any redshift (snapshot) while ensure the 
    #     mass is growing at that time.

    #     Used by find_major_mergers() with scheme='mratio_at_z_ensure_mgrow'

    #     Args:
    #         mass_mratio_key (str or dict) - Key for the tree dictionary to 
    #             get the mass used to calculate the mass ratio. For str/dict 
    #             definitions see get_property(). Default is 'Mass'
    #         mass_ratio_threshold (float) - Threshold for mass ratio of secondary
    #             to main branch subhalos, default 0.1
    #         snapnum_threshold (int) - Threshold for the snapshot number of the
    #             secondary branch subhalo, default 20
        
    #     Returns:
    #         major_merger_mlpid (np.array) - MainLeafProgenitorID of the 
    #             secondary branch that merges into the main branch, constituting 
    #             a major merger.
    #         major_merger_mass_ratio (np.array) - Mass ratio of the secondary 
    #             branch to the main branch at the time of the merger.
    #     '''
    #     mass = self.get_property(mass_mratio_key)
    #     snapnum = self.get_property('SnapNum')
    #     main_leaf_progenitor_id = self.get_property('MainLeafProgenitorID')
    #     main_branch_mask = self.main_branch_mask
    #     secondary_main_map = self.secondary_main_map
    #     assert secondary_main_map is not None, \
    #         'secondary_main_map is None, call find_mapping_secondary_to_main_branch()'
    #     main_branch_mass_grow_mask = self.main_branch_mass_grow_mask
    #     assert main_branch_mass_grow_mask is not None, \
    #         'main_branch_mass_grow_mask is None, call find_main_branch_mass_grow_mask()'

    #     # Make a mask based on mass ratio, snapshot number, and add in 
    #     # forced mass growth
    #     mass_ratio = mass[~main_branch_mask] /\
    #                  mass[main_branch_mask][secondary_main_map]
    #     major_mask = (mass_ratio > mass_ratio_threshold) &\
    #                  (mass_ratio < 1.0) &\
    #                  (snapnum[~main_branch_mask] > snapnum_threshold) &\
    #                  (main_branch_mass_grow_mask[secondary_main_map])
    #     major_merger_mlpid = main_leaf_progenitor_id[~main_branch_mask][major_mask]
    #     major_merger_mlpid_unique = np.unique(major_merger_mlpid)
    #     major_merger_mass_ratio_unique = np.zeros(
    #         len(major_merger_mlpid_unique))
    #     major_merger_mass_ratio_snap_unique = np.zeros(
    #         len(major_merger_mlpid_unique),dtype=int)
    #     # Get the largest mass ratio for each unique MainLeafProgenitorID
    #     for i,mlpid in enumerate(major_merger_mlpid_unique):
    #             mlpid_mask = major_merger_mlpid == mlpid
    #             indx = np.argmax(mass_ratio[major_mask][mlpid_mask])
    #             major_merger_mass_ratio_unique[i] = \
    #                 mass_ratio[major_mask][mlpid_mask][indx]
    #             major_merger_mass_ratio_snap_unique[i] = \
    #                 snapnum[~main_branch_mask][major_mask][mlpid_mask][indx]
                    
    #     return major_merger_mlpid_unique, major_merger_mass_ratio_unique,\
    #         major_merger_mass_ratio_snap_unique

    def _find_major_mergers_mratio_at_tmax(self, mass_tmax_key='Mass',
        mass_mratio_key=None, mass_ratio_threshold=0.1, snapnum_threshold=20,
        mask_main_branch_mass_growing=False, 
        use_interpolated_main_branch_mass=False):
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
            mask_main_branch_mass_growing (bool) - If True, then mask out 
                snapshots where the main branch mass is not increasing, and 
                determine the mass ratio using another snapshot, default False
            use_interpolated_main_branch_mass (bool) - If True, then use 
                linearly interpolated main branch mass where the mass is not 
                increasing, default False
        
        Returns:
            major_merger_mlpid (np.array) - MainLeafProgenitorID of the 
                secondary branch that merges into the main branch, constituting 
                a major merger.
            major_merger_mass_ratio (np.array) - Mass ratio of the secondary 
                branch to the main branch at the time of the merger.
        '''
        if mask_main_branch_mass_growing and use_interpolated_main_branch_mass:
            print('mask_main_branch_mass_growing and ' +\
                'use_interpolated_main_branch_mass are both True, ' +\
                'use_interpolated_main_branch_mass will be ignored')
            use_interpolated_main_branch_mass = False

        # Get important properties
        mass_tmax = self.get_property(mass_tmax_key)
        if mass_mratio_key is None:
            mass_mratio_key = mass_tmax_key
        mass_mratio = self.get_property(mass_mratio_key)
        if use_interpolated_main_branch_mass:
            mass_mratio_interp = self.get_main_branch_mass_growing_interpolator(
                mass_key=mass_mratio_key)
            main_branch_mass_mratio_interpolated = mass_mratio_interp(
                self.main_branch_snap)
        snapnum = self.get_property('SnapNum')
        main_leaf_progenitor_id = self.get_property('MainLeafProgenitorID')
        subhalo_id = self.get_property('SubhaloID')
        descendent_id = self.get_property('DescendantID')
        main_branch_mask = self.main_branch_mask
        secondary_main_map = self.secondary_main_map
        all_main_map = self.all_main_map
        assert secondary_main_map is not None, \
            'secondary_main_map is None, call find_mapping_secondary_to_main_branch()'
        
        # Recalculate the main branch mass growing mask to make sure it uses 
        # the same mass as mass ratio calculations
        main_branch_mass_grow_mask = self.find_where_main_branch_mass_growing(
            mass_key=mass_mratio_key, return_mask=True)
        
        # First find all branches that merge to the main branch
        descend_mask = np.isin(descendent_id[~main_branch_mask],
            subhalo_id[main_branch_mask])
        descend_mlpid = main_leaf_progenitor_id[~main_branch_mask][descend_mask]
        
        # Loop over all branches that merge to the main branch
        major_merger_mask = np.zeros(len(descend_mlpid),dtype=bool)
        mass_ratio = np.zeros(len(descend_mlpid))
        mass_ratio_snapnum = np.zeros(len(descend_mlpid),dtype=int)
        for i,mlpid in enumerate(descend_mlpid):
            branch_mask = main_leaf_progenitor_id == mlpid
            if mask_main_branch_mass_growing:
                branch_mask &= (main_branch_mass_grow_mask[all_main_map])
            if np.sum(branch_mask) == 0:
                major_merger_mask[i] = False
                continue
            branch_mass_max_ind = np.argmax(mass_tmax[branch_mask])
            branch_mass_max = mass_mratio[branch_mask][branch_mass_max_ind]
            if use_interpolated_main_branch_mass:
                main_mass_tmax = (main_branch_mass_mratio_interpolated
                    [secondary_main_map][branch_mask[~main_branch_mask]]
                    [branch_mass_max_ind])
            else:
                main_mass_tmax = (mass_mratio[main_branch_mask]
                    [secondary_main_map][branch_mask[~main_branch_mask]]
                    [branch_mass_max_ind])
            mass_ratio[i] = branch_mass_max / main_mass_tmax
            mass_ratio_snapnum[i] = snapnum[branch_mask][branch_mass_max_ind]
            branch_valid = (mass_ratio[i] > mass_ratio_threshold) &\
                (snapnum[branch_mask][branch_mass_max_ind] > snapnum_threshold)
            major_merger_mask[i] = branch_valid

        major_merger_mass_ratio = mass_ratio[major_merger_mask]
        major_merger_mlpid = descend_mlpid[major_merger_mask]
        major_merger_mass_ratio_snapnum = mass_ratio_snapnum[major_merger_mask]

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
            
        Returns:
            mlpid_descends_to_main (bool) - True if the branch descends onto the
                main branch at some point, False otherwise.
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
                indx = util.ptype_to_indx(ptype)
                output = output[:,indx]
            if numpy_wrap:
                output = np.asarray(output)
        return output

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
        warning_msg = (
            'Warning: detected mismatch between main and secondary branch '
            'for snapshots: '+str(unique_failed_snaps)+' - if this is for small '
            'numbered snapshots then it is likely due to the main branch '
            'not reaching snap 0')
        warnings.warn(warning_msg)
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


# ----------------------------------------------------------------------------

# Some classes for handling important information about the primary subhalo
# and major mergers into the primary, specifically data products.

class TreeInfo(object):
    '''TreeInfo:

    Superclass to hold information about a sublink tree, specifically various 
    mergers and the availability of snapshots/data products.

    Also provides access to methods common to TreePrimary and TreeMajorMerger.
    '''

    def __init__(self,tree_filename):#,**kwargs):
        '''__init__:

        Initialise the class.
        
        Args:
            tree_filename (str) - Absolute filename of the sublink tree file
        
        Raises:
            IOError: If the tree file does not exist
        '''
        if not os.path.isfile(tree_filename):
            raise IOError('Tree file {} does not exist'.format(tree_filename))
        self.tree_filename = tree_filename
        # self.attribute = kwargs.get('attribute',None)

    def get_tree(self):
        '''get_tree:

        Get the sublink tree which includes this primary subhalo

        Returns:
            tree (SublinkTree) - SublinkTree object
        '''
        return SublinkTree(self.tree_filename)

    def get_cutout_filename(self,data_dir,snapnum,subfind_id=None):
        '''get_cutout_filename:
        
        Using either the snapshot number or the subfind ID, get the filename
        of the cutout file for this primary subhalo.

        Args:
            data_dir (str) - Data directory path as per the config file
            snapnum (int) - Snapshot number of the cutout file, optional
            subfind_id (int) - Subfind ID of the cutout file, optional
        
        Returns:
            fname (str) - Absolute filename of the cutout file
        
        Raises:
            ValueError: If subfind_id is not provided and the class does not 
                have subfind_id/snapnum loaded as attributes from the tree file
            IOError: If the cutout file does not exist
        '''
        if data_dir[-1] != '/': data_dir += '/'
        if subfind_id is None:
            if hasattr(self,'subfind_id') and hasattr(self,'snapnum'):
                subfind_id = self.subfind_id[self.snapnum==snapnum][0]
            else:
                raise ValueError('Must provide either subfind_id or have '
                    'subfind_id/snapnum loaded as attributes from the tree '
                    'file (provide tree_filename)')
        snap_path = data_dir+'cutouts/snap_'+str(snapnum)+'/'
        fname = snap_path+'cutout_'+str(subfind_id)+'.hdf5'
        if not os.path.isfile(fname):
            raise IOError('File '+fname+' does not exist')
        return fname
    
    def get_unique_particle_ids(self,ptype,data_dir=None,snapnum=None):
        '''get_unique_particle_ids:

        Get all the unique particle IDs for this subhalo across all snapshots.

        Args:
            ptype (str) - Particle type to get unique particle IDs for
            data_dir (str) - Data directory path as per the config file
            snapnum (int or np.array) - Specify the snapshot numbers to get 
                unique particle IDs for. If None, get all snapshots.
        
        Returns:
            unique_particle_ids (np.array) - Array of unique particle IDs
        '''
        if data_dir is None:
            raise ValueError('Must currently provide data_dir')
        if snapnum is None:
            snapnum = self.snapnum
        else:
            snapnum = np.atleast_1d(snapnum)
            assert np.all(np.in1d(snapnum,self.snapnum)), \
                'snapnum must be in self.snapnum'
        unique_particle_ids = np.array([],dtype=int)
        for snap in snapnum:
            # print(snap)
            # Get the cutout file for this snapshot
            fname = self.get_cutout_filename(data_dir,snap)
            co = cutout.TNGCutout(fname)
            # Get the unique particle IDs for this snapshot
            try:
                pid = co.get_property(ptype,'ParticleIDs').astype(int)
                unique_particle_ids = np.unique(
                    np.concatenate((unique_particle_ids,pid))
                    )
            except KeyError:
                warnings.warn('No particle IDs found for snapshot '+str(snap))
                pass
            del co
        return unique_particle_ids.astype(int)


class TreePrimary(TreeInfo):
    '''TreePrimary:

    Class to hold information about the primary subhalo from a sublink tree.
    Useful for holding and accessing information relevent to data products, 
    as well as recorded mergers into the primary.
    '''

    def __init__(self,tree_filename,tree_major_mergers=[],**kwargs):
        '''__init__:
        
        Initialise the class.

        Args:
            tree_major_mergers (list) - List of TreeMajorMerger objects
                that merge into this primary subhalo
        
        Keyword Args:
            mlpid (int) - MainLeafProgenitorID of the primary subhalo
            tree_filename (str) - Absolute filename of the sublink tree file
        '''
        # Initialize base class
        super(TreePrimary,self).__init__(tree_filename)

        # Initialize this class
        self.tree_major_mergers = tree_major_mergers
        self.n_major_mergers = len(tree_major_mergers)
        self.mlpid = kwargs.get('mlpid',None)
        
        # Use the tree to get some of the important information
        self.snapnum = None
        self.subfind_id = None
        if self.tree_filename is not None:
            tree = self.get_tree()
            assert self.mlpid == tree.get_property('MainLeafProgenitorID')[0]
            self.snapnum = tree.main_branch_snap
            self.subfind_id = tree.get_property('SubfindID')[tree.main_branch_mask]

class TreeMajorMerger(TreeInfo):
    '''TreeMajorMerger:

    Class to hold information about one identified major merger from 
    a sublink tree.
    '''

    def __init__(self,tree_filename,**kwargs):
        '''__init__:
        
        Initialise the class.
        
        Args:
            tree_filename (str) - Absolute filename of the sublink tree file
        
        Keyword Args:
            secondary_mlpid (int) - MainLeafProgenitorID of the secondary branch
                representing the merger remnant
            primary_mlpid (int) - MainLeafProgenitorID of the primary branch
                into which the merger occurs
            star_mass_ratio (float) - Stellar mass ratio of the merger as per 
                the method used
            star_mass_ratio_snapnum (int) - Snapshot number where the stellar 
                mass ratio is calculated
            dm_mass_ratio (float) - Dark matter mass ratio of the merger as per
                the method used
            dm_mass_ratio_snapnum (int) - Snapshot number where the dark matter
                mass ratio is calculated
            merger_snapnum (int) - Snapshot number where the secondary is no
                longer detected
            scheme (str) - String representing the method used to identify the
                merger
            scheme_kwargs (dict) - Dictionary of keyword arguments used to
                identify the merger
        '''
        # Initialize base class
        super(TreeMajorMerger,self).__init__(tree_filename)

        # Initialize this class
        self.secondary_mlpid = kwargs.get('secondary_mlpid',None)
        self.primary_mlpid = kwargs.get('primary_mlpid',None)
        self.star_mass_ratio = kwargs.get('star_mass_ratio',None)
        self.star_mass_ratio_snapnum = kwargs.get('star_mass_ratio_snapnum',None)
        self.dm_mass_ratio = kwargs.get('dm_mass_ratio',None)
        self.dm_mass_ratio_snapnum = kwargs.get('dm_mass_ratio_snapnum',None)
        self.merger_snapnum = kwargs.get('merger_snapnum',None)
        self.scheme = kwargs.get('scheme',None)
        self.scheme_kwargs = kwargs.get('scheme_kwargs',None)

        # Use the tree to get some of the important information
        self.snapnum = None
        self.subfind_id = None
        self._tree_indx = None
        self._len_tree = None
        self._tree_mask = None
        if self.tree_filename is not None:
            tree = self.get_tree()
            tree_mask = tree.get_property('MainLeafProgenitorID') == \
                self.secondary_mlpid
            self.snapnum = tree.get_property('SnapNum')[tree_mask]
            self.subfind_id = tree.get_property('SubfindID')[tree_mask]
            self._tree_indx = np.where(tree_mask)
            self._len_tree = len(tree.get_property('Mass'))
            # self._tree_mask = tree_mask
            # assert np.all(tree_mask == self._get_tree_mask())