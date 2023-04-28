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

# ----------------------------------------------------------------------------

# Parsing sublink trees

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
#def
