# ----------------------------------------------------------------------------
#
# TITLE - io.py
# AUTHOR - James Lane
# PROJECT - tng-dfs
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''input/output functions
'''
__author__ = "James Lane"

### Imports
import numpy as np
import dill as pickle
import pdb

# ----------------------------------------------------------------------------

# Loading TNG subhalo lists and transforming into numpy rec_array

def load_subhalo_list(filename,recarray=True):
    '''load_subhalo_list:
    
    Load a list of subhalo dictionaries and possibly transform into a 
    numpy recarray
    
    Args:
        filename (string) - filename of binary holding list of subhalo dicts
        recarray () - Transform to numpy recarray? [True]
    
    Returns:
        subs () - 
    '''
    
    # Load from binary a list of subhalo dictionaries that are provided by 
    # the TNG API
    with open(filename,'rb') as f:
        subs = pickle.load(f)
    ##wi
    
    if recarray:
        return subhalo_list_to_recarray(subs)
    else:
        return subs
    ##ie
#def

def subhalo_list_to_recarray(subs):
    '''subhalo_list_to_recarray:
    
    Transform a list of subhalo dictionaries provided by the TNG API into a 
    numpy recarray.
    
    Args:
        subs (list) - List of subhalo dicts provided by TNG API
        
    Returns:
        subs_rec (numpy.recarray)
    '''
    
    subdict_keys = ['related','cutouts','trees','supplementary_data','vis',
                    'meta']

    # First get all keys for the recarray
    keys = []
    for i,d in enumerate(subs):
        for j,k in enumerate(d.keys()):
            if k not in keys:
                if k in subdict_keys or isinstance(d[k],dict):
                    continue # We'll do these at the end
                ##fi
                keys.append(k)
                if i>0: print('Warning: new key not in the first dict, '+\
                              str(k))
            ##fi
        ###i
    ###i

    for i,d in enumerate(subs):
        for j,k in enumerate(d.keys()):
            if not isinstance(d[k],dict):
                continue # Assume entry is dictionary
            ##fi
            for kk in d[k]:
                subkey = k+':'+kk
                if subkey in keys:
                    continue
                keys.append(subkey)
                if i>0: print('Warning: new key not in the first dict, '+\
                              str(subkey))
            ##kk
        ###k
    ###i

    # dtype should be int
    is_int = ['snap','id','len','len_gas','len_dm','len_stars','len_bhs',
              'prog_snap','prog_sfid','desc_snap','desc_sfid','parent',
              'grnr','primary_flag']

    # Now create the recarray dtypes
    dt = []
    for k in keys:
        if k in is_int:
            dt.append( (k,int) )
        elif ':' in k:
            dt.append( (k,object) )
        else:
            dt.append( (k,float) )
        ##ie
    ###k

    subs_rec = np.recarray((len(dt),),dtype=dt)

    for i,d in enumerate(subs):
        for j,k in enumerate(keys):
            if ':' in k:
                subks = k.split(':')
                subdict = d[subks[0]]
                subks =  subks[1:]
                for kk in subks:
                    try:
                        subdict = subdict[kk]
                    except KeyError: 
                        subdict = None
                        break
                    ##te
                ##kk
                subs_rec[k] = subdict
            else:
                try:
                    subs_rec[k] = d[k]
                except KeyError:
                    subs_rec[k] = None
                ##te
            ##ie
        ###j
    ###i
    
    return subs_rec
#def

# ----------------------------------------------------------------------------