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
import os
# import pdb

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
    
    if recarray:
        return subhalo_list_to_recarray(subs)
    else:
        return subs

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
                keys.append(k)
                if i>0: print('Warning: new key not in the first dict, '+\
                              str(k))

    for i,d in enumerate(subs):
        for j,k in enumerate(d.keys()):
            if not isinstance(d[k],dict):
                continue # Assume entry is dictionary
            for kk in d[k]:
                subkey = k+':'+kk
                if subkey in keys:
                    continue
                keys.append(subkey)
                if i>0: print('Warning: new key not in the first dict, '+\
                              str(subkey))

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
                subs_rec[k] = subdict
            else:
                try:
                    subs_rec[k] = d[k]
                except KeyError:
                    subs_rec[k] = None
    
    return subs_rec

# ----------------------------------------------------------------------------

# Loading emcee samplers

def median_params_from_emcee_sampler(filename, ncut=0, nthin=1, 
    percentiles=[50,], return_samples=False):
    '''median_params_from_emcee_sampler:

    Load an emcee sampler and return the median parameters (or percentiles)

    Args:
        filename (string) - filename of emcee sampler (assume pickled)
        ncut (int) - Number of samples to cut from the beginning of the chain, 
            default is 0
        nthin (int) - Number of samples to thin the chain by, default is 0
        percentiles (list) - List of percentiles to return, default is [50,]
            (i.e. the median)
        return_samples (bool) - Return the samples? Default is False
    
    Returns:
        params (array) - Array of parameters defined by percentiles with shape
            (nparams,len(percentiles))
        samples (array) - Array of samples from the emcee sampler if 
            return_samples is True
    '''
    assert os.path.isfile(filename), "File not found: "+str(filename)
    assert ncut >= 0, "ncut must be >= 0"
    assert nthin >= 0, "nthin must be >= 0"
    assert isinstance(percentiles,list), "percentiles must be a list"

    # Load the sampler
    with open(filename,'rb') as f:
        sampler = pickle.load(f)
    
    # Get the samples
    samples = sampler.get_chain(discard=ncut,thin=nthin,flat=True)

    # Get the percentiles, transpose to get shape (nparams,len(percentiles))
    params = np.percentile(samples,percentiles,axis=0).T

    if return_samples:
        return params, samples
    else:
        return params