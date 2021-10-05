# ----------------------------------------------------------------------------
#
# TITLE - util.py
# AUTHOR - James Lane
# PROJECT - tng-dfs
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Utilities and other misc functions. Includes config file loading and parsing.
'''
__author__ = "James Lane"

### Imports
import numpy as np

# ----------------------------------------------------------------------------

def load_config_to_dict(fname='./config.txt'):
    '''load_config_to_dict:
    
    Load a config file and convert to dictionary. Config file takes the form:
    
    KEYWORD1 = VALUE1 # comment
    KEYWORD2 = VALUE2
    etc..
    
    = sign must separate keywords from values. Trailing # indicates comment
    
    Args:
        fname (str) - Filename ['./config.txt']
        
    Returns:
        cdict (dict) - Dictionary of config keyword-value pairs
    '''
    cdict = {}
    with open(fname,'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.split('#')[0].strip() == '': continue # Empty line
            assert '=' in line, 'Keyword-Value pairs must be separated by "="'
            line_vals = line.split('#')[0].strip().split('=') # Remove comments and split at =
            cdict[line_vals[0].strip().upper()] = line_vals[1].strip()
        ##ln
    ##wi
    return cdict

def parse_config_dict(cdict,keyword):
    '''parse_config_dict:
    
    Parse config dictionary for keyword-value pairs. Valid keywords are:
        RO (float) - galpy distance scale
        VO (float) - galpy velocity scale
        ZO (float) - galpy vertical solar position
        HOME_DIR (string) - base directory for project
    
    Args:
        cdict (dict) - Dictionary of keyword-value pairs
        keyword (str or arr) - Keyword to extract, or array of keywords

    Returns:
        value (variable) - Value or result of the keyword
    '''
    if isinstance(keyword,(list,tuple,np.ndarray)): # Iterable but not string, many keywords
        _islist = True
        _keyword = []
        _value = []
        for key in keyword:
            assert key.upper() in cdict, 'Keyword not in cdict'
            _keyword.append(key.upper())
    else: # Assume string, just one keyword
        _islist = False
        _keyword = [keyword.upper(),]
        assert _keyword[0] in cdict, 'Keyword not in cdict'
    
    for key in _keyword:
        # Floats
        if key in ['RO','VO','ZO']:
            if _islist:
                _value.append( float(cdict[key]) )
            else:
                return float(cdict[key]) 
            ##ie
        ##fi       
        # Ints
        elif key in []:
            if _islist:
                _value.append( int(cdict[key]) )
            else:
                return int(cdict[key])
            ##ie
        ##ei
        # Strings 
        elif key in ['HOME_DIR',]:
            if _islist:
                _value.append( cdict[key] )
            else:
                return cdict[key]
            ##ie
        ##ei
        # No code, just pass value
        else:
            print('Warning: keyword '+key+' has no parsing code,'+
            ' just passing value')
            if _islist:
                _value.append( cdict[key] )
            else:
                return cdict[key]
    # Assume single key has returned already
    return _value
#def

# ----------------------------------------------------------------------------
