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

# API Querying

def get(path, params=None, timeout=10, directory='./', timeit=False):
    '''get:
    
    Make and HTTP get request to a path.
    '''
    if timeit:
        t1 = time.time()
    ##fi
    
    # The header contains the TNG API key stored as an environment variable
    headers = {'api-key':os.environ['ILLUSTRIS_TNG_API_KEY']}
    
    r = requests.get(path, params=params, headers=headers, timeout=timeout)
        
    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()
    
     # Parse and return JSON
    if r.headers['content-type'] == 'application/json':
        if timeit:
            t2 = time.time()
            print('get() took '+str(round(t2-t1,1))+'s')
        return r.json()
    ##fi
    
    # If a file is supplied save it
    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(directory+filename, 'wb') as f:
            f.write(r.content)
        ##wi
        if timeit:
            t2 = time.time()
            print('get() took '+str(round(t2-t1,1))+'s')
        ##fi
        return filename # return the filename string
    ##fi
    
    return r

# ----------------------------------------------------------------------------

# Unit conversion

def mass_code_to_physical(m,h=0.7,e10=True):
    '''mass_code_to_physical:
    Convert a mass in Illustris-TNG code units to physical units.
    
    Args:
        m (float or np.array) - mass in TNG code units
        h (float) - Hubble parameter
        e10 (bool) - Code masses in units of 1e10/h Msun?
    
    Returns:
        m_phys (float or np.array) mass in physical Msol
    '''
    m_phys = m/h
    if e10:
        m_phys *= 1e10
    return m_phys

def mass_physical_to_code(m,h=0.7,e10=True):
    '''mass_physical_to_code:
    Convert a mass in physical units to Illustris-TNG code units.
    
    Args:
        m (float or np.array) - mass in physical Msol
        h (float) - Hubble parameter
        e10 (bool) - Code masses in units of 1e10/h Msun?
    
    Returns:
        m_code (float or np.array) - mass in TNG code units
    '''
    m_code = m*h
    if e10:
        m_code /= 1e10
    return m_code

def distance_code_to_physical(d,h=0.7,z=0.):
    '''distance_code_to_physical:
    Convert a comoving distance in Illustris-TNG code units to phyical, 
    non-comoving units.
    
    Args:
        d (float or np.array) - distance in comoving TNG code units
        h (float) - Hubble parameter
        z (float) - Redshift
    
    Returns:
        d_phys (float or np.array) - distance in non-comoving, physical kpc
    '''
    a = 1./(z+1)
    return d*a/h

def distance_physical_to_code(d,h=0.7,z=0.):
    '''distance_physical_to_code:
    Convert a physical, non-comoving distance to comoving Illustris-TNG code 
    units.
    
    Args:
        d (float or np.array) - distance in non-comoving, physical kpc
        h (float) - Hubble parameter
        z (float) - Redshift
    
    Returns:
        d_code (float or np.array) - distance in comoving TNG code units
    '''
    a = 1./(z+1)
    return d*h/a

def velocity_code_to_physical(v,z=0.):
    '''velocity_code_to_physical:
    Convert a comoving velocity in Illustris-TNG code units to physical, 
    non-comoving units.
    
    Args:
        v (float or np.array) - velocity in comoving TNG code units
        z (float) - Redshift
    
    Returns:
        v_phys (float or np.array) - velocity in non-comoving, physical units
    '''
    a = 1./(z+1.)
    return v*a**0.5

def velocity_physical_to_code(v,z=0.):
    '''velocity_physical_to_code:
    Convert a physical, non-comoving velocity in km/s to comoving Illustris-TNG
    code units.
    
    Args:
        v (float or np.array) - velocity in non-comoving, physical units
        z (float) - Redshift
    
    Returns:
        v_code (float or np.array) - velocity in comoving TNG code units
    '''
    a = 1./(z+1.)
    return v/a**0.5

def energy_code_to_physical(e,z=0.):
    '''energy_code_to_physical:
    Convert a comoving energy in Illustris-TNG code units to physical, 
    non-comoving units.
    
    Args:
        e (float or np.array) - energy in comoving TNG code units
        z (float) - Redshift
    
    Returns:
        e_phys (float or np.array) - energy in non-comoving, physical units
    '''
    a = 1./(z+1.)
    return e*a

def energy_physical_to_code(e,z=0.):
    '''energy_physical_to_code:
    Convert a physical, non-comoving energy in units to comoving Illustris-TNG 
    code units.
    
    Args:
        e (float or np.array) - energy in non-comoving, physical units
        z (float) - Redshift
    
    Returns:
        e_code (float or np.array) - energy in comoving TNG code units
    '''
    a = 1./(z+1.)
    return e/a

# ----------------------------------------------------------------------------

# Converters and data wranglers

def subhalo_list_to_recarray(subs):
    '''subhalo_list_to_recarray:
    
    Transform a list of subhalo dictionaries provided by the TNG API into a 
    numpy recarray.
    
    Args:
        subs (list) - List of subhalo dicts provided by TNG API
        
    Returns:
        subs_rec (numpy.recarray)
    '''
    # Known API dict keys with dicts as elements
    subdict_keys = ['related','cutouts','trees','supplementary_data','vis',
                    'meta']

    # First get all keys for the recarray
    keys = []
    for i,sub in enumerate(subs):
        for j,key in enumerate(sub.keys()):
            if key not in keys:
                if key in subdict_keys or isinstance(sub[key],dict):
                    continue # We'll do these afterwards
                ##fi
                keys.append(key)
                if i>0: print('Warning: new key not in the first dict, '+\
                              str(key))
            ##fi
        ###i
    ###i

    for i,sub in enumerate(subs):
        for j,key in enumerate(sub.keys()):
            if not isinstance(sub[key],dict):
                continue # Assume entry is dictionary
            ##fi
            for kkey in sub[key]:
                subkey = key+':'+kkey
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
    for key in keys:
        if key in is_int:
            dt.append( (key,int) )
        elif ':' in key:
            dt.append( (key,object) )
        else:
            dt.append( (key,float) )
        ##ie
    ###k

    subs_rec = np.recarray((len(subs),),dtype=dt)

    for i,sub in enumerate(subs):
        for j,key in enumerate(keys):
            if ':' in key:
                subkeys = key.split(':')
                subdict = sub[subkeys[0]]
                subkeys =  subkeys[1:]
                for kkey in subkeys:
                    try:
                        subdict = subdict[kkey]
                    except KeyError: 
                        subdict = None
                        break
                    ##te
                ##kk
                subs_rec[key][i] = subdict
            else:
                try:
                    subs_rec[key][i] = sub[key]
                except KeyError:
                    subs_rec[key][i] = None
                ##te
            ##ie
        ###j
    ###i
    return subs_rec
#def