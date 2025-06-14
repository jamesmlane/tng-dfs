# ----------------------------------------------------------------------------
#
# TITLE - util.py
# AUTHOR - James Lane
# PROJECT - tng-dfs
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Utilities and other misc functions. Includes config file loading and parsing,
TNG API requests, unit conversions
'''
__author__ = "James Lane"

### Imports
import numpy as np
import os
import requests
import time
import copy
import dill as pickle
import astropy.units as apu
from galpy import orbit
import pdb

__snap_zs__ = None
_HUBBLE_PARAM = 0.6774 # Planck 2015 value

# ----------------------------------------------------------------------------

# Config file loading and parsing

def load_config_to_dict(fname='config.txt'):
    '''load_config_to_dict:
    
    Load a config file and convert to dictionary. Config file takes the form:
    
    KEYWORD1 = VALUE1 # comment
    KEYWORD2 = VALUE2
    etc..
    
    = sign must separate keywords from values. Trailing # indicates comment
    
    Args:
        fname (str) - Name of the config file ['config.txt']
        
    Returns:
        cdict (dict) - Dictionary of config keyword-value pairs
    '''
    cdict = {}
    fname_path = _find_config_file(fname)
    with open(fname_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.split('#')[0].strip() == '': continue # Empty line
            assert '=' in line, 'Keyword-Value pairs must be separated by "="'
            # Remove comments and split at =
            line_vals = line.split('#')[0].strip().split('=') 
            cdict[line_vals[0].strip().upper()] = line_vals[1].strip()
    return cdict

def _find_config_file(fname):
    '''_find_config_file:
    
    Recursively find a config file by searching upwards through the directory
    structure
    
    Args:
        fname (str) - Name of the configuration file to search for
    
    Returns:
        config_path (str) - Path to the configuration file
    '''
    config_dir = ''
    while True:
        if os.path.exists(config_dir+fname):
            return config_dir+fname
        if os.path.realpath(config_dir).split('/')[-1] == 'tng-dfs':
            raise FileNotFoundError('Could not find configuration file within'+
                                    ' project directory structure')
        if os.path.realpath(config_dir) == '/':
            raise RuntimeError('Reached base directory')
        config_dir = config_dir+'../'

def parse_config_dict(cdict,keyword):
    '''parse_config_dict:
    
    Parse config dictionary for keyword-value pairs. Valid keywords are:
        DATA_DIR (string) - Directory for project data
        MW_ANALOG_DIR (string) - Directory for MW analog data
        FIG_DIR_BASE (string) - Base directory for figures
        FITTING_DIR_BASE (string) - Base directory for fitting results
        RO (float) - galpy distance scale
        VO (float) - galpy velocity scale
        ZO (float) - galpy vertical solar position
        LITTLE_H (float) - Hubble constant in units of 100 km/s/Mpc
        MW_MASS_RANGE (list) - Range of Milky Way stellar masses in 10**10 Msun
    
    Args:
        cdict (dict) - Dictionary of keyword-value pairs
        keyword (str or arr) - Keyword to extract, or array of keywords

    Returns:
        value (variable) - Value or result of the keyword
    '''
    if isinstance(keyword,(list,tuple,np.ndarray)): # many keywords
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
        float_keys = ['RO','VO','ZO','LITTLE_H']
        int_keys = []
        string_keys = ['DATA_DIR','MW_ANALOG_DIR','FIG_DIR_BASE',
                       'FITTING_DIR_BASE']
        list_keys = ['MW_MASS_RANGE']
        # Floats
        if key in float_keys:
            if _islist:
                _value.append( float(cdict[key]) )
            else:
                return float(cdict[key])   
        # Ints
        elif key in int_keys:
            if _islist:
                _value.append( int(cdict[key]) )
            else:
                return int(cdict[key])
        # Strings 
        elif key in string_keys:
            if _islist:
                _value.append( cdict[key] )
            else:
                return cdict[key]
        # Lists
        elif key in list_keys:
            if _islist:
                _arr = cdict[key].strip('][').split(',')
                _arr = [float(x) for x in _arr]
                _value.append( _arr )
            else:
                _arr = cdict[key].strip('][').split(',')
                _arr = [float(x) for x in _arr]
                return _arr
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

# Standard notebook preparation

def prepare_mwsubs(mw_analog_dir,h=_HUBBLE_PARAM,mw_mass_range=[5,7],
    return_vars=False,force_mwsubs=False,bulge_disk_fraction_cuts=False,
    bulge_disk_fraction_id_filename='bulge_disk_fraction_z0sid.npy'):
    '''prepare_mwsubs:

    Do some standard prep: fetch simulation info with TNG API, load the 
    mwsubs file

    Args:
        mw_analog_dir (str) - Directory for MW Analog data
        h (float) - Hubble constant in units of 100 km/s/Mpc from config,
            default 0.7
        mw_mass_range (arr) - Range of Milky Way stellar masses in 10**10 Msun
            from config, default [5,7]
        return_vars (bool) - Return all variables as a dict instead of just 
            mwsubs.
        bulge_disk_fraction_cuts (bool) - Apply cuts to the sample based on
            the bulge and disk fraction analysis, default False
        bulge_disk_fraction_id_filename (str) - Filename for the bulge and
            disk fraction IDs. Will be queried within mw_analog_dir/masks/ . 
            Default 'bulge_disk_fraction_z0sid.npy'
    
    Returns:
        mwsubs (numpy recarray) - recarray  of MW analog subhalo information
        vars (dict) - Dictionary of most variables used in this function
    '''
    # Base URL
    base_url = 'http://www.tng-project.org/api/'
    # Get list of simulations
    base_url_path = os.path.join(mw_analog_dir,'subs','api','base_url.pkl')
    if force_mwsubs or os.path.exists(base_url_path) == False:
        print('Downloading base_url')
        r = get(base_url)
        with open(base_url_path,'wb') as f:
            pickle.dump(r,f)
    else:
        print('Loading base_url from '+base_url_path)
        with open(base_url_path,'rb') as f:
            r = pickle.load(f)
    sim_names = [sim['name'] for sim in r['simulations']]
    tng50_indices = [sim_names.index('TNG50-'+str(i+1)) for i in range(4)]
    # Choose the lowest resolution tng50 run
    tng50_urls = [r['simulations'][i]['url'] for i in tng50_indices]
    tng50_url = tng50_urls[0]

    # Get the simulation, snapshots, snapshot redshifts
    sim_path = os.path.join(mw_analog_dir,'subs','api','sim.pkl')
    if force_mwsubs or os.path.exists(sim_path) == False:
        print('Downloading simulation info')
        sim = get( tng50_url )
        with open(sim_path,'wb') as f:
            pickle.dump(sim,f)
    else:
        print('Loading simulation info from '+sim_path)
        with open(sim_path,'rb') as f:
            sim = pickle.load(f)
    snaps_path = os.path.join(mw_analog_dir,'subs','api','snaps.pkl')
    if force_mwsubs or os.path.exists(snaps_path) == False:
        print('Downloading snapshot info')
        snaps = get( sim['snapshots'] )
        with open(snaps_path,'wb') as f:
            pickle.dump(snaps,f)
    else:
        print('Loading snapshot info from '+snaps_path)
        with open(snaps_path,'rb') as f:
            snaps = pickle.load(f)
    snap_zs = [snap['redshift'] for snap in snaps]
    snap0_path = os.path.join(mw_analog_dir,'subs','api','snap0.pkl')
    if force_mwsubs or os.path.exists(snap0_path) == False:
        print('Downloading snapshot 0 info')
        snap0 = get( snaps[-1]['url'] )
        with open(snap0_path,'wb') as f:
            pickle.dump(snap0,f)
    else:
        print('Loading snapshot 0 info from '+snap0_path)
        with open(snap0_path,'rb') as f:
            snap0 = pickle.load(f)

    # Query the API for subhalos with stellar mass in a range near that of the 
    # Milky Way
    # mw_mass_range = np.array([5,7])*1e10
    mw_mass_range = np.asarray(mw_mass_range)
    if np.all(mw_mass_range > 1e5): raise ValueError('mw_mass_range probably not in units of 1e10')
    mw_mass_range *= 1e10
    mw_mass_range_code = mass_physical_to_code(mw_mass_range,h=h,e10=True)
    mw_search_query = '?mass_stars__gt='+str(mw_mass_range_code[0])+\
                        '&mass_stars__lt='+str(mw_mass_range_code[1])+\
                        '&primary_flag__gt=0'
    mw_search_query_path = os.path.join(mw_analog_dir,'subs','api',
        mw_search_query+'.pkl')
    if force_mwsubs or os.path.exists(mw_search_query_path) == False:
        print('Downloading Milky Way like galaxies')
        mw_search_results = get( snap0['subhalos']+mw_search_query )['results']
        with open(mw_search_query_path,'wb') as f:
            pickle.dump(mw_search_results,f)
    else:
        print('Loading Milky Way like galaxies from '+mw_search_query_path)
        with open(mw_search_query_path,'rb') as f:
            mw_search_results = pickle.load(f)
    print(str(len(mw_search_results))+' Milky way like galaxies found')
    n_mw = len(mw_search_results)

    # Get subhalo data
    mwsubs_path = os.path.join(mw_analog_dir,'subs','mwsubs.pkl')
    if force_mwsubs or os.path.exists(mwsubs_path) == False:
        print('Downloading subhalo data')
        mwsubs = []
        for i in range(len(mw_search_results)):
            mwsubs.append( get( mw_search_results[i]['url'], timeout=None ) )
        # Save subhalo data
        print('Saving subhalo data to '+mwsubs_path)
        with open(mwsubs_path,'wb') as f:
            pickle.dump(mwsubs,f)
    else:
        print('Loading subhalo data from '+mwsubs_path)
        with open(mwsubs_path,'rb') as f:
            mwsubs = pickle.load(f)
        print('File has '+str(len(mwsubs))+' subhalos')

    # Convert to numpy recarray 
    mwsubs_dict = copy.deepcopy(mwsubs)
    mwsubs = subhalo_list_to_recarray(mwsubs)

    if bulge_disk_fraction_cuts:
        print('Cutting on bulge and disk fractions')
        ids = mwsubs['id']
        gid_filename = os.path.join(mw_analog_dir, 'masks/', 
            bulge_disk_fraction_id_filename)
        try:
            gids = np.load(gid_filename)
        except FileNotFoundError:
            print('Could not find file '+gid_filename)
            print('Skipping bulge and disk fraction cuts')
            bulge_disk_fraction_cuts = False
        bulge_disk_fraction_mask = np.in1d(ids,gids)
        mwsubs = mwsubs[bulge_disk_fraction_mask]
        mwsubs_dict = [mwsubs_dict[i] for i in range(n_mw) if bulge_disk_fraction_mask[i]]
        n_mw = len(mwsubs)
        print('Cut to ',n_mw,' subhalos')
    else:
        bulge_disk_fraction_mask = None

    if return_vars:
        vars = {'base_url':base_url,'sim_names':sim_names,
                'tng50_indices':tng50_indices,'tng50_urls':tng50_urls,
                'tng50_url':tng50_url,'sim':sim,'snaps':snaps,'snap_zs':snap_zs,
                'snap0':snap0,'mw_mass_range':mw_mass_range,
                'mw_mass_range_code':mw_mass_range_code,
                'mw_search_query':mw_search_query,
                'mw_search_results':mw_search_results,
                'bulge_disk_fraction_mask':bulge_disk_fraction_mask,
                'n_mw':n_mw,'mwsubs':mwsubs,'mwsubs_dict':mwsubs_dict}
        return mwsubs,vars
    else:
        return mwsubs

# ----------------------------------------------------------------------------

# API Querying

def get(path, params=None, timeout=60, directory='./', timeit=False):
    '''get:
    
    Make and HTTP get request to a path.
    '''
    if timeit:
        t1 = time.time()
    
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
    
    # If a file is supplied save it
    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(directory+filename, 'wb') as f:
            f.write(r.content)
        if timeit:
            t2 = time.time()
            print('get() took '+str(round(t2-t1,1))+'s')
        return filename # return the filename string
    
    return r

# ----------------------------------------------------------------------------

# Unit conversion

def mass_code_to_physical(m,h=_HUBBLE_PARAM,e10=True):
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

def mass_physical_to_code(m,h=_HUBBLE_PARAM,e10=True):
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

def distance_code_to_physical(d,h=_HUBBLE_PARAM,z=0.):
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

def distance_physical_to_code(d,h=_HUBBLE_PARAM,z=0.):
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

def angular_momentum_code_to_physical(j,h=_HUBBLE_PARAM,z=0):
    '''angular_momentum_code_to_physical:
    Convert a comoving angular momentum in Illustris-TNG code units to physical,
    non-comoving units.

    Args:
        j (float or np.array) - angular momentum in comoving TNG code units
        h (float) - Hubble parameter
        z (float) - Redshift

    Returns:
        j_phys (float or np.array) - angular momentum in non-comoving, physical
            units
    '''
    a = 1./(z+1.)
    return j*a**1.5/h

def angular_momentum_physical_to_code(j,h=_HUBBLE_PARAM,z=0):
    '''angular_momentum_physical_to_code:
    Convert a physical, non-comoving angular momentum in units to comoving 
    Illustris-TNG code units.

    Args:
        j (float or np.array) - angular momentum in non-comoving, physical 
            units
        h (float) - Hubble parameter
        z (float) - Redshift

    Returns:
        j_code (float or np.array) - angular momentum in comoving TNG code 
            units
    '''
    a = 1./(z+1.)
    return j*h/a**1.5

def snapshot_to_redshift(snap,sim_url='http://www.tng-project.org/api/TNG50-1/'):
    '''snapshot_to_redshift:
    Convert a simulation snapshot number to redshift.
    
    Args:
        snap (int or array) - Snapshot number, larger N is smaller redshift
        sim_url (str) - URL of the simulation API [default: TNG50-1]
    
    Returns:
        z (float) - Redshift
    '''
    # First get the redshifts for all snapshots
    global __snap_zs__
    if __snap_zs__ is None:
        sim = get( sim_url )
        snaps = get( sim['snapshots'] )
        __snap_zs__ = [snap['redshift'] for snap in snaps]
    if isinstance(snap,(list,tuple,np.ndarray)):
        return np.array([__snap_zs__[s] for s in snap])
    else: # Assume can become an int
        return __snap_zs__[int(snap)]

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
                keys.append(key)
                if i>0: print('Warning: new key not in the first dict, '+\
                              str(key))

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
                subs_rec[key][i] = subdict
            else:
                try:
                    subs_rec[key][i] = sub[key]
                except KeyError:
                    subs_rec[key][i] = None
    return subs_rec

def ptype_to_indx(ptype):
        '''ptype_to_ind: Query the standard relationship between named 
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

def ptype_to_str(ptype):
    '''Lightweight wrapper to ptype_to_indx which returns the formal name of 
    the particle type for querying HDF5 files.
    '''
    _strs = ['PartType0','PartType1','PartType2','PartType3','PartType4',
             'PartType5']
    return _strs[ptype_to_indx(ptype)]

def parse_astropy_quantity(q, unit):
    if isinstance(q,apu.Quantity):
        q = q.to(unit).value
    elif isinstance(q,(list,tuple,np.ndarray)):
        q = np.array([parse_astropy_quantity(qq,unit) for qq in q])
    return q

def get_softening_length(ptype, z=0, sim_name='TNG50-1', physical=True):
    '''get_softening_length:
    
    Get the softening length for a given particle type at a given redshift.

    Args:
        ptype
    '''
    base_url = 'http://www.tng-project.org/api/'
    sim = get( base_url+sim_name )
    ptype_str = ptype_to_str(ptype)
    a = 1./(z+1)
    if ptype_str == 'PartType1': # DM
        softening_length_comoving = sim['softening_dm_comoving']
        softening_length_code = softening_length_comoving*a
        softening_max_code = sim['softening_dm_max_phys']
        softening_length_code = np.min([softening_length_code,
                                        softening_max_code])
    elif ptype_str == 'PartType4': # Stars
        softening_length_comoving = sim['softening_stars_comoving']
        softening_length_code = softening_length_comoving*a
        softening_max_code = sim['softening_stars_max_phys']
        softening_length_code = np.min([softening_length_code,
                                        softening_max_code])
    else:
        raise ValueError('Only PartType0 and 4 are supported')

    if physical:
        # Set z=0 because we already did comoving -> fixed
        return distance_code_to_physical(softening_length_code, z=0)
    else:
        return softening_length_code

# ----------------------------------------------------------------------------

# Pathing functions

def get_cutout_filename(mw_analog_dir, snapnum, subfind_id):
    '''get_cutout_filename:

    Get the filename of a cutout file for a given subhalo at a given snapshot

    Args:
        mw_analog_dir (str) - Directory for MW Analog data
        snapnum (int) - Snapshot number
        subfind_id (int) - Subfind ID

    Returns:
        fname (str) - Filename of the cutout file
    '''
    snap_path = mw_analog_dir+'cutouts/snap_'+str(snapnum)+'/'
    fname = snap_path+'cutout_'+str(subfind_id)+'.hdf5'
    if not os.path.isfile(fname):
        raise IOError('File '+fname+' does not exist')
    return fname

# ----------------------------------------------------------------------------

# Misc. functions

def find_contiguous(mask):
    '''find_contiguous: Find contiguous regions in a mask which are True with 
    no false in between'''
    assert len(mask.shape)==1, 'Mask must be 1D'
    contigs = []
    found_contig = False 
    for i,b in enumerate(mask):
        if b and not found_contig: # found the beginning, record index as start, set indicator
            contig = [i]
            found_contig = True 
        elif b and found_contig: # currently have contig, continuing it 
            pass
        elif not b and found_contig: # found the end, record previous index as end, reset indicator  
            contig.append(i-1)
            found_contig = False 
            contigs.append(tuple(contig))
        else: # currently don't have a contig, and didn't find one 
            pass 

    if b: # Check if the very last entry was True and we didn't get to finish 
        contig.append(i)
        found_contig = False 
        contigs.append(tuple(contig))
        
    return contigs

def flatten_list(nested_list, recursive=True, flatten_types=[list,tuple]):
    '''flatten_nested_list:
    
    Flatten a nested list into a single list.
    
    Args:
        a (list) - Nested list
        recursive (bool) - Continue to flatten sublists until all that remains
            are elements which are not in recursive_types, default False.
        recursive_types (list) - Types of elements to continue flattening
            if recursive is True, default [list,tuple]
    
    Returns:
        a_lin (list) - Flattened list
    '''
    flattened_list = []
    for element in nested_list:
        if recursive and any(isinstance(element, t) for t in flatten_types):
            flattened_list.extend(flatten_list(element, recursive, flatten_types))
        else:
            for el in element:
                flattened_list.append(el)
    return flattened_list

def moving_average(data, window_size):
    '''
    Compute the moving average of a list for a given window size.

    Parameters:
    - data: List of numeric values.
    - window_size: Size of the moving average window.

    Returns:
    - List of moving averages.
    '''
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")

    moving_averages = []
    window_sum = 0

    for i in range(len(data)):
        window_sum += data[i]

        if i >= window_size - 1:
            moving_average = window_sum / window_size
            moving_averages.append(moving_average)
            window_sum -= data[i - window_size + 1]

    return moving_averages

def join_orbs(orbs):
    '''join_orbs:
    
    Join a list of orbit.Orbit objects together. They must share ro,vo and 
    should otherwise have been initialized in a similar manner.
    
    Args:
        orbs (orbit.Orbit) - list of individual orbit.Orbit objects
    
    Returns:
        orbs_joined (orbit.Orbit) - Joined orbit.Orbit object
    '''
    for i,o in enumerate(orbs):
        if i == 0:
            ro = o._ro
            vo = o._vo
            vxvvs = o._call_internal()
        else:
            assert ro==o._ro and vo==o._vo, 'ro and/or vo do not match'
            vxvvs = np.append(vxvvs, o._call_internal(), axis=1)
    return orbit.Orbit(vxvvs.T,ro=ro,vo=vo)