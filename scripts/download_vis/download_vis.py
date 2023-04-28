# ----------------------------------------------------------------------------
#
# TITLE - download_vis.py
# AUTHOR - James Lane
# PROJECT - tng-dfs
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Download images visualizing the Milky Way - like galaxies from TNG
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os

sys.path.insert(0,'../../src/')
from tng_dfs import util as putil
from tng_dfs.util import get

# ----------------------------------------------------------------------------

# Keywords
cdict = putil.load_config_to_dict()
keywords = ['DATA_DIR','RO','VO','ZO','LITTLE_H']
data_dir,ro,vo,zo,h = putil.parse_config_dict(cdict,keywords)
get_subhalo_images = True
get_mergertree_images = True

# base URL
baseURL = 'http://www.tng-project.org/api/'

# Get list of simulations, choose TNG50-1
r = putil.get(baseURL)
sim_names = [sim['name'] for sim in r['simulations']]
tng50_indices = [sim_names.index('TNG50-'+str(i+1)) for i in range(4)]
tng50_urls = [r['simulations'][i]['url'] for i in tng50_indices]
tng50_url = tng50_urls[0]

# Get the simulation, snapshots, present-day snapshot
sim = putil.get( tng50_url )
snaps = get( sim['snapshots'] )
snap0 = get( snaps[-1]['url'] )

# Query the API for subhalos with stellar mass in a range near that of the 
# Milky Way
mw_mass_range = np.array([5,7])*1e10
mw_mass_range_code = putil.mass_physical_to_code(mw_mass_range,h=h)
mw_search_query = '?mass_stars__gt='+str(mw_mass_range_code[0]/1e10)+\
                       '&mass_stars__lt='+str(mw_mass_range_code[1]/1e10)+\
                       '&primary_flag__gt=0'
mw_search_results = get( snap0['subhalos']+mw_search_query )['results']
print(str(len(mw_search_results))+' Milky way like galaxies found')
n_mw = len(mw_search_results)

# Unpack subhalos
mwsubs = []
for i in range(len(mw_search_results)):
    mwsubs.append( get( mw_search_results[i]['url'] ) )
###i
print('Subhalos downloaded')

# Loop over subhalos and query face-on stellar density
if get_subhalo_images:
    for i in range(n_mw):
        directory = data_dir+'mw_analogs/visualizations/snap_'+\
            str(mwsubs[i]['snap'])+'/subhalo_'+str(mwsubs[i]['id'])+'/'
        if not os.path.exists(directory):
            os.makedirs(directory,exist_ok=True)
        ##fi
        vis_url = mwsubs[i]['vis']['galaxy_stellar_light_faceon']
        vis_url_bigview = vis_url.split('size=5.0')[0]+'size=8.0'+\
                          vis_url.split('size=5.0')[1]
        vis_url_bigview = vis_url.split('size=5.0')[0]+'size=100.0&sizeType=kpc'
        _ = get( vis_url_bigview, timeout=None, directory=directory, timeit=True)
        print('downloaded face on image for subhalo '+str(i)+\
              ' - id '+str(mwsubs[i]['id']))
    ###i
##fi

# Loop over subhalos and query images of the merger trees
if get_mergertree_images:
    for i in range(n_mw):
        directory = data_dir+'mw_analogs/sublink_trees/images/'
        if not os.path.exists(directory):
            os.makedirs(directory,exist_ok=True)
        ##fi
        img_url = mwsubs[i]['vis']['mergertree_sublink']
        _ = get( img_url, timeout=None, directory=directory, timeit=True )
        print('downloaded sublink merger tree image for subhalo '+str(i)+\
              ' - id '+str(mwsubs[i]['id']))
    ###i
##fi