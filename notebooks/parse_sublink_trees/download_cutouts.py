# ------------------------------------------------------------------------
#
# TITLE - parse_sublink_trees.py
# AUTHOR - James Lane
# PROJECT - tng-dfs
#
# ------------------------------------------------------------------------
#
# Docstring:
'''Parse sublink trees of MW analogs and find progenitors of high mass ratio 
mergers. Track those systems back in time and records their particles.
'''

__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb
import h5py
import glob
import copy
import dill as pickle

## Matplotlib
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0,'../../src/')
from tng_dfs import util as putil
from tng_dfs import tree as ptree
from tng_dfs.util import get

### Notebook setup
plt.style.use('../../src/_matplotlib/project.mplstyle') # This must be exactly here

# Keywords
cdict = putil.load_config_to_dict()
keywords = ['DATA_DIR','RO','VO','ZO','LITTLE_H']
data_dir,ro,vo,zo,h = putil.parse_config_dict(cdict,keywords)

# Base URL
baseURL = 'http://www.tng-project.org/api/'
# Get list of simulations
r = get(baseURL)
sim_names = [sim['name'] for sim in r['simulations']]
tng50_indices = [sim_names.index('TNG50-'+str(i+1)) for i in range(4)]
# Choose the lowest resolution tng50 run
tng50_urls = [r['simulations'][i]['url'] for i in tng50_indices]
tng50_url = tng50_urls[0]

# Get the simulation, snapshots, snapshot redshifts
sim = get( tng50_url )
snaps = get( sim['snapshots'] )

# Load major list
with open('./data/all_major_list.pkl','rb') as f:
    all_major_list = pickle.load(f)
##wi
# Number of primary MW analogs under consideration
n_mw = len(all_major_list)

# First prepare directory structure
for i in range(len(snaps)):
    snap_path = data_dir+'cutouts/snap_'+str(snaps[i]['number'])+'/'
    os.makedirs(snap_path,exist_ok=True)
###i

# Loop over all mergers and download
for i in range(n_mw):
    major_dict = all_major_list[i]
    major_list = major_dict['major_list']
    n_major = major_dict['n_major']
    print('Downloading info for z=0 subhalo '+\
              str(major_dict['primary_z0_subfind_id']))
    
    for j in range(n_major+1): # Loop over majors + primary (index 0)
        if j == 0:
            assert major_list[j]['is_primary'], 'Index 0 not primary'
            continue
        ###j
        major_snaps = major_list[j]['snaps']
        major_subfind_ids = major_list[j]['subfind_ids']
        major_nsnaps = len(major_snaps)
        
        for k in range(major_nsnaps):
            this_subhalo = get(snaps[major_snaps[k]]['url']+'subhalos/'+\
                               str(major_subfind_ids[k]), timeout=None)
            assert major_snaps[k] == this_subhalo['snap']
            snap_path = data_dir+'cutouts/snap_'+str(major_snaps[k])+'/'
            print('Downloading subhalo '+str(this_subhalo['id'])+\
                  ' of snapshot '+str(this_subhalo['snap']))
            _=get(this_subhalo['cutouts']['subhalo'],directory=snap_path,
                  timeout=None)
        ###k
    ###j
###i

# Loop over all primaries and download
for i in range(n_mw):
    major_dict = all_major_list[i]
    print('Downloading primary z=0 subhalo '+\
           str(major_dict['primary_z0_subfind_id']))
    this_subhalo = get(snaps[-1]['url']+'subhalos/'+\
                       str(major_dict['primary_z0_subfind_id']), timeout=None)
    snap_path = data_dir+'cutouts/snap_'+str(snaps[-1]['number'])+'/'
    print('Downloading subhalo '+str(this_subhalo['id'])+\
          ' of snapshot '+str(this_subhalo['snap']))
    _=get(this_subhalo['cutouts']['subhalo'],directory=snap_path,timeout=None)
###i