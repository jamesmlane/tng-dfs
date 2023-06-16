# ------------------------------------------------------------------------
#
# TITLE - download_cutouts.py
# AUTHOR - James Lane
# PROJECT - tng-dfs
#
# ------------------------------------------------------------------------
#
# Docstring:
'''Download cutouts for all primaries and major mergers in the TNG simulations.
'''

__author__ = "James Lane"

### Imports

## Basic
import sys, os, pdb
import dill as pickle
import numpy as np

sys.path.insert(0,'../../src/')
from tng_dfs import util as putil

# Keywords
cdict = putil.load_config_to_dict()
keywords = ['DATA_DIR',]
data_dir, = putil.parse_config_dict(cdict,keywords)

# Base URL
baseURL = 'http://www.tng-project.org/api/'
# Get list of simulations
r = putil.get(baseURL)
sim_names = [sim['name'] for sim in r['simulations']]
tng50_indices = [sim_names.index('TNG50-'+str(i+1)) for i in range(4)]
# Choose the lowest resolution tng50 run
tng50_urls = [r['simulations'][i]['url'] for i in tng50_indices]
tng50_url = tng50_urls[0]

# Get the simulation, snapshots, snapshot redshifts
sim = putil.get( tng50_url )
snaps = putil.get( sim['snapshots'] )

# Load primaries and major mergers
with open('../parse_sublink_trees/data/tree_primaries.pkl','rb') as handle:
    tree_primaries = pickle.load(handle)
with open('../parse_sublink_trees/data/tree_major_mergers.pkl','rb') as handle:
    tree_major_mergers = pickle.load(handle)
n_mw = len(tree_primaries)

# First prepare directory structure
for i in range(len(snaps)):
    snap_path = data_dir+'cutouts/snap_'+str(snaps[i]['number'])+'/'
    os.makedirs(snap_path,exist_ok=True)

# Open some log files
log_file = open('./log/download_cutouts.log','w')
exceptions = []

# First loop over all primaries and download cutouts for each
txt = 'Downloading cutouts for all primaries...\n'+'-'*50
print(txt)
log_file.write(txt+'\n')
for i in range(n_mw):
    if i > 10:
        continue

    primary = tree_primaries[i]
    z0_sid = primary.subfind_id[0]
    txt = 'Getting cutouts for primary with z=0 subfind id: '+str(z0_sid)+'...'
    print(txt)
    log_file.write(txt+'\n')
    n_snap = len(primary.snapnum)
    primary_has_data = np.zeros(n_snap,dtype=bool)
    for j in range(n_snap):
        sn = primary.snapnum[j]
        sid = primary.subfind_id[j]
        snap_path = data_dir+'cutouts/snap_'+str(sn)+'/'
        snap_filename = snap_path+'cutout_'+str(sid)+'.hdf5'
        if os.path.isfile(snap_filename):
            primary_has_data[j] = True
            txt = 'Already have cutout for subhalo '+str(sid)+' of snapshot '+\
                str(sn)
            log_file.write(txt+'\n')
            continue
        # Fetch the subhalo
        subhalo = putil.get(snaps[sn]['url']+'subhalos/'+str(sid),
            timeout=None)
        # Some consistency checks
        assert sn == subhalo['snap']
        assert sid == subhalo['id']
        # Download the cutout
        txt = 'Downloading subhalo '+str(sid)+' of snapshot '+str(sn)
        print(txt)
        log_file.write(txt+'\n')
        try:
            _=putil.get(subhalo['cutouts']['subhalo'],directory=snap_path,
                timeout=None)
        except Exception as e:
            exceptions.append((e,sn,sid))
            txt = 'Exception raised for subhalo '+str(sid)+' of snapshot '+\
                str(sn)
            print(txt)
            log_file.write(txt+'\n')
    # Communicate if all cutouts already exist
    if np.all(primary_has_data):
        print('Already have all cutouts for primary')


# Now loop over all primaries / major mergers and download cutouts
txt = '\n\n\nDownloading cutouts for major mergers...\n'+'-'*50
print(txt)
log_file.write(txt+'\n')
for i in range(n_mw):
    if i > 10:
        continue

    primary = tree_primaries[i]
    z0_sid = primary.subfind_id[0]
    txt = 'Getting major merger cutouts for primary with z=0 subfind id: '+\
        str(z0_sid)+'...'
    print(txt)
    log_file.write(txt+'\n')
    n_major = primary.n_major_mergers
    for j in range(n_major):
        print('Getting cutouts for major merger '+str(j+1)+' of '+str(n_major))
        major_merger = primary.tree_major_mergers[j]
        n_snap = len(major_merger.snapnum)
        major_has_data = np.zeros(n_snap,dtype=bool)

        for k in range(n_snap):

            sn = major_merger.snapnum[k]
            sid = major_merger.subfind_id[k]
            snap_path = data_dir+'cutouts/snap_'+str(sn)+'/'
            snap_filename = snap_path+'cutout_'+str(sid)+'.hdf5'
            if os.path.isfile(snap_filename):
                major_has_data[k] = True
                continue
            # Fetch the subhalo
            subhalo = putil.get(snaps[sn]['url']+'subhalos/'+str(sid),
                timeout=None)
            # Some consistency checks
            assert sn == subhalo['snap']
            assert sid == subhalo['id']
            # Download the cutout
            txt = 'Downloading subhalo '+str(sid)+' of snapshot '+str(sn)
            print(txt)
            log_file.write(txt+'\n')
            try:
                _=putil.get(subhalo['cutouts']['subhalo'],directory=snap_path,
                    timeout=None)
            except Exception as e:
                exceptions.append((e,sn,sid))
                txt = 'Exception raised for subhalo '+str(sid)+' of snapshot '+\
                    str(sn)
                print(txt)
                log_file.write(txt+'\n')
        
        # Communicate if all cutouts already exist
        if np.all(major_has_data):
            print('Already have all cutouts for major merger')

# Close log file
log_file.close()

# Save exceptions
with open('./log/exceptions.pkl','wb') as handle:
    pickle.dump(exceptions,handle)