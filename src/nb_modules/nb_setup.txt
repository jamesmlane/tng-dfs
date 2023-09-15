# Keywords
cdict = putil.load_config_to_dict()
keywords = ['DATA_DIR','MW_ANALOG_DIR','RO','VO','ZO','LITTLE_H',
            'MW_MASS_RANGE']
data_dir,mw_analog_dir,ro,vo,zo,h,mw_mass_range = \
    putil.parse_config_dict(cdict,keywords)

# MW Analog 
mwsubs,mwsubs_vars = putil.prepare_mwsubs(mw_analog_dir,h=h,
    mw_mass_range=mw_mass_range,return_vars=True,force_mwsubs=False)

# Figure path
fig_dir = './fig/sample/'
epsen_fig_dir = '/epsen_data/scr/lane/projects/tng-dfs/figs/notebooks/sample/'
os.makedirs(fig_dir,exist_ok=True)
os.makedirs(epsen_fig_dir,exist_ok=True)
show_plots = False

# Load tree data
with open('../parse_sublink_trees/data/tree_primaries.pkl','rb') as handle:
    tree_primaries = pickle.load(handle)
with open('../parse_sublink_trees/data/tree_major_mergers.pkl','rb') as handle:
    tree_major_mergers = pickle.load(handle)
n_mw = len(tree_primaries)