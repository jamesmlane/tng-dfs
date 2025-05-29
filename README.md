# TNG Distribution Functions

This repository contains the code used for the paper [Lane & Bovy 2025](https://arxiv.org/abs/2405.10945).

## Requirements

These packages must be installed to run the code in this repository:
- numpy
- scipy
- matplotlib
- h5py
- astropy
- galpy (must be > v1.9 to perform distribution function calculations)

These packages are required but could be worked around with minor modifications to the code:
- dill (could be replaced with pickle, but we encountered challenges with pickling DF objects and so don't recommend omitting dill)
- emcee (could be replaced with another MCMC sampler)
- corner (could be replaced with another corner plotter, or corner plots can be omitted)
- sklearn (A deprecated part of the code uses sklearn, so this shouldn't come up as a problem if it's not installed)

## Usage

### The config file

The config file contains some high-level parameters used throughout the project, stored in this file for consistency. The most important keywords are for pathing. The supported keywords are:

- `DATA_DIR`: The directory where the data is stored. This is a large directory, requiring about 1TB of space. All of the data and most of the figures are stored here. The majority of the storage space is taken up by the N-body snapshots, which are not included in this repository.
- `MW_ANALOG_DIR`: The directory where the MW-analog data is stored. This constitutes the principal analysis of the paper. It should probably be located within `DATA_DIR` (In fact `DATA_DIR` and `MW_ANALOG_DIR` could probably be the same directory if the code were re-written, but as it stands they are different, but confusingly very similar directories that are accessed by different keywords). We recommend $DATA_DIR/mw_analogs/ as the path.
- `FIG_DIR_BASE`: Base directory for the figure storage. Could be located at the same directory level as `DATA_DIR`, but is independent for flexibility. Probably requires about 5 GB of space.
- `FITTING_DIR_BASE`: Base directory for the fitting results. Could be located at the same directory level as `DATA_DIR`, but is independent for flexibility. Probably requires about 30 GB of space.
- `RO`/`VO`/`ZO`: Scales used for galpy. See galpy documentation for more information. Where necessary we use `RO=8.275` (Gravity Collab.+ 2021), `VO=220`, and `ZO=0.0208` (Bennett+2018). Since the analysis doesn't deal with real data, these values are not critical. They just allow galpy to be used with physical units.
- `LITTLE_H`: The Hubble constant. We used `0.6774` (Planck Collab. 2015, note this is the value used in the IllustrisTNG simulations).
- `MW_MASS_RANGE`: The stellar mass range for Milky Way analogs in units of 10^10 solar masses. We use [5,7].

There are many other 'keywords' used throughout the analysis. These are not stored in the config file but their location in the notebooks should be apparent enough. And the user can refer to the paper for details on parameter choice.

### Pathing

Locally, the notebooks should only save/store about 500 MB of data within the structure of the repository, and the user shouldn't need to create any directories, but if they do it should be fairly obvious in the code (i.e. need to create a ./fig/ directory for a notebook to save a figure). Most of the data (About 1TB) are stored in `DATA_DIR`. We recommend the following structure for the data-heavy directories keyworded above:

- SOME_DIRECTORY_WITH_LOTS_OF_SPACE
    - $DATA_DIR
        - $MW_ANALOG_DIR
    - $FIG_DIR_BASE
    - $FITTING_DIR_BASE

### Running the code

The notebooks are organized in the order they should be run. Running the notebooks in this order should produce the same results as in the paper. Errors should by minimal and hopefully restricted to minor pathing inconsistencies or post-run edits to the notebooks. All should be within the scope of the user to fix with minor edits.

The notebooks are organized as follows (headers are directories within `./notebooks`):

#### 0_preparation

- `1-prepare_project_paths.ipynb`: This notebook just prepares some paths for the rest of the projects. Since not all code creates the directories it will use/save data to, it's worthwhile to run this notebook to setup some paths within `DATA_DIR` and `MW_ANALOG_DIR`

#### 1_analog_sample

- `1_tng_analog_sample.ipynb`: Select the MW-analogs from the TNG50 simulation.
- `2_download_primary_cutouts.ipynb`: Download the cutouts for the primary MW-analogs over all snapshots. This is quite time consuming.
- `3_spheroid_disk_decomposition.ipynb`: Decompose the MW-analogs into spheroid and disk components in order to pare down the sample into those which are disk-like (kept in the analysis) and those which are not (excluded from the analysis).
- `4_pillepich23_analog_sample.ipynb`: Take a quick look at the Pillepich+2023 MW analog sample. Not necessary to run.

#### 2_merger_sample

- `1_parse_sublink_trees.ipynb`: Examine merger trees for each primary to find major mergers.
- `2_download_secondary_cutouts.ipynb`: Download the secondary cutouts corresponding to the major mergers. This can be somewhat time consuming.

#### 3_fit_density_profiles

- `1_interpolate_primaries.ipynb`: Create spherical representations of the host potential (DM and stars) for each primary at z=0.
- `2_fit_stellar_halo_density.ipynb`: Fit the stellar halo density profile for each secondary major merger remnant at z=0.

#### 4_fit_distribution_functions

- `1_fit_anisotropy.ipynb`: Fit simple anisotropy models (constant anisotropy and Osipkov-Merritt) to the kinematics of each remnant in the stellar halo.
- `2_fit_om_2_combination.ipynb`: Fit the linear combination Osipkov-Merritt model to the kinematics of each remnant in the stellar halo. This is much more involved and can be quite time consuming.
- `3_fit_stellar_halo_rotation.ipynb`: Fit the rotating DF prescription to the kinematics of each remnant in the stellar halo.

#### 5_compare_distribution_functions

- `0_make_merger_info.ipynb`: Create and stash a table of the merger information for each remnant and the corresponding primary.
- `1_construct_anisotropy_dfs.ipynb`: Create the anisotropy DFs for each model for each remnant. This is somewhat time consuming.
- `1.1_sample_anisotropic_dfs.ipynb`: Draw representative samples from the anisotropic DFs for each remnant.
- `2_anisotropic_df_dispersions.ipynb`: Compute velocity dispersion profiles, anisotropy profiles, and a simple comparison metric for further analysis for each DF and each remnant for the samples and N-body data.
- `3_anisotropic_df_likelihoods.ipynb`: Compute the likelihoods of the anisotropic DF samples and N-body data.
- `4_anisotropic_df_jeans.ipynb`: Assess the DF samples and the N-body remnants in the context of the Jeans equation.
- `5_close_look_merger_sample.ipynb`: Define a few remnants as being representative for close looks (e.g. a GS/E analog) and make some more detailed plots of the dispersions, anisotropy profiles, and energy/angular momentum planes.
- `6_higher_order_moments.ipynb`: Compute some higher order moments of the anisotropy DF samples. Not necessary to run.

#### paper
`1_galaxy_sample.ipynb`: Make plots of the overall properties of the MW analog sample for the paper. Currently only 1 figure.
`2_spheroid_disk_decomposition.ipynb`: Make plots of the spheroid/disk decomposition process for the paper. Currently only 1 figure.
`3_merger_stats.ipynb`: Make plots of the merger stats for the paper. Currently only 1 figure.
`4_df_stats.ipynb`: Make plots of the DF stats for the paper. Currently only 1 figure.

Many of the paper figures are made in `./notebooktes/5_compare_distribution_functions/`.

### src directory

The `src` directory contains code used across multiple notebooks. Each notebook accesses the `src` directory by adding it to `sys.path`. See any notebook for an example of how this is done. The python code in `src/tng_dfs/` is organized like a standard python project and is imported throughout the project. The modules are fairly self-explanatory, and have docstrings explaining how they work. `src/mpl` contains a `.mplstyle` file for nice looking plots. This is loaded in each notebook but can easily be replaced/removed. `src/nb_modules` contains text files that are loaded in each notebook using the `%load ../../src/nb_modules/...` magic. This just helps to standardize the code across notebooks. It's probably not necessary to re-run the magic for each notebook, but it can be done for consistency.