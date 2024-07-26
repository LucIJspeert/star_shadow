# STAR SHADOW
### Satellite Time-series Analysis Routine using Sinusoids and Harmonics Automatedly for Double stars with Occultations and Waves


![Language Badge](https://img.shields.io/badge/Language-Python-blue.svg)
<a href="./LICENCE.md"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License Badge"/></a>

[//]: # (<a href="https://github.com/LucIJspeert/star_shadow/blob/master/LICENCE.md"><img src="https://img.shields.io/github/license/LucIJspeert/star_shadow" alt="License Badge"/></a>)

[//]: # (make the badges dynamic...)

## What is STAR SHADOW?
STAR SHADOW is a Python code that is aimed at the automatic analysis of space based light curves of eclipsing binaries 
and provide a measurement of eccentricity, among other parameters. It provides a recipe to measure timings of eclipses 
using the time derivatives of the light curves, using a model of orbital harmonics obtained from an initial iterative 
prewhitening of sinusoids. Since the algorithm extracts the harmonics from the rest of the sinusoidal variability 
eclipse timings can be measured even in the presence of other (astrophysical) signals. The aim is to determine 
the orbital eccentricity automatically from the light curve along with information about the other variability present 
in the light curve. The output includes, but is not limited to, a sinusoid plus linear model of 
the light curve, the orbital period, the eccentricity, argument of periastron and inclination. See the documentation 
for more information.


### Reference Material

* This algorithm has been documented, tested and applied in the publication: [Automated eccentricity measurement from raw eclipsing binary light curves with intrinsic variability](https://ui.adsabs.harvard.edu/abs/2024arXiv240206084I/abstract)


![Example of an analysed EB light curve](star_shadow/data/sim_042_lc_readme_img.png?raw=true)

## Getting started

The easiest way to install STAR SHADOW is to use pip:

    pip install git+https://github.com/LucIJspeert/star_shadow@<version_tag>

Or install the master branch by leaving out the version number. 
One can then import the package from the python environment it was installed in. 
Of course one can always still manually download it or make a fork on GitHub. 
It is recommended to get the latest release from the GitHub page. 

**STAR SHADOW has only been tested in Python 3.7**. Using older versions could result in unexpected errors, 
although any Python version >=3.6 is expected to work.

**Package dependencies:** The following package versions have been used in the development of this code, 
meaning older versions can in principle work, but this is not guaranteed. NumPy 1.20.3, SciPy 1.7.3, Numba 0.55.1, 
h5py 3.7.0, Astropy 4.3.1, Matplotlib 3.5.3, Arviz 0.11.4, corner 2.2.1, pymc3 3.11.4 (optional), 
theano 1.1.2 (optional), fastprogress 1.0.0 (optional), ellc 1.8.5 (optional).

Newer versions are expected to work, and it is considered a bug if this is not the case.

Before first use, it is recommended to run one very short time-series (for example sim_000_lc.dat included in the data 
folder). This will make sure that the just-in-time compiler can do its magic and make everything run as fast as it can. See the 
script run_first_use.py.


### Example use

Since the main feature of STAR SHADOW is its fully automated operation, taking advantage of its functionality is 
as simple as running one function:

    import star_shadow as sts
    # to analyse any light curve from a file: 
    sts.analyse_lc_from_file(file, p_orb=0, i_sectors=None, stage='all', method='fitter', data_id='none', save_dir=None, overwrite=False, verbose=True)
    
    # or to analyse from a set of TESS data product .fits files:
    sts.analyse_lc_from_tic(tic, all_files, p_orb=0, i_sectors=None, stage='all', method='fitter', data_id=None, save_dir=None, overwrite=False, verbose=True)

The light curve file is expected to contain a time column, flux measurements (median normalised and non-negative), 
and flux measurement errors. The normalisation for TESS data products is handled automatically on a per-sector basis. 
The stage parameter can be set to indicate which parts of the analysis are performed, see the documentation for options.

If a save_dir is given, the outputs are saved in that directory with either the TIC number or the file name as 
identifier. If not given, files are saved in a subdirectory of where the light curve file is.
The 'overwrite' argument can be used to overwrite old data or to continue from a previous save file. The functions can 
print useful progress information if verbose=True. If an orbital period is known beforehand, this information 
will be used to find orbital harmonics in the prewhitened frequencies. If left zero, a period is found through 
a combination of phase dispersion minimisation, Lomb-Scargle periodogram and extracted frequencies. For the 
'analyse_from_tic' function, the files corresponding to the given TIC number are picked out from a list of all 
available TESS data files, provided by the user, for ease of use.

Either function can be used for a set of light curves by using:

    sts.analyse_set(target_list, function='analyse_from_tic', n_threads=os.cpu_count() - 2, **kwargs):


### Explanation of output

Results are saved mainly in hdf5 files, in addition to csv and sometimes nc4 files. A plain text log file keeps track 
of the start and end time of the analysis and can contain important messages about the operation of the algorithm, 
like a reason for early termination.

Currently, there are a total of 9 analysis steps. Normal operation can terminate at several intermediate stages. 
A log entry is made when this happens containing further information. The analysis can stop if for example no 
frequencies were extracted, the period is too long for the given data set, not enough orbital harmonics are found, 
or primary or secondary eclipse cannot be identified.

Each step produces at least an .hdf5 file with all the model parameters from that stage of the analysis. 
The utility module contains a function for reading these files, 'read_parameters_hdf5', which outputs a convenient 
format for the data (note that reading in these files with H5py directly will not result in formatting that can be used 
with the functions of STAR SHADOW). The .hdf5 files can also be translated into several plain text .csv files with 
'convert_hdf5_to_ascii'. The .nc4 files (a wrapper for hdf5) contain pymc3 sampling chains.


### Diagnostic plots

There are several plotting functions available that show various diagnostics from throughout the analysis. The function:

    sts.ut.sequential_plotting(times, signal, i_sectors, target_id, load_dir, save_dir=None, show=False)

saves and/or shows most of the available plots for one target. Unfortunately matplotlib plotting only works in
the main thread, so when processing a whole set of light curves in parallel, this function will have to be run 
sequentially on the results afterwards (hence the name).


## Bugs and Issues

Despite all the testing, I am certain that there are still bugs in this code, or will be created in future versions. 

If you happen to come across any bugs or issues, *please* contact me. Only known bugs can be resolved.
This can be done through opening an issue on the STAR SHADOW GitHub page: 
[LucIJspeert/star_shadow/issues](https://github.com/LucIJspeert/star_shadow/issues), 
or by contacting me directly (see below).

If you are (going to be) working on new or improved features, I would love to hear from you and see if it can be 
implemented in the source code.


## Contact

For questions and suggestions, please contact:

* luc.ijspeert(at)kuleuven.be

**Developer:** Luc IJspeert (KU Leuven)
