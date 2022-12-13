# STAR SHADOW
### Satellite Time-series Analysis Routine using Sinusoids and Harmonics Automatedly for Double stars with Occultations and Waves


## What is STAR SHADOW?
STAR SHADOW is a Python code that is aimed at the automatic analysis of space based light curves of eclipsing binaries and provide a measurement of eccentricity, among other parameters. It provides a recipe to measure timings of eclipses using the time derivatives of the light curves, using a model of orbital harmonics obtained from an initial iterative prewhitening of sinusoids. Since the algorithm extracts the harmonics from the rest of the sinusoidal variability eclipse timings can be measured even in the presence of other (astrophysical) signals.
The aim is to determine the orbital eccentricity automatically from the light curve along with information about the other variability present in the light curve. The output includes, but is not limited to, a sinusoid plus piece-wise linear model of the light curve, the orbital period, the eccentricity, argument of periastron and inclination. See the documentation for more information.


### Reference Material

* This algorithm has been documented, tested and applied in the publication: [link](link)


## Getting started

As of version 0.1.0 (August 2022), it is possible to install STAR SHADOW with pip using:

    pip install git+https://github.com/LucIJspeert/star_shadow@v0.1.0

Or install the master branch by leaving out the version number. One can then import the package from the python environment it was installed in. Of course one can always still manually download it or make a fork on GitHub. It is recommended to get the latest release from the GitHub page. 

**STAR SHADOW has only been tested in Python 3.7**. Using older versions could result in unexpected errors, although any Python version >3.6 is expected to work.


**Package dependencies:** The following package versions have been used in the development of this code, meaning older versions can in principle work, but this is not guaranteed. NumPy 1.20.3, SciPy 1.7.3, Numba 0.55.1, Matplotlib 3.5.3, Arviz 0.11.4, Astropy 4.3.1 (mainly for .fits functionality), h5py 3.7.0 (for saving results). Newer versions are expected to work, and it is considered a bug if this is not the case.

### Example use

Since the main feature of STAR SHADOW is its fully automated operation, taking advantage of its functionality is as simple as running one or two functions:

    import star_shadow as sts
    # to analyse any light curve from a file: 
    sts.analyse_from_file(file, p_orb=0, data_id=None, overwrite=False, verbose=True)
    # or to analyse from a set of TESS data product .fits files:
    sts.analyse_from_tic(tic, all_files, p_orb=0, save_dir=None, data_id=None, overwrite=False, verbose=True)

The light curve file is expected to contain a time column, flux measurements (median normalised and non-negative), and flux measurement errors. The normalisation for TESS data products is handled automatically on a per-sector basis. 

If a save_dir is given, the outputs are saved in a folder with either the TIC number or the file name as identifier. The overwrite argument can be used to overwrite old data or to continue from a previous save file. The functions can print useful progress information if verbose=True. If an orbital period is known beforehand, this information will be used to find orbital harmonics in the prewhitened frequencies. If left zero, a period is found through a combination of phase dispersion minimisation and Lomb-Scargle periodograms. For the analyse_from_tic function, the files corresponding to the given TIC number are picked out from a list of all available TESS data files, for ease of use.

Either function can be used for a set of light curves by using:

    sts.analyse_set(target_list, function='analyse_from_tic', n_threads=os.cpu_count() - 2, **kwargs):


### Explanation of output

Results are saved in a combination of hdf5 and csv files. A log file keeps track of the start and end time of the analysis and can contain important information about the operation of the algorithm.

Currently, there are a total of 19 analysis steps. Normal operation can terminate at several intermediate stages: 2, 3, 10, 13 and 14. A log entry is made when this happens containing further information. If it has stopped at stage 3, either the period found was too long for the given data set, or not enough orbital harmonics are found. If stage 10 is reached, but nothing further, this means that the algorithm wasn't able to detect the two eclipses (being the primary and secondary eclipse): both eclipses are needed for the further analysis.

In normal operation, the nine first steps produce .hdf5 files with all the model parameters at that stage of the analysis. The utility module contains a function for reading these files, however, separate .csv files are also produced at the end of these nine prewhitening steps for easy access. All following steps produce one or more .csv files with the results.

### Diagnostic plots

There are several plotting functions available that show various diagnostics from throughout the analysis. The function:

    sts.ut.sequential_plotting(tic, times, signal, i_sectors,  load_dir, save_dir=None, show=False)

saves and/or shows all these plots for one target. Unfortunately matplotlib plotting only works in the main thread, so when processing a whole set of light curves in parallel, this function will have to be run sequentially on the results afterwards (hence the name).

## Bugs and Issues

Despite all the testing, I am certain that there are still bugs in this code, or will be created in future versions. 

If you happen to come across any bugs or issues, *please* contact me. Only known bugs can be resolved.
This can be done through opening an issue on the STAR SHADOW GitHub page: [LucIJspeert/star_shadow/issues](https://github.com/LucIJspeert/star_shadow/issues), or by contacting me directly (see below).

If you are (going to be) working on new or improved features, I would love to hear from you and see if it can be implemented in the source code.


## Contact

For questions and suggestions, please contact:

* luc.ijspeert(at)kuleuven.be

**Developer:** Luc IJspeert (KU Leuven)
