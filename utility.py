"""STAR SHADOW
Satellite Time-series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This module contains utility functions for data processing, unit conversions
and loading in data (some functions specific to TESS data).

Code written by: Luc IJspeert
"""

import os
import datetime
import h5py
import numpy as np
import numba as nb
import astropy.io.fits as fits

from . import timeseries_functions as tsf
from . import analysis_functions as af
from . import visualisation as vis


@nb.njit(cache=True)
def weighted_mean(x, w):
    """Jitted weighted mean since Numba doesn't support numpy.average
    
    Parameters
    ----------
    x: numpy.array[float]
        Values to calculate the mean over
    w: numpy.array[float]
        Weights corresponding to each value
    
    Returns
    -------
    weighted_mean: float
        Mean of x weighted by w
    """
    weighted_mean = np.sum(x * w) / np.sum(w)
    return weighted_mean


@nb.njit(cache=True)
def decimal_figures(x, n_sf):
    """Determine the number of decimal figures to print given a target
    number of significant figures
    
    Parameters
    ----------
    x: float
        Value to determine the number of decimals for
    n_sf: int
        Number of significant figures to compute
    
    Returns
    -------
    decimals: int
        Number of decimal places to round to
    """
    if (x != 0):
        decimals = (n_sf - 1) - int(np.floor(np.log10(abs(x))))
    else:
        decimals = 1
    return decimals


@nb.njit(cache=True)
def signal_to_noise_threshold(n_points):
    """Determine the signal to noise threshold for accepting frequencies
    based on the number of points
    
    Parameters
    ----------
    n_points: int
        Number of data points in the time-series
    
    Returns
    -------
    sn_thr: float
        Signal-to-noise threshold for this data set
    
    Notes
    -----
    Baran & Koen 2021, eq 6.
    (https://ui.adsabs.harvard.edu/abs/2021AcA....71..113B/abstract)
    """
    sn_thr = 1.201 * np.sqrt(1.05 * np.log(n_points) + 7.184)
    sn_thr = np.round(sn_thr, 2)  # round to two decimals
    return sn_thr


@nb.njit(cache=True)
def normalise_counts(flux_counts, i_sectors):
    """Median-normalises flux (counts or otherwise, should be positive) by
    dividing by the median.
    
    Parameters
    ----------
    flux_counts: numpy.ndarray[float]
        Flux measurement values in counts of the time-series
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans.
        These can indicate the TESS observation sectors, but taking
        half the sectors is recommended. If only a single curve is
        wanted, set i_half_s = np.array([[0, len(times)]]).
    
    Returns
    -------
    flux_norm: numpy.ndarray[float]
        Normalised flux measurements
    median: numpy.ndarray[float]
        Median flux counts per sector
    
    Notes
    -----
    The result is positive and varies around one.
    The signal is processed per sector.
    """
    median = np.zeros(len(i_sectors))
    flux_norm = np.zeros(len(flux_counts))
    for i, s in enumerate(i_sectors):
        median[i] = np.median(flux_counts[s[0]:s[1]])
        flux_norm[s[0]:s[1]] = flux_counts[s[0]:s[1]] / median[i]
    return flux_norm, median


def get_tess_sectors(times, bjd_ref=2457000.0):
    """Load the times of the TESS sectors from a file and return a set of
    indices indicating the separate sectors in the time-series.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time-series
    bjd_ref: float
        BJD reference date
        
    Returns
    -------
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the TESS observing sectors
    
    Notes
    -----
    Make sure to use the appropriate Baricentric Julian Date (BJD)
    reference date for your data set. This reference date is subtracted
    from the loaded sector dates.
    """
    # the 0.5 offset comes from test results, and the fact that no exact JD were found (just calendar days)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # absolute dir the script is in
    jd_sectors = np.loadtxt(os.path.join(script_dir, 'tess_sectors.dat'), usecols=(2, 3)) - bjd_ref
    # use a quick searchsorted to get the positions of the sector transitions
    i_start = np.searchsorted(times, jd_sectors[:, 0])
    i_end = np.searchsorted(times, jd_sectors[:, 1])
    sectors_included = (i_start != i_end)  # this tells which sectors it received data for
    i_sectors = np.column_stack([i_start[sectors_included], i_end[sectors_included]])
    return i_sectors


def convert_tess_t_sectors(times, t_sectors):
    """Converts from the sector start and end times to the indices in the array.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time-series
    t_sectors: numpy.ndarray[float]
        Pair(s) of times indicating the timespans of each sector
    
    Returns
    -------
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans.
        These can indicate the TESS observation sectors, but taking
        half the sectors is recommended.
    """
    starts = np.searchsorted(times, t_sectors[:, 0])
    ends = np.searchsorted(times, t_sectors[:, 1])
    i_sectors = np.column_stack((starts, ends))
    return i_sectors


def load_tess_data(file_name):
    """Load in the data from a single fits file, TESS specific.
    
    Parameters
    ----------
    file_name: str, None
        File name (including path) for loading the results.
    
    Returns
    -------
    times: numpy.ndarray[float]
        Timestamps of the time-series
    sap_flux: numpy.ndarray[float]
        Raw measurement values of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    errors: numpy.ndarray[float]
        Errors in the measurement values
    qual_flags: numpy.ndarray[int]
        Integer values representing the quality of the
        data points. Zero means good quality.
    sector: int
        Sector number of the TESS observations
    crowdsap: float
        Light contamination parameter (1-third_light)
    
    Notes
    -----
    The SAP flux is Simple Aperture Phonometry, the processed data
    can be PDC_SAP or KSP_SAP depending on the data source.
    """
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
    # grab the time-series data, sector number, start and stop time
    with fits.open(file_name, mode='readonly') as hdul:
        sector = hdul[0].header['SECTOR']
        times = hdul[1].data['TIME']
        sap_flux = hdul[1].data['SAP_FLUX']
        if ('PDCSAP_FLUX' in hdul[1].data.columns.names):
            signal = hdul[1].data['PDCSAP_FLUX']
            errors = hdul[1].data['PDCSAP_FLUX_ERR']
        elif ('KSPSAP_FLUX' in hdul[1].data.columns.names):
            signal = hdul[1].data['KSPSAP_FLUX']
            errors = hdul[1].data['KSPSAP_FLUX_ERR']
        else:
            signal = np.zeros(len(sap_flux))
            if ('SAP_FLUX_ERR' in hdul[1].data.columns.names):
                errors = hdul[1].data['SAP_FLUX_ERR']
            else:
                errors = np.zeros(len(sap_flux))
            print('Only SAP data product found.')
        # quality flags
        qual_flags = hdul[1].data['QUALITY']
        # get crowding numbers if found
        if ('CROWDSAP' in hdul[1].header.keys()):
            crowdsap = hdul[1].header['CROWDSAP']
        else:
            crowdsap = -1
    return times, sap_flux, signal, errors, qual_flags, sector, crowdsap


def load_tess_lc(tic, all_files, apply_flags=True):
    """Load in the data from (potentially) multiple TESS specific fits files.
    
    Parameters
    ----------
    tic: int
        The TESS Input Catalog number for later reference
        Use any number (or even str) as reference if not available.
    all_files: list[str]
        A list of file names (including path) for loading the results.
        This list is searched for files with the correct tic number.
    apply_flags: bool
        Whether to apply the quality flags to the time-series data
        
    Returns
    -------
    times: numpy.ndarray[float]
        Timestamps of the time-series
    sap_signal: numpy.ndarray[float]
        Raw measurement values of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    sectors: list[int]
        List of sector numbers of the TESS observations
    t_sectors: numpy.ndarray[float]
        Pair(s) of times indicating the timespans of each sector
    crowdsap: list[float]
        Light contamination parameter (1-third_light) listed per sector
    """
    tic_files = [file for file in all_files if f'{tic:016.0f}' in file]
    times = np.array([])
    sap_signal = np.array([])
    signal = np.array([])
    signal_err = np.array([])
    qual_flags = np.array([])
    sectors = np.array([])
    t_sectors = []
    crowdsap = np.array([])
    for file in tic_files:
        # get the data from the file
        ti, s_fl, fl, err, qf, sec, cro = load_tess_data(file)
        dt = np.median(np.diff(ti[~np.isnan(ti)]))
        # keep track of the start and end time of every sector
        t_sectors.append([ti[0] - dt/2, ti[-1] + dt/2])
        # append all other data
        times = np.append(times, ti)
        sap_signal = np.append(sap_signal, s_fl)
        signal = np.append(signal, fl)
        signal_err = np.append(signal_err, err)
        qual_flags = np.append(qual_flags, qf)
        sectors = np.append(sectors, sec)
        crowdsap = np.append(crowdsap, cro)
    t_sectors = np.array(t_sectors)
    # apply quality flags
    if apply_flags:
        # convert quality flags to boolean mask
        quality = (qual_flags == 0)
        times = times[quality]
        sap_signal = sap_signal[quality]
        signal = signal[quality]
        signal_err = signal_err[quality]
    # clean up (only on times and signal, sap_signal assumed to be the same)
    finites = np.isfinite(times) & np.isfinite(signal)
    times = times[finites].astype(np.float_)
    sap_signal = sap_signal[finites].astype(np.float_)
    signal = signal[finites].astype(np.float_)
    signal_err = signal_err[finites].astype(np.float_)
    return times, sap_signal, signal, signal_err, sectors, t_sectors, crowdsap


def stitch_tess_sectors(times, signal, i_sectors):
    """Stitches the different TESS sectors of a light curve together.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time-series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the TESS observing sectors
    
    Returns
    -------
    times: numpy.ndarray[float]
        Timestamps of the time-series (shifted start at zero)
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    medians: numpy.ndarray[float]
        Median flux counts per sector
    t_start: float
        Original starting point of the time-series
    t_combined: numpy.ndarray[float]
        Pair(s) of times indicating the timespans of each half sector
    i_half_s: numpy.ndarray[int]
        Pair(s) of indices indicating the timespans of each half sector
    
    Notes
    -----
    The flux/counts are median-normalised per sector. The median values are returned.
    Each sector is divided in two and the timestamps are provided, since the
    momentum dump happens in the middle of each sector, which can cause a jump in the flux.
    It is recommended that these half-sector timestamps be used in the further analysis.
    The time of first observation is subtracted from all other times, for better numerical
    performance when deriving sinusoidal phase information. The original start point is given.
    """
    # median normalise
    signal, medians = normalise_counts(signal, i_sectors=i_sectors)
    # zero the timeseries
    t_start = times[0]
    times -= t_start
    # times of sector mid point and resulting half-sectors
    dt = np.median(np.diff(times))
    t_start = times[i_sectors[:, 0]] - dt/2
    t_end = times[i_sectors[:, 1] - 1] + dt/2
    t_mid = (t_start + t_end) / 2
    t_combined = np.column_stack((np.append(t_start, t_mid + dt/2), np.append(t_mid - dt/2, t_end)))
    i_half_s = convert_tess_t_sectors(times, t_combined)
    return times, signal, medians, t_start, t_combined, i_half_s


def group_fequencies_for_fit(a_n, g_min=20, g_max=25):
    """Groups frequencies into sets of 10 to 15 for multi-sine fitting
    
    Parameters
    ----------
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    g_min: int
        Minimum group size
    g_max: int
        Maximum group size (g_max > g_min)
    
    Returns
    -------
    groups: list[numpy.ndarray[int]]
        List of sets of indices indicating the groups
    
    Notes
    -----
    To make the task of fitting more managable, the free parameters are binned into groups,
    in which the remaining parameters are kept fixed. Frequencies of similar amplitude are
    grouped together, and the group cut-off is determined by the biggest gaps in amplitude
    between frequencies, but group size is always kept between g_min and g_max. g_min < g_max.
    The idea of using amplitudes is that frequencies of similar amplitude have a similar
    amount of influence on each other.
    """
    # keep track of which freqs have been used with the sorted indices
    not_used = np.argsort(a_n)[::-1]
    groups = []
    while (len(not_used) > 0):
        if (len(not_used) > g_min + 1):
            a_diff = np.diff(a_n[not_used[g_min:g_max + 1]])
            i_max = np.argmin(a_diff)  # the diffs are negative so this is max absolute difference
            i_group = g_min + i_max + 1
            group_i = not_used[:i_group]
        else:
            group_i = np.copy(not_used)
            i_group = len(not_used)
        not_used = np.delete(not_used, np.arange(i_group))
        groups.append(group_i)
    return groups


@nb.njit()
def correct_for_crowdsap(signal, crowdsap, i_sectors):
    """Correct the signal for flux contribution of a third source
    
    Parameters
    ----------
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    crowdsap: list[float], numpy.ndarray[float]
        Light contamination parameter (1-third_light) listed per sector
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans.
        These can indicate the TESS observation sectors, but taking
        half the sectors is recommended. If only a single curve is
        wanted, set i_half_s = np.array([[0, len(times)]]).
    
    Returns
    -------
    cor_signal: numpy.ndarray[float]
        Measurement values of the time-series corrected for
        contaminating light
    
    Notes
    -----
    Uses the parameter CROWDSAP included with some TESS data.
    flux_corrected = (flux - (1 - crowdsap)) / crowdsap
    where all quantities are median-normalised, including the result.
    This corresponds to subtracting a fraction of (1 - crowdsap) of third light
    from the (non-median-normalised) flux measurements.
    """
    cor_signal = np.zeros(len(signal))
    for i, s in enumerate(i_sectors):
        crowd = min(max(0, crowdsap[i]), 1)  # clip to avoid unphysical output
        cor_signal[s[0]:s[1]] = (signal[s[0]:s[1]] - 1 + crowd) / crowd
    return cor_signal


@nb.njit()
def model_crowdsap(signal, crowdsap, i_sectors):
    """Incorporate flux contribution of a third source into the signal
    
    Parameters
    ----------
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    crowdsap: list[float], numpy.ndarray[float]
        Light contamination parameter (1-third_light) listed per sector
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans.
        These can indicate the TESS observation sectors, but taking
        half the sectors is recommended. If only a single curve is
        wanted, set i_half_s = np.array([[0, len(times)]]).
    
    Returns
    -------
    model: numpy.ndarray[float]
        Model of the signal incorporating light contamination
    
    Notes
    -----
    Does the opposite as correct_for_crowdsap, to be able to model the effect of
    third light to some degree (can only achieve an upper bound on CROWDSAP).
    """
    model = np.zeros(len(signal))
    for i, s in enumerate(i_sectors):
        crowd = min(max(0, crowdsap[i]), 1)  # clip to avoid unphysical output
        model[s[0]:s[1]] = signal[s[0]:s[1]] * crowd + 1 - crowd
    return model


def check_crowdsap_correlation(min_third_light, i_sectors, crowdsap, verbose=False):
    """Check the CROWDSAP correlation with the data-extracted third light.

    Parameters
    ----------
    min_third_light: numpy.ndarray[float]
        Minimum amount of third light present per sector
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans.
        These can indicate the TESS observation sectors, but taking
        half the sectors is recommended. If only a single curve is
        wanted, set i_half_s = np.array([[0, len(times)]]).
    crowdsap: list[float], numpy.ndarray[float]
        Light contamination parameter (1-third_light) listed per sector
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    corr: float
        correlation found with the measured third light
        and crowdsap paramter
    check: bool
        Can be used to decide whether the data needs a manual check
    
    Notes
    -----
    If the CROWDSAP parameter from the TESS data is anti-correlated with the similar
    measure fitted from the light curve, it may indicate that the source is not the
    eclipsing binary but a neighbouring star is. This would indicate further
    investigation of the target pixels is needed.
    The fit for third light can be unreliable for few sectors and/or few eclipses.
    """
    # do the fit
    n_sectors = len(i_sectors)
    if verbose:
        if np.any(min_third_light > 0.05):
            print(f'Found third light above 0.05 for one or more sectors ({np.sum(min_third_light > 0.05)}).')
        elif (n_sectors > 1):
            print('Third light correction minimal (<5%) for all sectors.')
        else:
            print('Only one sector: no minimum third light can be infered.')
    # first convert the third light to crowdsap-like fraction
    max_crowd = 1 - min_third_light
    if (len(max_crowd) > 1):
        corr_coef = np.corrcoef(max_crowd, crowdsap)
        corr = corr_coef[0, 1]
    else:
        # not enough data
        corr = 0
    # decide to flag for check or not
    if (corr > 0.7):
        check = False
        s1, s2 = 'a strong', ', indicative of insufficient compensation'
    elif (corr > 0.5):
        check = False
        s1, s2 = 'a moderate', ', indicative of insufficient compensation'
    elif (corr > 0.3):
        check = False
        s1, s2 = 'only a weak', ', indicative of slightly incomplete compensation'
    elif (corr > -0.3):
        check = False
        s1, s2 = 'no significant', ''
    elif (corr > -0.5):
        check = True
        s1, s2 = 'only a weak', ', indicative that the target might not be the EB'
    elif (corr > -0.7):
        check = True
        s1, s2 = 'a moderate', ', indicative that the target might not be the EB'
    else:  # (corr <= -0.7)
        check = True
        s1, s2 = 'a strong', ', indicative that the target might not be the EB'

    if verbose:
        print(f'There is {s1} correlation between measured minimum third light and '
              f'CROWDSAP parameter{s2}. Corr={corr:1.3f}')
    return corr, check


def save_results(results, errors, stats, file_name, description='none', dataset='none'):
    """Save the full output of the frequency analysis function to an hdf5 file.
    
    Parameters
    ----------
    results: tuple[numpy.ndarray[float]]
        Results containing the following data:
        p_orb, const, slope, f_n, a_n, ph_n
    errors: tuple[numpy.ndarray[float]]
        Error values containing the following data:
        p_err, c_err, sl_err, f_n_err, a_n_err, ph_n_err
    stats: tuple[numpy.ndarray[float]]
        Statistic parameters: n_param, bic, noise_level
    file_name: str, None
        File name (including path) for saving the results.
    description: str
        Optional description of the saved results
    dataset: str
        Optional identifier for the data set used
    
    Returns
    -------
    None
    
    Notes
    -----
    The file contains the data sets (array-like) and attributes
    to describe the data, in hdf5 format.
    """
    # unpack all the variables
    p_orb, const, slope, f_n, a_n, ph_n = results
    p_err, c_err, sl_err, f_n_err, a_n_err, ph_n_err = errors
    n_param, bic, noise_level = stats
    # check some input
    if not file_name.endswith('.hdf5'):
        file_name += '.hdf5'
    # create the file
    with h5py.File(file_name, 'w') as file:
        file.attrs['identifier'] = os.path.splitext(os.path.basename(file_name))[0]  # the file name without extension
        file.attrs['description'] = description
        file.attrs['dataset'] = dataset
        file.attrs['date_time'] = str(datetime.datetime.now())
        file.attrs['n_param'] = n_param  # number of free parameters
        file.attrs['bic'] = bic  # Bayesian Information Criterion of the residuals
        file.attrs['noise_level'] = noise_level  # standard deviation of the residuals
        file.create_dataset('p_orb', data=[p_orb])
        file['p_orb'].attrs['unit'] = 'd'
        file['p_orb'].attrs['p_err'] = p_err
        file['p_orb'].attrs['description'] = 'Orbital period and error estimate.'
        file.create_dataset('const', data=const)
        file['const'].attrs['unit'] = 'median normalised'
        file['const'].attrs['c_err'] = c_err
        file['const'].attrs['description'] = 'y-intercept per analysed sector'
        file.create_dataset('slope', data=slope)
        file['slope'].attrs['unit'] = 'median normalised/d'
        file['slope'].attrs['sl_err'] = sl_err
        file['slope'].attrs['description'] = 'slope per analysed sector'
        file.create_dataset('f_n', data=[f'f_{i + 1}' for i in range(len(f_n))])
        file['f_n'].attrs['unit'] = '1/d'
        file.create_dataset('frequency', data=f_n)
        file.create_dataset('f_n_err', data=f_n_err)
        file.create_dataset('a_n', data=[f'a_{i + 1}' for i in range(len(f_n))])
        file['a_n'].attrs['unit'] = 'median normalised'
        file.create_dataset('amplitude', data=a_n)
        file.create_dataset('a_n_err', data=a_n_err)
        file.create_dataset('ph_n', data=[f'ph_{i + 1}' for i in range(len(f_n))])
        file['ph_n'].attrs['unit'] = 'radians'
        file['ph_n'].attrs['sinusoid'] = 'sine function'
        file['ph_n'].attrs['phase_zero_point'] = 'times[0] (time-series start)'
        file.create_dataset('phase', data=ph_n)
        file.create_dataset('ph_n_err', data=ph_n_err)
    return


def load_results(file_name):
    """Load the full output of the find_eclipses function from the hdf5 file.
    returns an h5py file object, which has to be closed by the user (file.close()).
    
    Parameters
    ----------
    file_name: str, None
        File name (including path) for loading the results.
    
    Returns
    -------
    file: Object
        The hdf5 file object
    """
    file = h5py.File(file_name, 'r')
    return file


def read_results(file_name, verbose=False):
    """Read the full output of the find_eclipses function from the hdf5 file.
    This returns the set of variables as they appear in eclipsr and closes the file.
    
    Parameters
    ----------
    file_name: str, None
        File name (including path) for loading the results.
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    results: tuple[numpy.ndarray[float]]
        Results containing the following data:
        p_orb, const, slope, f_n, a_n, ph_n
    errors: tuple[numpy.ndarray[float]]
        Error values containing the following data:
        p_err, c_err, sl_err, f_n_err, a_n_err, ph_n_err
    stats: tuple[numpy.ndarray[float]]
        Statistic parameters: n_param, bic, noise_level
    """
    with h5py.File(file_name, 'r') as file:
        identifier = file.attrs['identifier']
        description = file.attrs['description']
        dataset = file.attrs['dataset']
        date_time = file.attrs['date_time']
        # stats
        n_param = file.attrs['n_param']
        bic = file.attrs['bic']
        noise_level = file.attrs['noise_level']
        # main results and errors
        p_orb = np.copy(file['p_orb'])
        p_err = file['p_orb'].attrs['p_err']
        const = np.copy(file['const'])
        c_err = file['const'].attrs['c_err']
        slope = np.copy(file['slope'])
        sl_err = file['slope'].attrs['sl_err']
        f_n = np.copy(file['frequency'])
        f_n_err = np.copy(file['f_n_err'])
        a_n = np.copy(file['amplitude'])
        a_n_err = np.copy(file['a_n_err'])
        ph_n = np.copy(file['phase'])
        ph_n_err = np.copy(file['ph_n_err'])
        
    results = (p_orb, const, slope, f_n, a_n, ph_n)
    errors = (p_err, c_err, sl_err, f_n_err, a_n_err, ph_n_err)
    stats = (n_param, bic, noise_level)
    
    if verbose:
        print(f'Opened frequency analysis file with identifier: {identifier}, created on {date_time}. \n'
              f'Dataset: {dataset}. Description: {description} \n')
    return results, errors, stats


def save_results_timings(t_zero, timings, depths, t_bottoms, timing_errs, depths_err, ecl_indices, file_name,
                         data_id=None):
    """Save the results of the eclipse timings
    
    Parameters
    ----------
    t_zero: float
        Time of deepest minimum modulo p_orb
    timings: numpy.ndarray[float]
        Eclipse timings of minima and first and last contact points,
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2
    depths: numpy.ndarray[float]
        Eclipse depth of the primary and secondary, depth_1, depth_2
    t_bottoms: numpy.ndarray[float]
        Eclipse timings of the possible flat bottom (internal tangency)
        t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2
    timing_errs: numpy.ndarray[float]
        Error estimates for the eclipse timings,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err, or
        t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err
    depths_err: numpy.ndarray[float]
        Error estimates for the depths
    ecl_indices: numpy.ndarray[int]
        Indices of several important points in the harmonic model
        as generated here (see function for details)
    file_name: str, None
        File name (including path) for saving the results.
    data_id: int, str, None
        Identification for the dataset used
    
    Returns
    -------
    None
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2 = timings
    t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = t_bottoms
    t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err = timing_errs
    d_1_err, d_2_err = depths_err
    var_names = ['t_0', 't_1', 't_2', 't_1_1', 't_1_2', 't_2_1', 't_2_2', 'depth_1', 'depth_2',
                 't_b_1_1', 't_b_1_2', 't_b_2_1', 't_b_2_2', 't_1_err', 't_2_err',
                 't_1_1_err', 't_1_2_err', 't_2_1_err', 't_2_2_err', 'd_1_err', 'd_2_err']
    var_desc = ['time of primary minimum modulo the period',
                'time of primary minimum minus t_0', 'time of secondary minimum minus t_0',
                'time of primary first contact minus t_0', 'time of primary last contact minus t_0',
                'time of secondary first contact minus t_0', 'time of secondary last contact minus t_0',
                'depth of primary minimum', 'depth of secondary minimum',
                'start of (flat) eclipse bottom left of primary minimum',
                'end of (flat) eclipse bottom right of primary minimum',
                'start of (flat) eclipse bottom left of secondary minimum',
                'end of (flat) eclipse bottom right of secondary minimum',
                'error in time of primary minimum (t_1)',
                'error in time of secondary minimum (t_2)',
                'error in time of primary first contact (t_1_1 or tau_1_1)',
                'error in time of primary last contact (t_1_2 or tau_1_2)',
                'error in time of secondary first contact (t_2_1 or tau_2_1)',
                'error in time of secondary last contact (t_2_2 or tau_2_2)',
                'error in depth of primary minimum', 'error in depth of secondary minimum']
    values = [str(t_zero), str(t_1), str(t_2), str(t_1_1), str(t_1_2), str(t_2_1), str(t_2_2),
              str(depths[0]), str(depths[1]), str(t_b_1_1), str(t_b_1_2), str(t_b_2_1), str(t_b_2_2),
              str(t_1_err), str(t_2_err), str(tau_1_1_err), str(tau_1_2_err), str(tau_2_1_err), str(tau_2_2_err),
              str(d_1_err), str(d_2_err)]
    table = np.column_stack((var_names, values, var_desc))
    file_id = os.path.splitext(os.path.basename(file_name))[0]  # the file name without extension
    description = 'Eclipse timings and depths.'
    hdr = f'{file_id}, {data_id}, {description}\nname, value, description'
    np.savetxt(file_name, table, delimiter=',', fmt='%s', header=hdr)
    # save eclipse indices separately
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_2 = file_name.replace(fn_ext, '_ecl_indices' + fn_ext)
    description = 'Eclipse indices (see function measure_eclipses_dt).'
    hdr = (f'{file_id}, {data_id}, {description}\nzeros_1, minimum_1, peaks_2_n, peaks_1, peaks_2_p, minimum_0, '
           f'peaks_2_p, peaks_1, peaks_2_n, minimum_1, zeros_1')
    np.savetxt(file_name_2, ecl_indices, delimiter=',', fmt='%s', header=hdr)
    return


def read_results_timings(file_name):
    """Read in the results of the eclipse timings
    
    Parameters
    ----------
    file_name: str, None
        File name (including path) for loading the results.
    
    Returns
    -------
    t_zero: float
        Time of deepest minimum modulo p_orb
    timings: numpy.ndarray[float]
        Eclipse timings of minima and first and last contact points,
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2
    depths: numpy.ndarray[float]
        Eclipse depth of the primary and secondary, depth_1, depth_2
    t_bottoms: numpy.ndarray[float]
        Eclipse timings of the possible flat bottom (internal tangency)
        t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2
    timing_errs: numpy.ndarray[float]
        Error estimates for the eclipse timings,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err, or
        t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err
    depths_err: numpy.ndarray[float]
        Error estimates for the depths
    ecl_indices: numpy.ndarray[int]
        Indices of several important points in the harmonic model
        as generated here (see function for details)
    """
    values = np.loadtxt(file_name, usecols=(1,), delimiter=',', unpack=True)
    t_zero, t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, depth_1, depth_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = values[:13]
    t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err, d_1_err, d_2_err = values[13:]
    # put these into some arrays
    depths = np.array([depth_1, depth_2])
    t_bottoms = np.array([t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2])
    timings = np.array([t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2])
    timing_errs = np.array([t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err])
    depths_err = np.array([d_1_err, d_2_err])
    # eclipse indices
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_2 = file_name.replace(fn_ext, '_ecl_indices' + fn_ext)
    ecl_indices = np.loadtxt(file_name_2, delimiter=',', dtype=int)
    return t_zero, timings, depths, t_bottoms, timing_errs, depths_err, ecl_indices


def save_results_hsep(const_ho, f_ho, a_ho, ph_ho, f_he, a_he, ph_he, file_name, data_id=None):
    """Save the results of the harmonic separation
    
    Parameters
    ----------
    const_ho: numpy.ndarray[float]
        Mean of the residual
    f_ho: numpy.ndarray[float]
        Frequencies of a number of harmonic sine waves
    a_ho: numpy.ndarray[float]
        Amplitudes of a number of harmonic sine waves
    ph_ho: numpy.ndarray[float]
        Phases of a number of harmonic sine waves
    f_he: numpy.ndarray[float]
        Frequencies of a number of harmonic sine waves
    a_he: numpy.ndarray[float]
        Amplitudes of a number of harmonic sine waves
    ph_he: numpy.ndarray[float]
        Phases of a number of harmonic sine waves
    file_name: str, None
        File name (including path) for saving the results.
    data_id: int, str, None
        Identification for the dataset used
    
    Returns
    -------
    None
    """
    len_ho = len(f_ho)
    len_he = len(f_he)
    var_names = ['const_ho']
    var_names.extend([f'f_{i + 1}o' for i in range(len_ho)])
    var_names.extend([f'a_{i + 1}o' for i in range(len_ho)])
    var_names.extend([f'ph_{i + 1}o' for i in range(len_ho)])
    var_names.extend([f'f_{i + 1}e' for i in range(len_he)])
    var_names.extend([f'a_{i + 1}e' for i in range(len_he)])
    var_names.extend([f'ph_{i + 1}e' for i in range(len_he)])
    var_desc = ['mean of the residual after subtraction of the o.o.e. harmonics']
    var_desc.extend([f'frequency of harmonic {i} of the o.o.e. signal' for i in range(len_ho)])
    var_desc.extend([f'amplitude of harmonic {i} of the o.o.e. signal' for i in range(len_ho)])
    var_desc.extend([f'phase of harmonic {i} of the o.o.e. signal' for i in range(len_ho)])
    var_desc.extend([f'frequency of harmonic {i} of the subtracted harmonics' for i in range(len_he)])
    var_desc.extend([f'amplitude of harmonic {i} of the subtracted harmonics' for i in range(len_he)])
    var_desc.extend([f'phase of harmonic {i} of the subtracted harmonics' for i in range(len_he)])
    values = [str(const_ho)]
    values.extend([str(f_ho[i]) for i in range(len_ho)])
    values.extend([str(a_ho[i]) for i in range(len_ho)])
    values.extend([str(ph_ho[i]) for i in range(len_ho)])
    values.extend([str(f_he[i]) for i in range(len_he)])
    values.extend([str(a_he[i]) for i in range(len_he)])
    values.extend([str(ph_he[i]) for i in range(len_he)])
    table = np.column_stack((var_names, values, var_desc))
    file_id = os.path.splitext(os.path.basename(file_name))[0]  # the file name without extension
    description = 'Eclipse timings and depths.'
    hdr = f'{file_id}, {data_id}, {description}\nname, value, description'
    np.savetxt(file_name, table, delimiter=',', fmt='%s', header=hdr)
    return


def read_results_hsep(file_name):
    """Read in the results of the harmonic separation
    
    Parameters
    ----------
    file_name: str, None
        File name (including path) for loading the results.
    
    Returns
    -------
    const_ho: numpy.ndarray[float]
        Mean of the residual
    f_ho: numpy.ndarray[float]
        Frequencies of a number of harmonic sine waves
    a_ho: numpy.ndarray[float]
        Amplitudes of a number of harmonic sine waves
    ph_ho: numpy.ndarray[float]
        Phases of a number of harmonic sine waves
    f_he: numpy.ndarray[float]
        Frequencies of a number of harmonic sine waves
    a_he: numpy.ndarray[float]
        Amplitudes of a number of harmonic sine waves
    ph_he: numpy.ndarray[float]
        Phases of a number of harmonic sine waves
    """
    var_names = np.loadtxt(file_name, usecols=(0,), delimiter=',', dtype=str, unpack=True)
    values = np.loadtxt(file_name, usecols=(1,), delimiter=',', unpack=True)
    # find out how many sinusoids
    f_ho_names = var_names[np.char.startswith(var_names, 'f_') & np.char.endswith(var_names, 'o')]
    len_ho = len(f_ho_names)
    len_he = (len(var_names) - 1 - 3 * len_ho) // 3
    const_ho = values[0]
    f_ho = values[1:1 + len_ho]
    a_ho = values[1 + len_ho:1 + 2 * len_ho]
    ph_ho = values[1 + 2 * len_ho:1 + 3 * len_ho]
    f_he = values[1 + 3 * len_ho:1 + 3 * len_ho + len_he]
    a_he = values[1 + 3 * len_ho + len_he:1 + 3 * len_ho + 2 * len_he]
    ph_he = values[1 + 3 * len_ho + 2 * len_he:1 + 3 * len_ho + 3 * len_he]
    return const_ho, f_ho, a_ho, ph_ho, f_he, a_he, ph_he


def save_results_elements(e, w, i, phi_0, psi_0, r_sum_sma, r_dif_sma, r_ratio, sb_ratio, errors, intervals, bounds,
                          formal_errors, dists_in, dists_out, file_name, data_id=None):
    """Save the results of the determination of orbital elements
    
    Parameters
    ----------
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit
    phi_0: float
        Auxilary angle (see Kopal 1959)
    psi_0: float
        Auxilary angle like phi_0 but for the eclipse bottoms
    r_sum_sma: float
        Sum of radii in units of the semi-major axis
    r_dif_sma: float
        Absolute difference of radii in units of the semi-major axis
    r_ratio: float
        Radius ratio r_2/r_1
    sb_ratio: float
        Surface brightness ratio sb_2/sb_1
    errors: tuple[numpy.ndarray[float]]
        The (non-symmetric) errors for the same parameters as intervals.
        Derived from the intervals
    intervals: tuple[numpy.ndarray[float]]
        The HDIs (hdi_prob=0.683) for the parameters:
        e, w, i, phi_0, psi_0, r_sum_sma, r_dif_sma, r_ratio,
        sb_ratio, e*cos(w), e*sin(w), f_c, f_s
    bounds: tuple[numpy.ndarray[float]]
        The HDIs (hdi_prob=0.997) for the same parameters as intervals
    formal_errors: tuple[float]
        Formal (symmetric) errors in the parameters:
        e, w, phi_0, r_sum_sma, ecosw, esinw, f_c, f_s
    dists_in: tuple[numpy.ndarray[float]]
        Full input distributions for: t_1, t_2,
        tau_1_1, tau_1_2, tau_2_1, tau_2_2, d_1, d_2, bot_1, bot_2
    dists_out: tuple[numpy.ndarray[float]]
        Full output distributions for the same parameters as intervals
    file_name: str, None
        File name (including path) for saving the results.
    data_id: int, str, None
        Identification for the dataset used
    
    Returns
    -------
    None
    """
    e_err, w_err, i_err, phi_0_err, psi_0_err, r_sum_sma_err, r_dif_sma_err, r_ratio_err, sb_ratio_err = errors[:9]
    ecosw_err, esinw_err, f_c_err, f_s_err = errors[9:]
    e_bds, w_bds, i_bds, phi_0_bds, psi_0_bds, r_sum_sma_bds, r_dif_sma_bds, r_ratio_bds, sb_ratio_bds = bounds[:9]
    ecosw_bds, esinw_bds, f_c_bds, f_s_bds = bounds[9:]
    sigma_e, sigma_w, sigma_phi_0, sigma_r_sum_sma, sigma_ecosw, sigma_esinw, sigma_f_c, sigma_f_s = formal_errors
    # multi
    if (len(np.shape(w_bds)) > 1):
        w_interval = intervals[:, [1, 2]]  # (this is probably not right)
        w_bds_2 = w_bds[np.sign((w - w_interval)[:, 0] * (w - w_interval)[:, 1]) == 1]
        w_bds = w_bds[np.sign((w - w_interval)[:, 0] * (w - w_interval)[:, 1]) == -1]
    else:
        w_bds_2 = None
    var_names = ['e', 'w', 'i', 'phi_0', 'psi_0', 'r_sum_sma', 'r_dif_sma', 'r_ratio', 'sb_ratio',
                 'e_upper', 'e_lower', 'w_upper', 'w_lower', 'i_upper', 'i_lower', 'phi_0_upper', 'phi_0_lower',
                 'r_sum_sma_upper', 'r_sum_sma_lower', 'r_ratio_upper', 'r_ratio_lower',
                 'sb_ratio_upper', 'sb_ratio_lower', 'ecosw_upper', 'ecosw_lower',
                 'esinw_upper', 'esinw_lower', 'f_c_upper', 'f_c_lower', 'f_s_upper', 'f_s_lower',
                 'e_ubnd', 'e_lbnd', 'w_ubnd', 'w_lbnd', 'i_ubnd', 'i_lbnd', 'phi_0_ubnd', 'phi_0_lbnd',
                 'psi_0_ubnd', 'psi_0_lbnd', 'r_sum_sma_ubnd', 'r_sum_sma_lbnd', 'r_dif_sma_ubnd', 'r_dif_sma_lbnd',
                 'r_ratio_ubnd', 'r_ratio_lbnd', 'sb_ratio_ubnd', 'sb_ratio_lbnd', 'ecosw_ubnd', 'ecosw_lbnd',
                 'esinw_ubnd', 'esinw_lbnd', 'f_c_ubnd', 'f_c_lbnd', 'f_s_ubnd', 'f_s_lbnd',
                 'sigma_e', 'sigma_w', 'sigma_phi_0', 'sigma_r_sum_sma', 'sigma_ecosw', 'sigma_esinw',
                 'sigma_f_c', 'sigma_f_s']
    var_desc = ['eccentricity', 'argument of periastron (radians)', 'inclination (radians)',
                'auxiliary angle phi_0 (see Kopal 1959) (radians)', 'auxiliary angle psi_0 (radians)',
                'sum of radii divided by the semi-major axis of the relative orbit',
                'limit on difference of radii divided by the semi-major axis of the relative orbit',
                'radius ratio r2/r1', 'surface brightness ratio sb2/sb1',
                'upper error estimate in e', 'lower error estimate in e',
                'upper error estimate in w', 'lower error estimate in w',
                'upper error estimate in i', 'lower error estimate in i',
                'upper error estimate in phi_0', 'lower error estimate in phi_0',
                'upper error estimate in r_sum_sma', 'lower error estimate in r_sum_sma',
                'upper error estimate in r_ratio', 'lower error estimate in r_ratio',
                'upper error estimate in sb_ratio', 'lower error estimate in sb_ratio',
                'upper error estimate in ecos(w)', 'lower error estimate in ecos(w)',
                'upper error estimate in esin(w)', 'lower error estimate in esin(w)',
                'upper error estimate in f_c', 'lower error estimate in f_c',
                'upper error estimate in f_s', 'lower error estimate in f_s',
                'upper bound in e (hdi_prob=997)', 'lower bound in e (hdi_prob=997)',
                'upper bound in w (hdi_prob=997)', 'lower bound in w (hdi_prob=997)',
                'upper bound in i (hdi_prob=997)', 'lower bound in i (hdi_prob=997)',
                'upper bound in phi_0 (hdi_prob=997)', 'lower bound in phi_0 (hdi_prob=997)',
                'upper bound in psi_0 (hdi_prob=997)', 'lower bound in psi_0 (hdi_prob=997)',
                'upper bound in r_sum_sma (hdi_prob=997)', 'lower bound in r_sum_sma (hdi_prob=997)',
                'upper bound in r_dif_sma (hdi_prob=997)', 'lower bound in r_dif_sma (hdi_prob=997)',
                'upper bound in r_ratio (hdi_prob=997)', 'lower bound in r_ratio (hdi_prob=997)',
                'upper bound in sb_ratio (hdi_prob=997)', 'lower bound in sb_ratio (hdi_prob=997)',
                'upper bound in ecos(w) (hdi_prob=997)', 'lower bound in ecos(w) (hdi_prob=997)',
                'upper bound in esin(w) (hdi_prob=997)', 'lower bound in esin(w) (hdi_prob=997)',
                'upper bound in f_c (hdi_prob=997)', 'lower bound in f_c (hdi_prob=997)',
                'upper bound in f_s (hdi_prob=997)', 'lower bound in f_s (hdi_prob=997)',
                'formal uncorrelated error in e', 'formal uncorrelated error in w',
                'formal uncorrelated error in phi_0', 'formal uncorrelated error in r_sum_sma',
                'formal uncorrelated error in ecos(w)', 'formal uncorrelated error in esin(w)',
                'formal uncorrelated error in f_c', 'formal uncorrelated error in f_s']
    values = [str(e), str(w), str(i), str(phi_0), str(psi_0), str(r_sum_sma), str(r_dif_sma), str(r_ratio),
              str(sb_ratio), str(e_err[1]), str(e_err[0]), str(w_err[1]), str(w_err[0]), str(i_err[1]), str(i_err[0]),
              str(phi_0_err[1]), str(phi_0_err[0]), str(r_sum_sma_err[1]), str(r_sum_sma_err[0]),
              str(r_ratio_err[1]), str(r_ratio_err[0]), str(sb_ratio_err[1]), str(sb_ratio_err[0]),
              str(ecosw_err[1]), str(ecosw_err[0]), str(esinw_err[1]), str(esinw_err[0]),
              str(f_c_err[1]), str(f_c_err[0]), str(f_s_err[1]), str(f_s_err[0]),
              str(e_bds[1]), str(e_bds[0]), str(w_bds[1]), str(w_bds[0]), str(i_bds[1]), str(i_bds[0]),
              str(phi_0_bds[1]), str(phi_0_bds[0]), str(psi_0_bds[1]), str(psi_0_bds[0]),
              str(r_sum_sma_bds[1]), str(r_sum_sma_bds[0]), str(r_dif_sma_bds[1]), str(r_dif_sma_bds[0]),
              str(r_ratio_bds[1]), str(r_ratio_bds[0]), str(sb_ratio_bds[1]), str(sb_ratio_bds[0]),
              str(ecosw_bds[1]), str(ecosw_bds[0]), str(esinw_bds[1]), str(esinw_bds[0]),
              str(f_c_bds[1]), str(f_c_bds[0]), str(f_s_bds[1]), str(f_s_bds[0]),
              str(sigma_e), str(sigma_w), str(sigma_phi_0), str(sigma_r_sum_sma), str(sigma_ecosw), str(sigma_esinw),
              str(sigma_f_c), str(sigma_f_s)]
    table = np.column_stack((var_names, values, var_desc))
    if (len(np.shape(intervals[1])) > 1):
        # omega is somewhere around 90 or 270 deg, giving rise to a disjunct confidence interval
        var_names_ext = ['w_interval_1_low', 'w_interval_1_high', 'w_interval_2_low', 'w_interval_2_high']
        var_desc_ext = ['lower bound of the first w interval', 'upper bound of the first w interval',
                        'lower bound of the second w interval', 'upper bound of the second w interval']
        values_ext = [str(intervals[1][0, 0]), str(intervals[1][0, 1]),
                      str(intervals[1][1, 0]), str(intervals[1][1, 1])]
        table = np.vstack((table, np.column_stack((var_names_ext, var_desc_ext, values_ext))))
    if w_bds_2 is not None:
        # omega is somewhere around 90 or 270 deg, giving rise to a disjunct confidence interval
        var_names_ext = ['w_ubnd_2', 'w_lbnd_2']
        var_desc_ext = ['second upper bound in w (hdi_prob=997)', 'second lower bound in w (hdi_prob=997)']
        values_ext = [str(w_bds_2[1]), str(w_bds_2[0])]
        table = np.vstack((table, np.column_stack((var_names_ext, var_desc_ext, values_ext))))
    file_id = os.path.splitext(os.path.basename(file_name))[0]  # the file name without extension
    description = 'Determination of orbital elements.'
    hdr = f'{file_id}, {data_id}, {description}\nname, value, description'
    np.savetxt(file_name, table, delimiter=',', fmt='%s', header=hdr)
    # save the distributions separately
    data = np.column_stack((*dists_in, *dists_out))
    # file name just adds 'dists' at the end
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_2 = file_name.replace(fn_ext, '_dists' + fn_ext)
    description = 'Prior and posterior distributions (not MCMC).'
    hdr = (f'{file_id}, {data_id}, {description}\n'
           + 't_1_vals, t_2_vals, tau_1_1_vals, tau_1_2_vals, tau_2_1_vals, tau_2_2_vals, '
           + 'd_1_vals, d_2_vals, bot_1_vals, bot_2_vals, '
           + 'e_vals, w_vals, i_vals, phi0_vals, psi0_vals, rsumsma_vals, rdifsma_vals, rratio_vals, sbratio_vals')
    np.savetxt(file_name_2, data, delimiter=',', header=hdr)
    return


def read_results_elements(file_name):
    """Read in the results of the determination of orbital elements
    
    Parameters
    ----------
    file_name: str, None
        File name (including path) for loading the results.
    
    Returns
    -------
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit
    phi_0: float
        Auxilary angle (see Kopal 1959)
    psi_0: float
        Auxilary angle like phi_0 but for the eclipse bottoms
    r_sum_sma: float
        Sum of radii in units of the semi-major axis
    r_dif_sma: float
        Absolute difference of radii in units of the semi-major axis
    r_ratio: float
        Radius ratio r_2/r_1
    sb_ratio: float
        Surface brightness ratio sb_2/sb_1
    errors: tuple[numpy.ndarray[float]]
        The (non-symmetric) errors for the same parameters as intervals.
        Derived from the intervals
    intervals: tuple[numpy.ndarray[float]]
        The HDIs (hdi_prob=0.683) for the parameters:
        e, w, i, phi_0, psi_0, r_sum_sma, r_dif_sma, r_ratio,
        sb_ratio, e*cos(w), e*sin(w), f_c, f_s
    bounds: tuple[numpy.ndarray[float]]
        The HDIs (hdi_prob=0.997) for the same parameters as intervals
    formal_errors: tuple[float]
        Formal (symmetric) errors in the parameters:
        e, w, phi_0, r_sum_sma, ecosw, esinw, f_c, f_s
    dists_in: tuple[numpy.ndarray[float]]
        Full input distributions for: t_1, t_2,
        tau_1_1, tau_1_2, tau_2_1, tau_2_2, d_1, d_2, bot_1, bot_2
    dists_out: tuple[numpy.ndarray[float]]
        Full output distributions for the same parameters as intervals
    """
    results = np.loadtxt(file_name, usecols=(1,), delimiter=',', unpack=True)
    e, w, i, phi_0, psi_0, r_sum_sma, r_dif_sma, r_ratio, sb_ratio = results[:9]
    errors = results[9:31].reshape((11, 2))
    bounds = results[31:53].reshape((11, 2))
    formal_errors = results[53:61]
    # intervals_w  # ? for when the interval is disjoint
    # distributions
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    all_dists = np.loadtxt(file_name.replace(fn_ext, '_dists' + fn_ext), delimiter=',', unpack=True)
    dists_in = all_dists[:10]
    dists_out = all_dists[10:]
    return (e, w, i, phi_0, psi_0, r_sum_sma, r_dif_sma, r_ratio, sb_ratio, errors, bounds, formal_errors,
            dists_in, dists_out)


def save_results_fit_ellc(par_init, par_fit, file_name, data_id=None):
    """Save the results of the fit with ellc models
    
    Parameters
    ----------
    par_init: tuple[float], list[float], numpy.ndarray[float]
        Initial eclipse parameters to start the fit, consisting of:
        e, w, i, r_sum_sma, r_ratio, sb_ratio
    par_fit: tuple[float]
        Optimised eclipse parameters , consisting of:
        e, w, i, r_sum_sma, r_ratio, sb_ratio, offset
        The offset is a constant added to the model to match
        the light level of the data
    file_name: str, None
        File name (including path) for saving the results.
    data_id: int, str, None
        Identification for the dataset used
    
    Returns
    -------
    None
    """
    var_names = ['e_0', 'w_0', 'i_0', 'r_sum_0', 'r_rat_0', 'sb_rat_0',
                 'e_1', 'w_1', 'i_1', 'r_sum_1', 'r_rat_1', 'sb_rat_1', 'offset']
    var_desc = ['initial eccentricity', 'initial argument of periastron', 'initial orbital inclination i (radians)',
                'initial sum of fractional radii (r1+r2)/a', 'initial radius ratio r2/r1',
                'initial surface brightness ratio sb2/sb1 or (Teff2/Teff1)^4',
                'eccentricity after fit', 'argument of periastron after fit', 'i after fit (radians)',
                '(r1+r2)/a after fit', 'r2/r1 after fit', 'sb2/sb1 after fit', 'ellc lc offset']
    values = [str(par_init[0]), str(par_init[1]), str(par_init[2]), str(par_init[3]), str(par_init[4]),
              str(par_init[5]), str(par_fit[0]), str(par_fit[1]), str(par_fit[2]), str(par_fit[3]), str(par_fit[4]),
              str(par_fit[5]), str(par_fit[6])]
    table = np.column_stack((var_names, values, var_desc))
    file_id = os.path.splitext(os.path.basename(file_name))[0]  # the file name without extension
    description = f'Fit for the light curve parameters. Fit uses the eclipses only.'
    hdr = f'{file_id}, {data_id}, {description}\nname, value, description'
    np.savetxt(file_name, table, delimiter=',', fmt='%s', header=hdr)
    return


def read_results_fit_ellc(file_name):
    """Read in the results of the fit with ellc models
    
    Parameters
    ----------
    file_name: str, None
        File name (including path) for loading the results.
    
    Returns
    -------
    par_init: tuple[float], list[float], numpy.ndarray[float]
        Initial eclipse parameters to start the fit, consisting of:
        e, w, i, r_sum_sma, r_ratio, sb_ratio
    par_fit: tuple[float]
        Optimised eclipse parameters , consisting of:
        e, w, i, r_sum_sma, r_ratio, sb_ratio, offset
        The offset is a constant added to the model to match
        the light level of the data
        
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit
    r_sum_sma: float
        Sum of radii in units of the semi-major axis
    r_ratio: float
        Radius ratio r_2/r_1
    sb_ratio: float
        Surface brightness ratio sb_2/sb_1
    opt_e: float
        Optimised eccentricity of the orbit
    opt_w: float
        Optimised argument of periastron
    opt_i: float
        Optimised inclination of the orbit
    opt_r_sum_sma: float
        Optimised sum of radii in units of the semi-major axis
    opt_r_ratio: float
        Optimised radius ratio r_2/r_1
    opt_sb_ratio: float
        Optimised surface brightness ratio sb_2/sb_1
    """
    results = np.loadtxt(file_name, usecols=(1,), delimiter=',', unpack=True)
    e, w, i, r_sum_sma, r_ratio, sb_ratio = results[:6]
    opt_e, opt_w, opt_i, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio, offset = results[6:]
    return (e, w, i, r_sum_sma, r_ratio, sb_ratio, opt_e, opt_w, opt_i, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio,
            offset)


def save_results_fselect(f_n, a_n, ph_n, passed_nh_sigma, passed_nh_snr, file_name, data_id=None):
    """Save the results of the frequency selection
    
    Parameters
    ----------
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    passed_nh_sigma: numpy.ndarray[bool]
        Non-harmonic frequencies that passed the sigma check
    passed_nh_snr: numpy.ndarray[bool]
        Non-harmonic frequencies that passed the signal-to-noise check
    file_name: str, None
        File name (including path) for saving the results.
    data_id: int, str, None
        Identification for the dataset used
    
    Returns
    -------
    None
    """
    # passing both
    passed_nh_b = (passed_nh_sigma & passed_nh_snr)
    # stick together
    table = np.column_stack((np.arange(1, len(f_n)+1), f_n, a_n, ph_n, passed_nh_sigma, passed_nh_snr, passed_nh_b))
    file_id = os.path.splitext(os.path.basename(file_name))[0]  # the file name without extension
    description = f'Selection of credible non-harmonic frequencies'
    hdr = f'{file_id}, {data_id}, {description}\nn, f_n, a_n, ph_n, pass_sigma_check, pass_snr_check, pass_all'
    np.savetxt(file_name, table, delimiter=',', header=hdr)
    return


def read_results_fselect(file_name):
    """Read in the results of the frequency selection
    
    Parameters
    ----------
    file_name: str, None
        File name (including path) for loading the results.
    
    Returns
    -------
    passed_nh_sigma: numpy.ndarray[bool]
        Non-harmonic frequencies that passed the sigma check
    passed_nh_snr: numpy.ndarray[bool]
        Non-harmonic frequencies that passed the signal-to-noise check
    passed_nh_b: numpy.ndarray[bool]
        Non-harmonic frequencies that passed both checks
    """
    results = np.loadtxt(file_name, usecols=(4, 5, 6), delimiter=',', unpack=True)
    passed_nh_sigma, passed_nh_snr, passed_nh_b = results
    passed_nh_sigma = passed_nh_sigma.astype(int).astype(bool)  # stored as floats
    passed_nh_snr = passed_nh_snr.astype(int).astype(bool)  # stored as floats
    passed_nh_b = passed_nh_b.astype(int).astype(bool)  # stored as floats
    return passed_nh_sigma, passed_nh_snr, passed_nh_b


def save_results_disentangle(const_r, f_n_r, a_n_r, ph_n_r, file_name, data_id=None):
    """Save the results of disentangling the harmonics from the eclipse model
    
    Parameters
    ----------
    const_r: numpy.ndarray[float]
        Mean of the residual
    f_n_r: numpy.ndarray[float]
        Frequencies of a number of harmonic sine waves
    a_n_r: numpy.ndarray[float]
        Amplitudes of a number of harmonic sine waves
    ph_n_r: numpy.ndarray[float]
        Phases of a number of harmonic sine waves
    file_name: str, None
        File name (including path) for saving the results.
    data_id: int, str, None
        Identification for the dataset used
    
    Returns
    -------
    None
    """
    table = np.column_stack((np.arange(len(f_n_r)+1), np.append([0], f_n_r), np.append([const_r], a_n_r),
                             np.append([0], ph_n_r)))
    file_id = os.path.splitext(os.path.basename(file_name))[0]  # the file name without extension
    description = f'Disentangelment of harmonics using ellc lc model'
    hdr = f'{file_id}, {data_id}, {description}\nn, f_n_r, a_n_r, ph_n_r'
    np.savetxt(file_name, table, delimiter=',', header=hdr)
    return


def read_results_disentangle(file_name):
    """Read in the results of disentangling the harmonics from the eclipse model
    
    Parameters
    ----------
    file_name: str, None
        File name (including path) for loading the results.
    
    Returns
    -------
    const_r: numpy.ndarray[float]
        Mean of the residual
    f_n_r: numpy.ndarray[float]
        Frequencies of a number of harmonic sine waves
    a_n_r: numpy.ndarray[float]
        Amplitudes of a number of harmonic sine waves
    ph_n_r: numpy.ndarray[float]
        Phases of a number of harmonic sine waves
    """
    results = np.loadtxt(file_name, usecols=(1, 2, 3), delimiter=',', unpack=True)
    const_r = results[1, 0]
    f_n_r, a_n_r, ph_n_r = results[:, 1:]
    return const_r, f_n_r, a_n_r, ph_n_r


def sequential_plotting(tic, times, signal, i_sectors, load_dir, save_dir=None, show=False):
    """Due to plotting not working under multiprocessing this function is
    made to make plots after running the analysis in parallel.
    
    Parameters
    ----------
    tic: int
        The TESS Input Catalog number for later reference
        Use any number (or even str) as reference if not available.
    times: numpy.ndarray[float]
        Timestamps of the time-series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    load_dir: str
        Path to a directory for loading analysis results.
    save_dir: str, None
        Path to a directory for save the plots.
    
    Returns
    -------
    None
    """
    # open all the data
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_1.hdf5')
    if os.path.isfile(file_name):
        results, errors_12, stats = read_results(file_name, verbose=False)
        p_orb_1, const_1, slope_1, f_n_1, a_n_1, ph_n_1 = results
        p_orb_1 = p_orb_1[0]  # must be a float
        p_err_1, c_err_1, sl_err_1, f_n_err_1, a_n_err_1, ph_n_err_1 = errors_12
        n_param_1, bic_1, noise_level_1 = stats
        model_1 = tsf.linear_curve(times, const_1, slope_1, i_sectors)
        model_1 += tsf.sum_sines(times, f_n_1, a_n_1, ph_n_1)
    else:
        p_orb_1, const_1, slope_1, f_n_1, a_n_1, ph_n_1 = np.array([[], [], [], [], [], []])
        p_orb_1 = 0
        p_err_1, c_err_1, sl_err_1, f_n_err_1, a_n_err_1, ph_n_err_1 = np.array([[], [], [], [], [], []])
        n_param_1, bic_1, noise_level_1 = 0, 0, 0
        model_1 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_2.hdf5')
    if os.path.isfile(file_name):
        results, errors_12, stats = read_results(file_name, verbose=False)
        p_orb_2, const_2, slope_2, f_n_2, a_n_2, ph_n_2 = results
        p_orb_2 = p_orb_2[0]  # must be a float
        p_err_2, c_err_2, sl_err_2, f_n_err_2, a_n_err_2, ph_n_err_2 = errors_12
        n_param_2, bic_2, noise_level_2 = stats
        model_2 = tsf.linear_curve(times, const_2, slope_2, i_sectors)
        model_2 += tsf.sum_sines(times, f_n_2, a_n_2, ph_n_2)
    else:
        p_orb_2, const_2, slope_2, f_n_2, a_n_2, ph_n_2 = np.array([[], [], [], [], [], []])
        p_orb_2 = 0
        p_err_2, c_err_2, sl_err_2, f_n_err_2, a_n_err_2, ph_n_err_2 = np.array([[], [], [], [], [], []])
        n_param_2, bic_2, noise_level_2 = 0, 0, 0
        model_2 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_3.hdf5')
    if os.path.isfile(file_name):
        results, errors_12, stats = read_results(file_name, verbose=False)
        p_orb_3, const_3, slope_3, f_n_3, a_n_3, ph_n_3 = results
        p_orb_3 = p_orb_3[0]  # must be a float
        p_err_3, c_err_3, sl_err_3, f_n_err_3, a_n_err_3, ph_n_err_3 = errors_12
        n_param_3, bic_3, noise_level_3 = stats
        model_3 = tsf.linear_curve(times, const_3, slope_3, i_sectors)
        model_3 += tsf.sum_sines(times, f_n_3, a_n_3, ph_n_3)
    else:
        p_orb_3, const_3, slope_3, f_n_3, a_n_3, ph_n_3 = np.array([[], [], [], [], [], []])
        p_orb_3 = 0
        p_err_3, c_err_3, sl_err_3, f_n_err_3, a_n_err_3, ph_n_err_3 = np.array([[], [], [], [], [], []])
        n_param_3, bic_3, noise_level_3 = 0, 0, 0
        model_3 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_4.hdf5')
    if os.path.isfile(file_name):
        results, errors_12, stats = read_results(file_name, verbose=False)
        p_orb_4, const_4, slope_4, f_n_4, a_n_4, ph_n_4 = results
        p_orb_4 = p_orb_4[0]  # must be a float
        p_err_4, c_err_4, sl_err_4, f_n_err_4, a_n_err_4, ph_n_err_4 = errors_12
        n_param_4, bic_4, noise_level_4 = stats
        model_4 = tsf.linear_curve(times, const_4, slope_4, i_sectors)
        model_4 += tsf.sum_sines(times, f_n_4, a_n_4, ph_n_4)
    else:
        p_orb_4, const_4, slope_4, f_n_4, a_n_4, ph_n_4 = np.array([[], [], [], [], [], []])
        p_orb_4 = 0
        p_err_4, c_err_4, sl_err_4, f_n_err_4, a_n_err_4, ph_n_err_4 = np.array([[], [], [], [], [], []])
        n_param_4, bic_4, noise_level_4 = 0, 0, 0
        model_4 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_5.hdf5')
    if os.path.isfile(file_name):
        results, errors_12, stats = read_results(file_name, verbose=False)
        p_orb_5, const_5, slope_5, f_n_5, a_n_5, ph_n_5 = results
        p_orb_5 = p_orb_5[0]  # must be a float
        p_err_5, c_err_5, sl_err_5, f_n_err_5, a_n_err_5, ph_n_err_5 = errors_12
        n_param_5, bic_5, noise_level_5 = stats
        model_5 = tsf.linear_curve(times, const_5, slope_5, i_sectors)
        model_5 += tsf.sum_sines(times, f_n_5, a_n_5, ph_n_5)
    else:
        p_orb_5, const_5, slope_5, f_n_5, a_n_5, ph_n_5 = np.array([[], [], [], [], [], []])
        p_orb_5 = 0
        p_err_5, c_err_5, sl_err_5, f_n_err_5, a_n_err_5, ph_n_err_5 = np.array([[], [], [], [], [], []])
        n_param_5, bic_5, noise_level_5 = 0, 0, 0
        model_5 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_6.hdf5')
    if os.path.isfile(file_name):
        results, errors_12, stats = read_results(file_name, verbose=False)
        p_orb_6, const_6, slope_6, f_n_6, a_n_6, ph_n_6 = results
        p_orb_6 = p_orb_6[0]  # must be a float
        p_err_6, c_err_6, sl_err_6, f_n_err_6, a_n_err_6, ph_n_err_6 = errors_12
        n_param_6, bic_6, noise_level_6 = stats
        model_6 = tsf.linear_curve(times, const_6, slope_6, i_sectors)
        model_6 += tsf.sum_sines(times, f_n_6, a_n_6, ph_n_6)
    else:
        p_orb_6, const_6, slope_6, f_n_6, a_n_6, ph_n_6 = np.array([[], [], [], [], [], []])
        p_orb_6 = 0
        p_err_6, c_err_6, sl_err_6, f_n_err_6, a_n_err_6, ph_n_err_6 = np.array([[], [], [], [], [], []])
        n_param_6, bic_6, noise_level_6 = 0, 0, 0
        model_6 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_7.hdf5')
    if os.path.isfile(file_name):
        results, errors_12, stats = read_results(file_name, verbose=False)
        p_orb_7, const_7, slope_7, f_n_7, a_n_7, ph_n_7 = results
        p_orb_7 = p_orb_7[0]  # must be a float
        p_err_7, c_err_7, sl_err_7, f_n_err_7, a_n_err_7, ph_n_err_7 = errors_12
        n_param_7, bic_7, noise_level_7 = stats
        model_7 = tsf.linear_curve(times, const_7, slope_7, i_sectors)
        model_7 += tsf.sum_sines(times, f_n_7, a_n_7, ph_n_7)
    else:
        p_orb_7, const_7, slope_7, f_n_7, a_n_7, ph_n_7 = np.array([[], [], [], [], [], []])
        p_orb_7 = 0
        p_err_7, c_err_7, sl_err_7, f_n_err_7, a_n_err_7, ph_n_err_7 = np.array([[], [], [], [], [], []])
        n_param_7, bic_7, noise_level_7 = 0, 0, 0
        model_7 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_8.hdf5')
    if os.path.isfile(file_name):
        results, errors_12, stats = read_results(file_name, verbose=False)
        p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8 = results
        p_orb_8 = p_orb_8[0]  # must be a float
        p_err_8, c_err_8, sl_err_8, f_n_err_8, a_n_err_8, ph_n_err_8 = errors_12
        n_param_8, bic_8, noise_level_8 = stats
        model_8 = tsf.linear_curve(times, const_8, slope_8, i_sectors)
        model_8 += tsf.sum_sines(times, f_n_8, a_n_8, ph_n_8)
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_8, p_orb_8)
        f_h_8, a_h_8, ph_h_8 = f_n_8[harmonics], a_n_8[harmonics], ph_n_8[harmonics]
    else:
        p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8 = np.array([[], [], [], [], [], []])
        p_orb_8 = 0
        p_err_8, c_err_8, sl_err_8, f_n_err_8, a_n_err_8, ph_n_err_8 = np.array([[], [], [], [], [], []])
        n_param_8, bic_8, noise_level_8 = 0, 0, 0
        model_8 = np.zeros(len(times))
    # stick together for sending to plot function
    models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8]
    p_orb_i = [0, 0, p_orb_3, p_orb_3, p_orb_5, p_orb_5, p_orb_7, p_orb_8]
    f_n_i = [f_n_1, f_n_2, f_n_3, f_n_4, f_n_5, f_n_6, f_n_7, f_n_8]
    a_n_i = [a_n_1, a_n_2, a_n_3, a_n_4, a_n_5, a_n_6, a_n_7, a_n_8]
    # open some more data - timings, harmonic separation, eclipse parameters
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_9.csv')
    if os.path.isfile(file_name):
        results_9 = read_results_timings(file_name)
        t_zero_9, timings_9, depths_9, t_bottoms_9, timing_errs_9, depths_err_9 = results_9[:6]
        ecl_indices_9 = results_9[6]
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_10.csv')
    if os.path.isfile(file_name):
        results_10 = read_results_hsep(file_name)
        const_ho_10, f_ho_10, a_ho_10, ph_ho_10, f_he_10, a_he_10, ph_he_10 = results_10
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_11.csv')
    if os.path.isfile(file_name):
        results_11 = read_results_timings(file_name)
        t_zero_11, timings_11, depths_11, t_bottoms_11, timing_errs_11, depths_err_11 = results_11[:6]
        ecl_indices_11 = results_11[6]
        timings_tau_11 = np.array([timings_11[0], timings_11[1],
                                   timings_11[0] - timings_11[2], timings_11[3] - timings_11[0],
                                   timings_11[1] - timings_11[4], timings_11[5] - timings_11[1]])
        bottom_dur = np.array([t_bottoms_11[1] - t_bottoms_11[0], t_bottoms_11[3] - t_bottoms_11[2]])
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_12.csv')
    if os.path.isfile(file_name):
        results_12 = read_results_elements(file_name)
        e_12, w_12, i_12, phi_0_12, psi_0_12, r_sum_sma_12, r_dif_sma_12, r_ratio_12, sb_ratio_12 = results_12[:9]
        errors_12, bounds_12, formal_errors_12, dists_in_12, dists_out_12 = results_12[9:]
        # intervals_w #? for when the interval is disjoint
        ecl_pars = (e_12, w_12, i_12, phi_0_12, r_sum_sma_12, r_ratio_12, sb_ratio_12)
    # open some more data - ellc fits and pulsation analysis
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_13.csv')
    if os.path.isfile(file_name):
        results_13 = read_results_fit_ellc(file_name)
        par_init_12, par_opt_13 = results_13[:6], results_13[6:]
        par_ellc = np.copy(par_opt_13)
        par_ellc[0] = np.sqrt(par_opt_13[0]) * np.cos(par_opt_13[1])
        par_ellc[1] = np.sqrt(par_opt_13[0]) * np.sin(par_opt_13[1])
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_14.csv')
    if os.path.isfile(file_name):
        results_14 = read_results_fselect(file_name)
        pass_sigma, pass_snr, passed_nh_b = results_14
    file_name = os.path.join(load_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_15.csv')
    if os.path.isfile(file_name):
        results_15 = read_results_disentangle(file_name)
        const_r, f_n_r, a_n_r, ph_n_r = results_15
    # frequency_analysis
    if save_dir is not None:
        try:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_frequency_analysis_full_pd.png')
            vis.plot_pd_full_output(times, signal, models, p_orb_i, f_n_i, a_n_i, i_sectors, save_file=file_name,
                                    show=False)
            plot_nr = np.arange(len(p_orb_i))[np.nonzero(p_orb_i)][-1] + 1
            plot_data = [eval(f'p_orb_{plot_nr}'), eval(f'const_{plot_nr}'), eval(f'slope_{plot_nr}'),
                         eval(f'f_n_{plot_nr}'), eval(f'a_n_{plot_nr}'), eval(f'ph_n_{plot_nr}')]
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis',
                                     f'tic_{tic}_frequency_analysis_models_{plot_nr}.png')
            vis.plot_harmonic_output(times, signal, *plot_data, i_sectors, save_file=file_name, show=False)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
    if show:
        try:
            vis.plot_pd_full_output(times, signal, models, p_orb_i, f_n_i, a_n_i, i_sectors, save_file=None, show=True)
            plot_nr = np.arange(len(p_orb_i))[np.nonzero(p_orb_i)][-1] + 1
            plot_data = [eval(f'p_orb_{plot_nr}'), eval(f'const_{plot_nr}'), eval(f'slope_{plot_nr}'),
                         eval(f'f_n_{plot_nr}'), eval(f'a_n_{plot_nr}'), eval(f'ph_n_{plot_nr}')]
            vis.plot_harmonic_output(times, signal, *plot_data, i_sectors, save_file=None, show=True)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
    # eclipse_analysis
    if save_dir is not None:
        try:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_eclipse_analysis_derivatives.png')
            vis.plot_lc_derivatives(p_orb_8, f_h_8, a_h_8, ph_h_8, ecl_indices_11, save_file=file_name, show=False)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_eclipse_analysis_hsep.png')
            vis.plot_lc_harmonic_separation(times, signal, p_orb_8, t_zero_11, timings_11, const_8, slope_8, f_n_8,
                                            a_n_8, ph_n_8, const_ho_10, f_ho_10, a_ho_10, ph_ho_10,
                                            f_he_10, a_he_10, ph_he_10, i_sectors, save_file=file_name, show=False)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_eclipse_analysis_timestamps.png')
            vis.plot_lc_eclipse_timestamps(times, signal, p_orb_8, t_zero_11, timings_11, depths_11, t_bottoms_11,
                                           timing_errs_11, depths_err_11, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                                           i_sectors, save_file=file_name, show=False)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_eclipse_analysis_simple_lc.png')
            vis.plot_lc_eclipse_parameters_simple(times, signal, p_orb_8, t_zero_11, timings_11, const_8, slope_8,
                                                  f_n_8, a_n_8, ph_n_8, ecl_pars, i_sectors, save_file=file_name,
                                                  show=False)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_eclipse_analysis_corner.png')
            vis.plot_corner_eclipse_parameters(timings_tau_11, depths_11, bottom_dur, *dists_in_12, e_12, w_12, i_12,
                                               phi_0_12, psi_0_12, r_sum_sma_12, r_dif_sma_12, r_ratio_12, sb_ratio_12,
                                               *dists_out_12, save_file=file_name, show=False)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_eclipse_analysis_ellc_fit.png')
            vis.plot_lc_ellc_fit(times, signal, p_orb_8, t_zero_11, timings_11, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                                 par_init_12, par_opt_13, i_sectors, save_file=file_name, show=False)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_eclipse_analysis_ellc_corner.png')
            vis.plot_corner_ellc_pars(par_init_12, par_opt_13, dists_out_12, save_file=file_name, show=False)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
    if show:
        try:
            vis.plot_lc_derivatives(p_orb_8, f_h_8, a_h_8, ph_h_8, ecl_indices_11, save_file=None, show=True)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            vis.plot_lc_harmonic_separation(times, signal, p_orb_8, t_zero_11, timings_11, const_8, slope_8, f_n_8,
                                            a_n_8, ph_n_8, const_ho_10, f_ho_10, a_ho_10, ph_ho_10,
                                            f_he_10, a_he_10, ph_he_10, i_sectors, save_file=None, show=True)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            vis.plot_lc_eclipse_timestamps(times, signal, p_orb_8, t_zero_11, timings_11, depths_11, t_bottoms_11,
                                           timing_errs_11, depths_err_11, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                                           i_sectors, save_file=None, show=True)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            vis.plot_lc_eclipse_parameters_simple(times, signal, p_orb_8, t_zero_11, timings_11, const_8, slope_8,
                                                  f_n_8, a_n_8, ph_n_8, ecl_pars, i_sectors, save_file=None, show=True)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            vis.plot_dists_eclipse_parameters(e_12, w_12, i_12, phi_0_12, psi_0_12, r_sum_sma_12, r_dif_sma_12,
                                              r_ratio_12, sb_ratio_12, *dists_out_12)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            vis.plot_corner_eclipse_parameters(timings_tau_11, depths_11, bottom_dur, *dists_in_12, e_12, w_12, i_12,
                                               phi_0_12, psi_0_12, r_sum_sma_12, r_dif_sma_12, r_ratio_12, sb_ratio_12,
                                               *dists_out_12, save_file=None, show=True)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            vis.plot_lc_ellc_fit(times, signal, p_orb_8, t_zero_11, timings_11, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                                 par_init_12, par_opt_13, i_sectors, save_file=None, show=True)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            vis.plot_corner_ellc_pars(par_init_12, par_opt_13, dists_out_12, save_file=None, show=True)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
    # pulsation_analysis
    if save_dir is not None:
        try:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_pulsation_analysis_pd.png')
            vis.plot_pd_pulsation_analysis(times, signal, p_orb_8, f_n_8, a_n_8, noise_level_8, passed_nh_b,
                                           save_file=file_name, show=False)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_pulsation_analysis_lc.png')
            vis.plot_lc_pulsation_analysis(times, signal, p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8, i_sectors,
                                           passed_nh_b, t_zero_11, const_r, f_n_r, a_n_r, ph_n_r, par_opt_13,
                                           save_file=file_name, show=False)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_pulsation_analysis_ellc_lc.png')
            vis.plot_lc_ellc_harmonics(times, signal, p_orb_8, t_zero_11, timings_11, const_8, slope_8,
                                       f_n_8, a_n_8, ph_n_8, i_sectors, const_r, f_n_r, a_n_r, ph_n_r, par_ellc,
                                       save_file=file_name, show=False)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_pulsation_analysis_ellc_pd.png')
            vis.plot_pd_ellc_harmonics(times, signal, p_orb_8, t_zero_11, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                                       noise_level_8, const_r, f_n_r, a_n_r, par_opt_13, i_sectors,
                                       save_file=file_name, show=False)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
    if show:
        try:
            vis.plot_pd_pulsation_analysis(times, signal, p_orb_8, f_n_8, a_n_8, noise_level_8, passed_nh_b,
                                           save_file=None, show=True)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            vis.plot_lc_pulsation_analysis(times, signal, p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8, i_sectors,
                                           passed_nh_b, t_zero_11, const_r, f_n_r, a_n_r, ph_n_r, par_opt_13,
                                           save_file=None, show=True)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            vis.plot_lc_ellc_harmonics(times, signal, p_orb_8, t_zero_11, timings_11, const_8, slope_8,
                                       f_n_8, a_n_8, ph_n_8, i_sectors, const_r, f_n_r, a_n_r, ph_n_r, par_ellc,
                                       save_file=None, show=True)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
        try:
            vis.plot_pd_ellc_harmonics(times, signal, p_orb_8, t_zero_11, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                                       noise_level_8, const_r, f_n_r, a_n_r, par_opt_13, i_sectors,
                                       save_file=None, show=True)
        except NameError:
            pass  # some variable wasn't loaded (file did not exist)
    return
