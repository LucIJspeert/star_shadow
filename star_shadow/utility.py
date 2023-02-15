"""STAR SHADOW
Satellite Time series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This module contains utility functions for data processing, unit conversions
and loading in data (some functions specific to TESS data).

Code written by: Luc IJspeert
"""

import os
import datetime
import fnmatch
import h5py
import numpy as np
import numba as nb
import astropy.io.fits as fits
import arviz as az

from . import timeseries_functions as tsf
from . import analysis_functions as af
from . import visualisation as vis


@nb.njit(cache=True)
def float_to_str(x, dec=2):
    """Convert float to string for Numba up to some decimal place
    
    Parameters
    ----------
    x: float
        Value to convert
    dec: int
        Number of decimals (be careful with large numbers here)
    
    Returns
    -------
    s: str
        String with the value x
    """
    x_round = np.round(x, dec)
    x_int = int(x_round)
    x_dec = int(np.round(x_round - x_int, dec) * 10**dec)
    s = str(x_int) + '.' + str(x_dec)
    return s


@nb.njit(cache=True)
def weighted_mean(x, w):
    """Weighted mean since Numba doesn't support numpy.average
    
    Parameters
    ----------
    x: numpy.ndarray[float]
        Values to calculate the mean over
    w: numpy.ndarray[float]
        Weights corresponding to each value
    
    Returns
    -------
    w_mean: float
        Mean of x weighted by w
    """
    w_mean = np.sum(x * w) / np.sum(w)
    return w_mean


@nb.njit(cache=True)
def interp_two_points(x, xp1, yp1, xp2, yp2):
    """Interpolate on a straight line between two points

    Parameters
    ----------
    x: numpy.ndarray[float]
        The x-coordinates at which to evaluate the interpolated values.
        All other inputs must have the same length.
    xp1: float, numpy.ndarray[float]
        The x-coordinate of the left point(s)
    yp1: float, numpy.ndarray[float]
        The y-coordinate of the left point(s)
    xp2: float, numpy.ndarray[float]
        The x-coordinate of the right point(s)
    yp2: float, numpy.ndarray[float]
        The y-coordinate of the right point(s)

    Returns
    -------
    y: numpy.ndarray[float]
        The interpolated values, same shape as x.
    """
    y_inter, slope = tsf.linear_pars_two_points(xp1, yp1, xp2, yp2)
    y = y_inter + slope * (x - (xp1 + xp2) / 2)  # assumes output of y_inter is for mean-centered x
    return y


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


def bounds_multiplicity_check(bounds, value):
    """Some bounds can have multiple intervals
    
    Parameters
    ----------
    bounds: numpy.ndarray
        One or more sets of bounding values
    value: float
        Value that is bounded (by one of the bounds)
    
    Returns
    -------
    bounds_1: numpy.ndarray[float]
        Bounds that contain the value
    bounds_2: numpy.ndarray[float], None
        Bounds that do not contain the value
    
    Notes
    -----
    If the value is not contained within one of the bounds
    the closest bounds are taken in terms of interval width.
    """
    if hasattr(bounds[0], '__len__'):
        if (len(bounds) == 1):
            bounds_1 = bounds[0]
            bounds_2 = None
        elif (len(bounds) > 1):
            # sign only changes if w is not in the interval
            sign_change = (np.sign((value - bounds[:, 0]) * (bounds[:, 1] - value)) == 1)
            if np.any(sign_change):
                bounds_1 = bounds[sign_change][0]
                bounds_2 = bounds[sign_change == False]
            else:
                # take the closest bounds (in terms of bound width)
                width = bounds[:, 1] - bounds[:, 0]
                mid = (bounds[:, 0] + bounds[:, 1]) / 2
                distance = np.abs(value - mid) / width
                closest = (distance == np.min(distance))
                bounds_1 = bounds[closest][0]
                bounds_2 = bounds[closest == False]
            # if bounds_2 still contains multiple bounds, pick the largest one
            if (len(bounds_2) == 1):
                bounds_2 = bounds_2[0]
            else:
                width = bounds_2[:, 1] - bounds_2[:, 0]
                bounds_2 = bounds_2[np.argmax(width)]
        else:
            bounds_1 = None
            bounds_2 = None
    else:
        bounds_1 = bounds
        bounds_2 = None
    return bounds_1, bounds_2


@nb.njit(cache=True)
def signal_to_noise_threshold(n_points):
    """Determine the signal-to-noise threshold for accepting frequencies
    based on the number of points
    
    Parameters
    ----------
    n_points: int
        Number of data points in the time series
    
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
def normalise_counts(flux_counts, i_sectors, flux_counts_err=None):
    """Median-normalises flux (counts or otherwise, should be positive) by
    dividing by the median.
    
    Parameters
    ----------
    flux_counts: numpy.ndarray[float]
        Flux measurement values in counts of the time series
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans.
        These can indicate the TESS observation sectors, but taking
        half the sectors is recommended. If only a single curve is
        wanted, set i_half_s = np.array([[0, len(times)]]).
    flux_counts_err: numpy.ndarray[float], None
        Errors in the flux measurements
    
    Returns
    -------
    flux_norm: numpy.ndarray[float]
        Normalised flux measurements
    median: numpy.ndarray[float]
        Median flux counts per sector
    flux_err_norm: numpy.ndarray[float]
        Normalised flux errors (zeros if flux_counts_err is None)
    
    Notes
    -----
    The result is positive and varies around one.
    The signal is processed per sector.
    """
    median = np.zeros(len(i_sectors))
    flux_norm = np.zeros(len(flux_counts))
    flux_err_norm = np.zeros(len(flux_counts))
    for i, s in enumerate(i_sectors):
        median[i] = np.median(flux_counts[s[0]:s[1]])
        flux_norm[s[0]:s[1]] = flux_counts[s[0]:s[1]] / median[i]
        if flux_counts_err is not None:
            flux_err_norm[s[0]:s[1]] = flux_counts_err[s[0]:s[1]] / median[i]
    return flux_norm, median, flux_err_norm


def get_tess_sectors(times, bjd_ref=2457000.0):
    """Load the times of the TESS sectors from a file and return a set of
    indices indicating the separate sectors in the time series.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    bjd_ref: float
        BJD reference date
        
    Returns
    -------
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the TESS observing sectors
    
    Notes
    -----
    Make sure to use the appropriate Baricentric Julian Date (BJD)
    reference date for your data set. This reference date is subtracted
    from the loaded sector dates.
    """
    # the 0.5 offset comes from test results, and the fact that no exact JD were found (just calendar days)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # absolute dir the script is in
    data_dir = script_dir.replace('star_shadow/star_shadow', 'star_shadow/data')
    jd_sectors = np.loadtxt(os.path.join(data_dir, 'tess_sectors.dat'), usecols=(2, 3)) - bjd_ref
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
        Timestamps of the time series
    t_sectors: numpy.ndarray[float]
        Pair(s) of times indicating the timespans of each sector
    
    Returns
    -------
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans.
        These can indicate the TESS observation sectors, but taking
        half the sectors is recommended.
    """
    starts = np.searchsorted(times, t_sectors[:, 0])
    ends = np.searchsorted(times, t_sectors[:, 1])
    i_sectors = np.column_stack((starts, ends))
    return i_sectors


@nb.njit(cache=True)
def time_zero_points(times, i_sectors):
    """Determines the time reference points to zero the time series
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the TESS observing sectors
    
    Returns
    -------
    times_fzp: float
        Zero point of the full time series
    times_szp: numpy.ndarray[float]
        Zero point(s) of the time series per observing sector
    
    Notes
    -----
    Mean-center the time array to reduce correlations
    """
    times_fzp = np.mean(times)
    times_szp = np.zeros(len(i_sectors))
    for i, s in enumerate(i_sectors):
        times_szp[i] = np.mean(times[s[0]:s[1]])
    return times_fzp, times_szp


def load_tess_data(file_name):
    """Load in the data from a single fits file, TESS specific.
    
    Parameters
    ----------
    file_name: str
        File name (including path) for loading the results.
    
    Returns
    -------
    times: numpy.ndarray[float]
        Timestamps of the time series
    sap_flux: numpy.ndarray[float]
        Raw measurement values of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
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
    The SAP flux is Simple Aperture Photometry, the processed data
    can be PDC_SAP or KSP_SAP depending on the data source.
    """
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
    # grab the time series data, sector number, start and stop time
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
        Whether to apply the quality flags to the time series data
        
    Returns
    -------
    times: numpy.ndarray[float]
        Timestamps of the time series
    sap_signal: numpy.ndarray[float]
        Raw measurement values of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
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
        t_sectors.append([ti[0] - dt / 2, ti[-1] + dt / 2])
        # append all other data
        times = np.append(times, ti)
        sap_signal = np.append(sap_signal, s_fl)
        signal = np.append(signal, fl)
        signal_err = np.append(signal_err, err)
        qual_flags = np.append(qual_flags, qf)
        sectors = np.append(sectors, sec)
        crowdsap = np.append(crowdsap, cro)
    t_sectors = np.array(t_sectors)
    # sort by sector (and merges duplicate sectors as byproduct)
    if np.any(np.diff(times) < 0):
        sec_sorter = np.argsort(sectors)
        time_sorter = np.argsort(times)
        times = times[time_sorter]
        sap_signal = sap_signal[time_sorter]
        signal = signal[time_sorter]
        signal_err = signal_err[time_sorter]
        sectors = sectors[sec_sorter]
        t_sectors = t_sectors[sec_sorter]
        crowdsap = crowdsap[sec_sorter]
    # apply quality flags
    if apply_flags:
        # convert quality flags to boolean mask
        quality = (qual_flags == 0)
        times = times[quality]
        sap_signal = sap_signal[quality]
        signal = signal[quality]
        signal_err = signal_err[quality]
    # clean up (only on times and signal, sap_signal assumed to be the same)
    finite = np.isfinite(times) & np.isfinite(signal)
    times = times[finite].astype(np.float_)
    sap_signal = sap_signal[finite].astype(np.float_)
    signal = signal[finite].astype(np.float_)
    signal_err = signal_err[finite].astype(np.float_)
    return times, sap_signal, signal, signal_err, sectors, t_sectors, crowdsap


def stitch_tess_sectors(times, signal, signal_err, i_sectors):
    """Stitches the different TESS sectors of a light curve together.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the TESS observing sectors
    
    Returns
    -------
    times: numpy.ndarray[float]
        Timestamps of the time series (shifted start at zero)
    signal: numpy.ndarray[float]
        Measurement values of the time series (normalised)
    signal_err: numpy.ndarray[float]
        Errors in the measurement values (normalised)
    medians: numpy.ndarray[float]
        Median flux counts per sector
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
    signal, medians, signal_err = normalise_counts(signal, i_sectors=i_sectors, flux_counts_err=signal_err)
    # times of sector mid-point and resulting half-sectors
    dt = np.median(np.diff(times))
    t_start = times[i_sectors[:, 0]] - dt / 2
    t_end = times[i_sectors[:, 1] - 1] + dt / 2
    t_mid = (t_start + t_end) / 2
    t_combined = np.column_stack((np.append(t_start, t_mid + dt / 2), np.append(t_mid - dt / 2, t_end)))
    i_half_s = convert_tess_t_sectors(times, t_combined)
    return times, signal, signal_err, medians, t_combined, i_half_s


def group_frequencies_for_fit(a_n, g_min=20, g_max=25):
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
    To make the task of fitting more manageable, the free parameters are binned into groups,
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


@nb.njit(cache=True)
def correct_for_crowdsap(signal, crowdsap, i_sectors):
    """Correct the signal for flux contribution of a third source
    
    Parameters
    ----------
    signal: numpy.ndarray[float]
        Measurement values of the time series
    crowdsap: list[float], numpy.ndarray[float]
        Light contamination parameter (1-third_light) listed per sector
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans.
        These can indicate the TESS observation sectors, but taking
        half the sectors is recommended. If only a single curve is
        wanted, set i_half_s = np.array([[0, len(times)]]).
    
    Returns
    -------
    cor_signal: numpy.ndarray[float]
        Measurement values of the time series corrected for
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


@nb.njit(cache=True)
def model_crowdsap(signal, crowdsap, i_sectors):
    """Incorporate flux contribution of a third source into the signal
    
    Parameters
    ----------
    signal: numpy.ndarray[float]
        Measurement values of the time series
    crowdsap: list[float], numpy.ndarray[float]
        Light contamination parameter (1-third_light) listed per sector
    i_sectors: numpy.ndarray[int]
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
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans.
        These can indicate the TESS observation sectors, but taking
        half the sectors is recommended. If only a single curve is
        wanted, set i_half_s = np.array([[0, len(times)]]).
    crowdsap: list[float], numpy.ndarray[float]
        Light contamination parameter (1-third_light) listed per sector
    verbose: bool
        If set to True, this function will print some information.
    
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
            print('Only one sector: no minimum third light can be inferred.')
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


def save_parameters_hdf5(file_name, sin_mean, sin_err, sin_hdi, sin_select, ecl_mean, ecl_err, ecl_hdi, timings,
                         timings_err, timings_hdi, var_stats, stats, i_sectors, description='none', data_id='none'):
    """Save the full model parameters of the linear, sinusoid
    and eclipse models to an hdf5 file.

    Parameters
    ----------
    file_name: str
        File name (including path) for saving the results.
    sin_mean: None, list[numpy.ndarray[float]]
        Parameter mean values for the linear and sinusoid model in the order they appear below.
        linear (these are arrays): const, slope,
        sinusoid (these are arrays): f_n, a_n, ph_n
    sin_err: None, list[numpy.ndarray[float]]
        Parameter error values for the linear and sinusoid model in the order they appear below.
        linear (these are arrays): c_err, sl_err,
        sinusoid (these are arrays): f_n_err, a_n_err, ph_n_err
    sin_hdi: None, list[numpy.ndarray[float]]
        Parameter hdi values for the linear and sinusoid model in the order they appear below.
        linear (these are arrays): c_err, sl_err,
        sinusoid (these are arrays): f_n_err, a_n_err, ph_n_err
    sin_select: None, list[numpy.ndarray[bool]]
        Sinusoids that pass certain selection criteria
        passed_sigma, passed_snr, passed_h
    ecl_mean: None, numpy.ndarray[float]
        Parameter mean values for the eclipse model in the order they appear below.
        eclipses: p_orb, t_zero, ecosw, esinw, cosi, phi_0, r_rat, sb_rat,
        eclipses (extra parametrisations): e, w, i, r_sum
    ecl_err: None, numpy.ndarray[float]
        Parameter error values for the eclipse model in the order they appear below.
        eclipses: p_err, t_zero_err, ecosw_err, esinw_err, cosi_err, phi_0_err, r_rat_err, sb_rat_err,
        eclipses (extra parametrisations): e_err, w_err, i_err, r_sum_err
    ecl_hdi: None, numpy.ndarray[float]
        Parameter hdi values for the eclipse model in the order they appear below.
        eclipses: p_err, t_zero_err, ecosw_err, esinw_err, cosi_err, phi_0_err, r_rat_err, sb_rat_err,
        eclipses (extra parametrisations): e_err, w_err, i_err, r_sum_err
    timings: None, numpy.ndarray[float]
        Eclipse timings of minima and first and last contact points,
        eclipse timings of the possible flat bottom (internal tangency),
        and eclipse depth of the primary and secondary:
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2
    timings_err: None, numpy.ndarray[float]
        Error estimates for the eclipse timings and depths:
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err, depth_1_err, depth_2_err
    timings_hdi: None, numpy.ndarray[float]
        Error estimates for the eclipse timings and depths, same format as timings
    var_stats: None, list[union(float, numpy.ndarray[float])]
        Varability level diagnostic statistics
        std_1, std_2, std_3, std_4, ratios_1, ratios_2, ratios_3, ratios_4
    stats: None, list[float]
        Some statistics: t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level
    i_sectors: None, numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve.
    description: str
        Optional description of the saved results
    data_id: int, str
        Optional identifier for the data set used
    
    Returns
    -------
    None

    Notes
    -----
    The file contains the data sets (array-like) and attributes
    to describe the data, in hdf5 format.
    
    Any missing data is filled up with -1
    """
    # check for Nones
    if sin_mean is None:
        sin_mean = [np.zeros(1) for _ in range(5)]
    if sin_err is None:
        sin_err = [np.zeros(1) for _ in range(5)]
    if sin_hdi is None:
        sin_hdi = [np.zeros((1, 2)) for _ in range(5)]
    if sin_select is None:
        sin_select = [np.array([True for _ in range(len(sin_mean[2]))]) for _ in range(3)]
    if ecl_mean is None:
        ecl_mean = -np.ones(12)
    if ecl_err is None:
        ecl_err = -np.ones(12)
    if ecl_hdi is None:
        ecl_hdi = -np.ones((12, 2))
    if timings is None:
        timings = -np.ones(12)
    if timings_err is None:
        timings_err = -np.ones(8)
    if timings_hdi is None:
        timings_hdi = -np.ones((12, 2))
    if var_stats is None:
        var_stats = [-1 for _ in range(4)] + [np.array([-1, -1]) for _ in range(4)]
    if stats is None:
        stats = [-1 for _ in range(7)]
    if i_sectors is None:
        i_sectors = -np.ones((1, 2))
    # unpack all the variables
    const, slope, f_n, a_n, ph_n = sin_mean
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = sin_err
    c_hdi, sl_hdi, f_n_hdi, a_n_hdi, ph_n_hdi = sin_hdi
    p_orb, t_zero, ecosw, esinw, cosi, phi_0, r_rat, sb_rat, e, w, i, r_sum = ecl_mean
    p_err, t_zero_err, ecosw_err, esinw_err, cosi_err, phi_0_err, r_rat_err, sb_rat_err = ecl_err[:8]
    e_err, w_err, i_err, r_sum_err = ecl_err[8:]
    p_hdi, t_zero_hdi, ecosw_hdi, esinw_hdi, cosi_hdi, phi_0_hdi, r_rat_hdi, sb_rat_hdi = ecl_hdi[:8]
    e_hdi, w_hdi, i_hdi, r_sum_hdi = ecl_hdi[8:]
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2 = timings
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err, depth_1_err, depth_2_err = timings_err
    t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err = t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err
    t_1_hdi, t_2_hdi, t_1_1_hdi, t_1_2_hdi, t_2_1_hdi, t_2_2_hdi = timings_hdi[:6]
    t_b_1_1_hdi, t_b_1_2_hdi, t_b_2_1_hdi, t_b_2_2_hdi, depth_1_hdi, depth_2_hdi = timings_hdi[6:]
    passed_sigma, passed_snr, passed_h = sin_select
    passed_b = (passed_sigma & passed_snr)  # passed both
    std_1, std_2, std_3, std_4, ratios_1, ratios_2, ratios_3, ratios_4 = var_stats
    t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level = stats
    # check some input
    ext = os.path.splitext(os.path.basename(file_name))[1]
    if (ext != '.hdf5'):
        file_name = file_name.replace(ext, '.hdf5')
    # create the file
    with h5py.File(file_name, 'w') as file:
        file.attrs['identifier'] = os.path.splitext(os.path.basename(file_name))[0]  # the file name without extension
        file.attrs['description'] = description
        file.attrs['data_id'] = data_id
        file.attrs['date_time'] = str(datetime.datetime.now())
        file.attrs['t_tot'] = t_tot  # total time base of observations
        file.attrs['t_mean'] = t_mean  # time reference (zero) point
        file.attrs['t_mean_s'] = t_mean_s  # time reference (zero) point per observing sector
        file.attrs['t_int'] = t_int  # integration time of observations
        file.attrs['n_param'] = n_param  # number of free parameters
        file.attrs['bic'] = bic  # Bayesian Information Criterion of the residuals
        file.attrs['noise_level'] = noise_level  # standard deviation of the residuals
        # orbital period and time of deepest eclipse
        file.create_dataset('p_orb', data=np.array([p_orb, p_err, p_hdi[0], p_hdi[1]]))
        file['p_orb'].attrs['unit'] = 'd'
        file['p_orb'].attrs['description'] = 'Orbital period and error estimates.'
        file.create_dataset('t_zero', data=np.array([t_zero, t_zero_err, t_zero_hdi[0], t_zero_hdi[1]]))
        file['t_zero'].attrs['unit'] = 'd'
        file['t_zero'].attrs['description'] = 'time of deepest eclipse with reference point t_mean and error estimates.'
        # the linear model
        # y-intercepts
        file.create_dataset('const', data=const)
        file['const'].attrs['unit'] = 'median normalised flux'
        file['const'].attrs['description'] = 'y-intercept per analysed sector'
        file.create_dataset('c_err', data=c_err)
        file['c_err'].attrs['unit'] = 'median normalised flux'
        file['c_err'].attrs['description'] = 'errors in the y-intercept per analysed sector'
        file.create_dataset('c_hdi', data=c_hdi)
        file['c_hdi'].attrs['unit'] = 'median normalised flux'
        file['c_hdi'].attrs['description'] = 'HDI for the y-intercept per analysed sector'
        # slopes
        file.create_dataset('slope', data=slope)
        file['slope'].attrs['unit'] = 'median normalised flux / d'
        file['slope'].attrs['description'] = 'slope per analysed sector'
        file.create_dataset('sl_err', data=sl_err)
        file['sl_err'].attrs['unit'] = 'median normalised flux / d'
        file['sl_err'].attrs['description'] = 'error in the slope per analysed sector'
        file.create_dataset('sl_hdi', data=sl_hdi)
        file['sl_hdi'].attrs['unit'] = 'median normalised flux / d'
        file['sl_hdi'].attrs['description'] = 'HDI for the slope per analysed sector'
        # sector indices
        file.create_dataset('i_sectors', data=i_sectors)
        file['i_sectors'].attrs['description'] = ('pairs of indices indicating chunks in time defining '
                                                  'the pieces of the piece-wise linear curve')
        # the sinusoid model
        # frequencies
        file.create_dataset('f_n', data=f_n)
        file['f_n'].attrs['unit'] = '1 / d'
        file['f_n'].attrs['description'] = 'frequencies of a number of sine waves'
        file.create_dataset('f_n_err', data=f_n_err)
        file['f_n_err'].attrs['unit'] = '1 / d'
        file['f_n_err'].attrs['description'] = 'errors in the frequencies of a number of sine waves'
        file.create_dataset('f_n_hdi', data=f_n_hdi)
        file['f_n_hdi'].attrs['unit'] = '1 / d'
        file['f_n_hdi'].attrs['description'] = 'HDI for the frequencies of a number of sine waves'
        # amplitudes
        file.create_dataset('a_n', data=a_n)
        file['a_n'].attrs['unit'] = 'median normalised flux'
        file['a_n'].attrs['description'] = 'amplitudes of a number of sine waves'
        file.create_dataset('a_n_err', data=a_n_err)
        file['a_n_err'].attrs['unit'] = 'median normalised flux'
        file['a_n_err'].attrs['description'] = 'errors in the amplitudes of a number of sine waves'
        file.create_dataset('a_n_hdi', data=a_n_hdi)
        file['a_n_hdi'].attrs['unit'] = 'median normalised flux'
        file['a_n_hdi'].attrs['description'] = 'HDI for the amplitudes of a number of sine waves'
        # phases
        file.create_dataset('ph_n', data=ph_n)
        file['ph_n'].attrs['unit'] = 'radians'
        file['ph_n'].attrs['description'] = 'phases of a number of sine waves, with reference point t_mean'
        file.create_dataset('ph_n_err', data=ph_n_err)
        file['ph_n_err'].attrs['unit'] = 'radians'
        file['ph_n_err'].attrs['description'] = 'errors in the phases of a number of sine waves'
        file.create_dataset('ph_n_hdi', data=ph_n_hdi)
        file['ph_n_hdi'].attrs['unit'] = 'radians'
        file['ph_n_hdi'].attrs['description'] = 'HDI for the phases of a number of sine waves'
        # selection criteria
        file.create_dataset('passed_sigma', data=passed_sigma)
        file['passed_sigma'].attrs['description'] = 'sinusoids passing the sigma criterion'
        file.create_dataset('passed_snr', data=passed_snr)
        file['passed_snr'].attrs['description'] = 'sinusoids passing the signal to noise criterion'
        file.create_dataset('passed_b', data=passed_b)
        file['passed_b'].attrs['description'] = 'sinusoids passing both the sigma and the signal to noise critera'
        file.create_dataset('passed_h', data=passed_h)
        file['passed_h'].attrs['description'] = 'harmonic sinusoids passing the sigma criterion'
        # eclipse timings for the empirical eclipse model
        # minima
        file.create_dataset('t_1', data=np.array([t_1, t_1_err, t_1_hdi[0], t_1_hdi[1]]))
        file['t_1'].attrs['description'] = 'time of primary minimum with respect to t_zero'
        file.create_dataset('t_2', data=np.array([t_2, t_2_err, t_2_hdi[0], t_2_hdi[1]]))
        file['t_2'].attrs['description'] = 'time of secondary minimum with respect to t_zero'
        # contact
        file.create_dataset('t_1_1', data=np.array([t_1_1, t_1_1_err, t_1_1_hdi[0], t_1_1_hdi[1]]))
        file['t_1_1'].attrs['description'] = 'time of primary first contact with respect to t_zero'
        file.create_dataset('t_1_2', data=np.array([t_1_2, t_1_2_err, t_1_2_hdi[0], t_1_2_hdi[1]]))
        file['t_1_2'].attrs['description'] = 'time of primary last contact with respect to t_zero'
        file.create_dataset('t_2_1', data=np.array([t_2_1, t_2_1_err, t_2_1_hdi[0], t_2_1_hdi[1]]))
        file['t_2_1'].attrs['description'] = 'time of secondary first contact with respect to t_zero'
        file.create_dataset('t_2_2', data=np.array([t_2_2, t_2_2_err, t_2_2_hdi[0], t_2_2_hdi[1]]))
        file['t_2_2'].attrs['description'] = 'time of secondary last contact with respect to t_zero'
        # internal tangency
        file.create_dataset('t_b_1_1', data=np.array([t_b_1_1, t_b_1_1_err, t_b_1_1_hdi[0], t_b_1_1_hdi[1]]))
        file['t_b_1_1'].attrs['description'] = 'time of primary first internal tangency with respect to t_zero'
        file.create_dataset('t_b_1_2', data=np.array([t_b_1_2, t_b_1_2_err, t_b_1_2_hdi[0], t_b_1_2_hdi[1]]))
        file['t_b_1_2'].attrs['description'] = 'time of primary last internal tangency with respect to t_zero'
        file.create_dataset('t_b_2_1', data=np.array([t_b_2_1, t_b_2_1_err, t_b_2_1_hdi[0], t_b_2_1_hdi[1]]))
        file['t_b_2_1'].attrs['description'] = 'time of secondary first internal tangency with respect to t_zero'
        file.create_dataset('t_b_2_2', data=np.array([t_b_2_2, t_b_2_2_err, t_b_2_2_hdi[0], t_b_2_2_hdi[1]]))
        file['t_b_2_2'].attrs['description'] = 'time of secondary last internal tangency with respect to t_zero'
        # depths
        file.create_dataset('depth_1', data=np.array([depth_1, depth_1_err, depth_1_hdi[0], depth_1_hdi[1]]))
        file['depth_1'].attrs['description'] = 'depth of primary minimum (median normalised flux)'
        file.create_dataset('depth_2', data=np.array([depth_2, depth_2_err, depth_2_hdi[0], depth_2_hdi[1]]))
        file['depth_2'].attrs['description'] = 'depth of secondary minimum (median normalised flux)'
        # variability to eclipse depth ratios
        file.create_dataset('ratios_1', data=np.array([std_1, ratios_1[0], ratios_1[1]]))
        desc = ('Standard deviation of the residuals of the linear+sinusoid+eclipse model, '
                'Ratio of the first eclipse depth to std_1, Ratio of the second eclipse depth to std_1')
        file['ratios_1'].attrs['description'] = desc
        file.create_dataset('ratios_2', data=np.array([std_2, ratios_2[0], ratios_2[1]]))
        desc = ('Standard deviation of the residuals of the linear+eclipse model, '
                'Ratio of the first eclipse depth to std_2, Ratio of the second eclipse depth to std_2')
        file['ratios_2'].attrs['description'] = desc
        file.create_dataset('ratios_3', data=np.array([std_3, ratios_3[0], ratios_3[1]]))
        desc = ('Standard deviation of the residuals of the linear+harmonic 1 and 2+eclipse model, '
                'Ratio of the first eclipse depth to std_3, Ratio of the second eclipse depth to std_3')
        file['ratios_3'].attrs['description'] = desc
        file.create_dataset('ratios_4', data=np.array([std_4, ratios_4[0], ratios_4[1]]))
        desc = ('Standard deviation of the residuals of the linear+non-harmonic sinusoid+eclipse model, '
                'Ratio of the first eclipse depth to std_4, Ratio of the second eclipse depth to std_4')
        file['ratios_4'].attrs['description'] = desc
        # the physical eclipse model parameters
        file.create_dataset('ecosw', data=np.array([ecosw, ecosw_err, ecosw_hdi[0], ecosw_hdi[1]]))
        file['ecosw'].attrs['description'] = 'tangential part of the eccentricity'
        file.create_dataset('esinw', data=np.array([esinw, esinw_err, esinw_hdi[0], esinw_hdi[1]]))
        file['esinw'].attrs['description'] = 'radial part of the eccentricity'
        file.create_dataset('cosi', data=np.array([cosi, cosi_err, cosi_hdi[0], cosi_hdi[1]]))
        file['cosi'].attrs['description'] = 'cosine of the inclination'
        file.create_dataset('phi_0', data=np.array([phi_0, phi_0_err, phi_0_hdi[0], phi_0_hdi[1]]))
        file['phi_0'].attrs['description'] = 'auxilary angle of Kopal 1959, measures the sum of eclipse durations'
        file.create_dataset('r_rat', data=np.array([r_rat, r_rat_err, r_rat_hdi[0], r_rat_hdi[1]]))
        file['r_rat'].attrs['description'] = 'ratio of radii r_2 / r_1'
        file.create_dataset('sb_rat', data=np.array([sb_rat, sb_rat_err, sb_rat_hdi[0], sb_rat_hdi[1]]))
        file['sb_rat'].attrs['description'] = 'ratio of surface brightnesses sb_2 / sb_1'
        # some alternate parameterisations
        file.create_dataset('e', data=np.array([e, e_err, e_hdi[0], e_hdi[1]]))
        file['e'].attrs['description'] = 'orbital eccentricity'
        file.create_dataset('w', data=np.array([w, w_err, w_hdi[0], w_hdi[1]]))
        file['w'].attrs['description'] = 'argument of periastron'
        file.create_dataset('i', data=np.array([i, i_err, i_hdi[0], i_hdi[1]]))
        file['i'].attrs['description'] = 'orbital inclination'
        file.create_dataset('r_sum', data=np.array([r_sum, r_sum_err, r_sum_hdi[0], r_sum_hdi[1]]))
        file['r_sum'].attrs['description'] = 'sum of radii scaled to the semi-major axis'
    return None


def read_parameters_hdf5(file_name, verbose=False):
    """Read the full model parameters of the linear, sinusoid
    and eclipse models to an hdf5 file.

    Parameters
    ----------
    file_name: str
        File name (including path) for loading the results.
    verbose: bool
        If set to True, this function will print some information.

    Returns
    -------
    sin_mean: list[numpy.ndarray[float]]
        Parameter mean values for the linear and sinusoid model in the order they appear below.
        linear (these are arrays): const, slope,
        sinusoid (these are arrays): f_n, a_n, ph_n,
    sin_err: list[numpy.ndarray[float]]
        Parameter error values for the linear and sinusoid model in the order they appear below.
        linear (these are arrays): c_err, sl_err,
        sinusoid (these are arrays): f_n_err, a_n_err, ph_n_err,
    sin_hdi: list[numpy.ndarray[float]]
        Parameter hdi values for the linear and sinusoid model in the order they appear below.
        linear (these are arrays): c_err, sl_err,
        sinusoid (these are arrays): f_n_err, a_n_err, ph_n_err,
    sin_select: list[numpy.ndarray[bool]]
        Sinusoids that pass certain selection criteria
        passed_sigma, passed_snr, passed_b, passed_h
    ecl_mean: numpy.ndarray[float]
        Parameter mean values for the eclipse model in the order they appear below.
        eclipses: p_orb, t_zero, ecosw, esinw, cosi, phi_0, r_rat, sb_rat,
        eclipses (extra parametrisations): e, w, i, r_sum
    ecl_err: numpy.ndarray[float]
        Parameter error values for the eclipse model in the order they appear below.
        eclipses: p_err, t_zero_err, ecosw_err, esinw_err, cosi_err, phi_0_err, r_rat_err, sb_rat_err,
        eclipses (extra parametrisations): e_err, w_err, i_err, r_sum_err
    ecl_hdi: numpy.ndarray[float]
        Parameter hdi values for the eclipse model in the order they appear below.
        eclipses: p_err, t_zero_err, ecosw_err, esinw_err, cosi_err, phi_0_err, r_rat_err, sb_rat_err,
        eclipses (extra parametrisations): e_err, w_err, i_err, r_sum_err
    timings: numpy.ndarray[float]
        Eclipse timings of minima and first and last contact points,
        eclipse timings of the possible flat bottom (internal tangency),
        and eclipse depth of the primary and secondary:
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2
    timings_err: numpy.ndarray[float]
        Error estimates for the eclipse timings and depths:
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err, depth_1_err, depth_2_err
    timings_hdi: numpy.ndarray[float]
        Error estimates for the eclipse timings and depths:
        t_1_hdi, t_2_hdi, t_1_1_hdi, t_1_2_hdi, t_2_1_hdi, t_2_2_hdi,
        t_b_1_1_hdi, t_b_1_2_hdi, t_b_2_1_hdi, t_b_2_2_hdi, depth_1_hdi, depth_2_hdi
    var_stats: list[union(float, numpy.ndarray[float])]
        Varability level diagnostic statistics
        std_1, std_2, std_3, std_4, ratios_1, ratios_2, ratios_3, ratios_4
    stats: list[float]
        Some statistics: t_tot, t_mean, t_mean_s, n_param, bic, noise_level
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve.
    text: list[str]
        Some information about the file and data:
        identifier, data_id, description and date_time
    """
    # check some input
    ext = os.path.splitext(os.path.basename(file_name))[1]
    if (ext != '.hdf5'):
        file_name = file_name.replace(ext, '.hdf5')
    # create the file
    with h5py.File(file_name, 'r') as file:
        identifier = file.attrs['identifier']
        description = file.attrs['description']
        data_id = file.attrs['data_id']
        date_time = file.attrs['date_time']
        t_tot = file.attrs['t_tot']
        t_mean = file.attrs['t_mean']
        t_mean_s = file.attrs['t_mean_s']
        t_int = file.attrs['t_int']
        n_param = file.attrs['n_param']
        bic = file.attrs['bic']
        noise_level = file.attrs['noise_level']
        # orbital period and time of deepest eclipse
        p_orb = np.copy(file['p_orb'])
        t_zero = np.copy(file['t_zero'])
        # the linear model
        # y-intercepts
        const = np.copy(file['const'])
        c_err = np.copy(file['c_err'])
        c_hdi = np.copy(file['c_hdi'])
        # slopes
        slope = np.copy(file['slope'])
        sl_err = np.copy(file['sl_err'])
        sl_hdi = np.copy(file['sl_hdi'])
        # sector indices
        i_sectors = np.copy(file['i_sectors'])
        # the sinusoid model
        # frequencies
        f_n = np.copy(file['f_n'])
        f_n_err = np.copy(file['f_n_err'])
        f_n_hdi = np.copy(file['f_n_hdi'])
        # amplitudes
        a_n = np.copy(file['a_n'])
        a_n_err = np.copy(file['a_n_err'])
        a_n_hdi = np.copy(file['a_n_hdi'])
        # phases
        ph_n = np.copy(file['ph_n'])
        ph_n_err = np.copy(file['ph_n_err'])
        ph_n_hdi = np.copy(file['ph_n_hdi'])
        # passing criteria
        passed_sigma = np.copy(file['passed_sigma'])
        passed_snr = np.copy(file['passed_snr'])
        passed_b = np.copy(file['passed_b'])
        passed_h = np.copy(file['passed_h'])
        # eclipse timings for the empirical eclipse model
        t_1 = np.copy(file['t_1'])
        t_2 = np.copy(file['t_2'])
        t_1_1 = np.copy(file['t_1_1'])
        t_1_2 = np.copy(file['t_1_2'])
        t_2_1 = np.copy(file['t_2_1'])
        t_2_2 = np.copy(file['t_2_2'])
        t_b_1_1 = np.copy(file['t_b_1_1'])
        t_b_1_2 = np.copy(file['t_b_1_2'])
        t_b_2_1 = np.copy(file['t_b_2_1'])
        t_b_2_2 = np.copy(file['t_b_2_2'])
        depth_1 = np.copy(file['depth_1'])
        depth_2 = np.copy(file['depth_2'])
        # variability to eclipse depth ratios
        ratios_1 = np.copy(file['ratios_1'])
        ratios_2 = np.copy(file['ratios_2'])
        ratios_3 = np.copy(file['ratios_3'])
        ratios_4 = np.copy(file['ratios_4'])
        # the physical eclipse model parameters
        ecosw = np.copy(file['ecosw'])
        esinw = np.copy(file['esinw'])
        cosi = np.copy(file['cosi'])
        phi_0 = np.copy(file['phi_0'])
        r_rat = np.copy(file['r_rat'])
        sb_rat = np.copy(file['sb_rat'])
        # some alternate parameterisations
        e = np.copy(file['e'])
        w = np.copy(file['w'])
        i = np.copy(file['i'])
        r_sum = np.copy(file['r_sum'])
    
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    sin_hdi = [c_hdi, sl_hdi, f_n_hdi, a_n_hdi, ph_n_hdi]
    sin_select = [passed_sigma, passed_snr, passed_b, passed_h]
    ecl_mean = [p_orb[0], t_zero[0], ecosw[0], esinw[0], cosi[0], phi_0[0], r_rat[0], sb_rat[0],
                e[0], w[0], i[0], r_sum[0]]
    ecl_err = [p_orb[1], t_zero[1], ecosw[1], esinw[1], cosi[1], phi_0[1], r_rat[1], sb_rat[1],
               e[1], w[1], i[1], r_sum[1]]
    ecl_hdi = [p_orb[2:4], t_zero[2:4], ecosw[2:4], esinw[2:4], cosi[2:4], phi_0[2:4], r_rat[2:4], sb_rat[2:4],
               e[2:4], w[2:4], i[2:4], r_sum[2:4]]
    timings = np.array([t_1[0], t_2[0], t_1_1[0], t_1_2[0], t_2_1[0], t_2_2[0],
                        t_b_1_1[0], t_b_1_2[0], t_b_2_1[0], t_b_2_2[0], depth_1[0], depth_2[0]])
    timings_err = np.array([t_1[1], t_2[1], t_1_1[1], t_1_2[1], t_2_1[1], t_2_2[1], depth_1[1], depth_2[1]])
    # t_b_1_1[1], t_b_1_2[1], t_b_2_1[1], t_b_2_2[1]
    timings_hdi = np.array([t_1[2:4], t_2[2:4], t_1_1[2:4], t_1_2[2:4], t_2_1[2:4], t_2_2[2:4],
                            t_b_1_1[2:4], t_b_1_2[2:4], t_b_2_1[2:4], t_b_2_2[2:4], depth_1[2:4], depth_2[2:4]])
    var_stats = [ratios_1[0], ratios_2[0], ratios_3[0], ratios_4[0],
                 ratios_1[1:], ratios_2[1:], ratios_3[1:], ratios_4[1:]]
    stats = [t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level]
    text = [identifier, data_id, description, date_time]
    if verbose:
        print(f'Loaded analysis file with identifier: {identifier}, created on {date_time}. \n'
              f'data_id: {data_id}. Description: {description} \n')
    return (sin_mean, sin_err, sin_hdi, sin_select, ecl_mean, ecl_err, ecl_hdi, timings, timings_err, timings_hdi,
            var_stats, stats, i_sectors, text)


def convert_hdf5_to_ascii(file_name):
    """Convert a save file in hdf5 format to multiple ascii save files
    
    Parameters
    ----------
    file_name: str
        File name (including path) for saving the results.
    
    Returns
    -------
    None
    
    Notes
    -----
    Only saves text files of certain groups of values when they
    are available in the hdf5 file.
    """
    # check the file name
    ext = os.path.splitext(os.path.basename(file_name))[1]
    if (ext != '.hdf5'):
        file_name = file_name.replace(ext, '.hdf5')
    data = read_parameters_hdf5(file_name, verbose=False)
    sin_mean, sin_err, sin_hdi, sin_select, ecl_mean, ecl_err, ecl_hdi, timings, timings_err, timings_hdi = data[:10]
    var_stats, stats, i_sectors, text = data[10:]
    # unpack all parameters
    const, slope, f_n, a_n, ph_n = sin_mean
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = sin_err
    c_hdi, sl_hdi, f_n_hdi, a_n_hdi, ph_n_hdi = sin_hdi
    p_orb, t_zero, ecosw, esinw, cosi, phi_0, r_rat, sb_rat, e, w, i, r_sum = ecl_mean
    p_err, t_zero_err, ecosw_err, esinw_err, cosi_err, phi_0_err, r_rat_err, sb_rat_err = ecl_err[:8]
    e_err, w_err, i_err, r_sum_err = ecl_err[8:]
    p_hdi, t_zero_hdi, ecosw_hdi, esinw_hdi, cosi_hdi, phi_0_hdi, r_rat_hdi, sb_rat_hdi = ecl_hdi[:8]
    e_hdi, w_hdi, i_hdi, r_sum_hdi = ecl_hdi[8:]
    t_1_hdi, t_2_hdi, t_1_1_hdi, t_1_2_hdi, t_2_1_hdi, t_2_2_hdi = timings_hdi[:6]
    t_b_1_1_hdi, t_b_1_2_hdi, t_b_2_1_hdi, t_b_2_2_hdi, depth_1_hdi, depth_2_hdi = timings_hdi[6:]
    passed_sigma, passed_snr, passed_b, passed_h = sin_select
    std_1, std_2, std_3, std_4, ratios_1, ratios_2, ratios_3, ratios_4 = var_stats
    t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level = stats
    target_id, data_id, description, date_time = text
    # check for -1 values and ignore those parameters - errors or hdis are always stored alongside
    # linear model parameters
    if (not np.all(const == -1)) & (not np.all(slope == -1)):
        data = np.column_stack((const, c_err, c_hdi[:, 0], c_hdi[:, 1],
                                slope, sl_err, sl_hdi[:, 0], sl_hdi[:, 1],
                                i_sectors[:, 0], i_sectors[:, 1]))
        hdr = ('const, c_err, c_hdi_l, c_hdi_r, slope, sl_err, sl_hdi_l, sl_hdi_r, sector_start, sector_end')
        file_name_lin = file_name.replace(ext, '_linear.csv')
        np.savetxt(file_name_lin, data, delimiter=',', header=hdr)
    # sinusoid model parameters
    if (not np.all(f_n == -1)) & (not np.all(a_n == -1)) & (not np.all(ph_n == -1)):
        data = np.column_stack((f_n, f_n_err, f_n_hdi[:, 0], f_n_hdi[:, 1],
                                a_n, a_n_err, a_n_hdi[:, 0], a_n_hdi[:, 1],
                                ph_n, ph_n_err, ph_n_hdi[:, 0], ph_n_hdi[:, 1],
                                passed_sigma, passed_snr, passed_b, passed_h))
        hdr = ('f_n, f_n_err, f_n_hdi_l, f_n_hdi_r, a_n, a_n_err, a_n_hdi_l, a_n_hdi_r, '
               'ph_n, ph_n_err, ph_n_hdi_l, ph_n_hdi_r, passed_sigma, passed_snr, passed_b, passed_h')
        file_name_sin = file_name.replace(ext, '_sinusoid.csv')
        np.savetxt(file_name_sin, data, delimiter=',', header=hdr)
    # eclipse timings
    if not np.all(timings == -1):
        names = ('t_1', 't_2', 't_1_1', 't_1_2', 't_2_1', 't_2_2', 't_b_1_1', 't_b_1_2', 't_b_2_1', 't_b_2_2',
                 'depth_1', 'depth_2')
        hdi_l = (t_1_hdi[0], t_2_hdi[0], t_1_1_hdi[0], t_1_2_hdi[0], t_2_1_hdi[0], t_2_2_hdi[0],
                 t_b_1_1_hdi[0], t_b_1_2_hdi[0], t_b_2_1_hdi[0], t_b_2_2_hdi[0], depth_1_hdi[0], depth_2_hdi[0])
        hdi_r = (t_1_hdi[1], t_2_hdi[1], t_1_1_hdi[1], t_1_2_hdi[1], t_2_1_hdi[1], t_2_2_hdi[1],
                 t_b_1_1_hdi[1], t_b_1_2_hdi[1], t_b_2_1_hdi[1], t_b_2_2_hdi[1], depth_1_hdi[1], depth_2_hdi[1])
        var_desc = ['time of primary minimum with respect to t_zero',
                    'time of secondary minimum with respect to t_zero',
                    'time of primary first contact with respect to t_zero',
                    'time of primary last contact with respect to t_zero',
                    'time of secondary first contact with respect to t_zero',
                    'time of secondary last contact with respect to t_zero',
                    'time of primary first internal tangency with respect to t_zero',
                    'time of primary last internal tangency with respect to t_zero',
                    'time of secondary first internal tangency with respect to t_zero',
                    'time of secondary last internal tangency with respect to t_zero',
                    'depth of primary minimum (median normalised flux)',
                    'depth of secondary minimum (median normalised flux)']
        data = np.column_stack((names, timings, timings_err, hdi_l, hdi_r, var_desc))
        description = 'Eclipse timings and their error estimates'
        hdr = f'{target_id}, {data_id}, {description}\nname, value, error, hdi_l, hdi_r, description'
        file_name_tim = file_name.replace(ext, '_timings.csv')
        np.savetxt(file_name_tim, data, delimiter=',', header=hdr, fmt='%s')
    # variability statistics
    if not np.all(var_stats[:4] == -1):
        names = ['std_1', 'std_2', 'std_3', 'std_4', 'ratio_1_1', 'ratio_1_2', 'ratio_2_1', 'ratio_2_2',
                 'ratio_3_1', 'ratio_3_2', 'ratio_4_1', 'ratio_4_2']
        data = np.array([std_1, std_2, std_3, std_4, ratios_1[0], ratios_1[1],
                         ratios_2[0], ratios_2[1], ratios_3[0], ratios_3[1],
                         ratios_4[0], ratios_4[1]]).astype(str)
        desc = ['Standard deviation of the residuals of the linear+sinusoid+eclipse model',
                'Standard deviation of the residuals of the linear+eclipse model',
                'Standard deviation of the residuals of the linear+harmonic 1 and 2+eclipse model',
                'Standard deviation of the residuals of the linear+non-harmonic sinusoid+eclipse model',
                'Ratio of the first eclipse depth to std_1', 'Ratio of the second eclipse depth to std_1',
                'Ratio of the first eclipse depth to std_2', 'Ratio of the second eclipse depth to std_2',
                'Ratio of the first eclipse depth to std_3', 'Ratio of the second eclipse depth to std_3',
                'Ratio of the first eclipse depth to std_4', 'Ratio of the second eclipse depth to std_4']
        table = np.column_stack((names, data, desc))
        target_id = os.path.splitext(os.path.basename(file_name))[0]  # the file name without extension
        description = 'Variability levels of different model residuals'
        hdr = f'{target_id}, {data_id}, {description}\nname, value, description'
        file_name_var = file_name.replace(ext, '_var_stats.csv')
        np.savetxt(file_name_var, table, delimiter=',', fmt='%s', header=hdr)
    # eclipse model parameters
    values = np.array([p_orb, t_zero, ecosw, esinw, cosi, phi_0, r_rat, sb_rat, e, w, i, r_sum])
    if (not np.all(values == -1)):
        names = ('p_orb', 't_zero', 'ecosw', 'esinw', 'cosi', 'phi_0', 'r_rat', 'sb_rat', 'e', 'w', 'i', 'r_sum')
        errors = (p_err, t_zero_err, ecosw_err, esinw_err, cosi_err, phi_0_err, r_rat_err, sb_rat_err,
                  e_err, w_err, i_err, r_sum_err)
        hdi_l = (p_hdi[0], t_zero_hdi[0], ecosw_hdi[0], esinw_hdi[0], cosi_hdi[0], phi_0_hdi[0], r_rat_hdi[0],
                 sb_rat_hdi[0], e_hdi[0], w_hdi[0], i_hdi[0], r_sum_hdi[0])
        hdi_r = (p_hdi[1], t_zero_hdi[1], ecosw_hdi[1], esinw_hdi[1], cosi_hdi[1], phi_0_hdi[1], r_rat_hdi[1],
                 sb_rat_hdi[1], e_hdi[1], w_hdi[1], i_hdi[1], r_sum_hdi[1])
        desc = ['Orbital period', 'time of deepest eclipse with reference point t_mean',
                'tangential part of the eccentricity', 'radial part of the eccentricity',
                'cosine of the orbital inclination',
                'auxilary angle of Kopal 1959, measures the sum of eclipse durations',
                'ratio of radii r_2 / r_1', 'ratio of surface brightnesses sb_2 / sb_1', 'orbital eccentricity',
                'argument of periastron', 'orbital inclination', 'sum of radii scaled to the semi-major axis']
        data = np.column_stack((names, values, errors, hdi_l, hdi_r, desc))
        description = 'Eclipse model parameters and their error estimates'
        hdr = f'{target_id}, {data_id}, {description}\nname, value, error, hdi_l, hdi_r, description'
        file_name_ecl = file_name.replace(ext, '_eclipse.csv')
        np.savetxt(file_name_ecl, data, delimiter=',', header=hdr, fmt='%s')
    # statistics
    if not np.all(stats == -1):
        names = ('t_tot', 't_mean', 't_int', 'n_param', 'bic', 'noise_level')
        stats = (t_tot, t_mean, t_int, n_param, bic, noise_level)
        desc = ['Total time base of observations', 'Time reference (zero) point',
                'Integration time of observations', 'Number of free parameters',
                'Bayesian Information Criterion of the residuals', 'Standard deviation of the residuals']
        data = np.column_stack((names, stats, desc))
        description = 'Time series and model statistics'
        hdr = f'{target_id}, {data_id}, {description}\nname, value, description'
        file_name_stats = file_name.replace(ext, '_stats.csv')
        np.savetxt(file_name_stats, data, delimiter=',', header=hdr, fmt='%s')
    return None


def save_results_ecl_indices(file_name, ecl_indices, data_id='none'):
    """Save the eclipse indices of the eclipse timings

    Parameters
    ----------
    file_name: str
        File name (including path) for saving the results.
        This name is altered slightly to not interfere with another
        save file (from save_results_timings).
    ecl_indices: numpy.ndarray[int]
        Indices of several important points in the harmonic model
        as generated here (see function for details)
    data_id: int, str
        Identification for the dataset used

    Returns
    -------
    None
    """
    split_name = os.path.splitext(os.path.basename(file_name))
    target_id = split_name[0]  # the file name without extension
    fn_ext = split_name[1]
    file_name_2 = file_name.replace(fn_ext, '_ecl_indices.csv')
    description = 'Eclipse indices (see function measure_eclipses_dt).'
    hdr = (f'{target_id}, {data_id}, {description}\nzeros_1, minimum_1, peaks_2_n, peaks_1, p_2_p, zeros_1_in, '
           f'minimum_0, zeros_1_in, p_2_p, peaks_1, peaks_2_n, minimum_1, zeros_1')
    np.savetxt(file_name_2, ecl_indices, delimiter=',', fmt='%s', header=hdr)
    return None


def read_results_ecl_indices(file_name):
    """Read in the eclipse indices of the eclipse timings

    Parameters
    ----------
    file_name: str
        File name (including path) for loading the results.

    Returns
    -------
    ecl_indices: numpy.ndarray[int]
        Indices of several important points in the harmonic model
        as generated here (see function for details)
    """
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_2 = file_name.replace(fn_ext, '_ecl_indices.csv')
    ecl_indices = np.loadtxt(file_name_2, delimiter=',', dtype=int)
    return ecl_indices


def save_results_dists(file_name, dists_in, dists_out, data_id='none'):
    """Save the distributions of the determination of orbital elements
    
    Parameters
    ----------
    file_name: str
        File name (including path) for saving the results.
    dists_in: tuple[numpy.ndarray[float]]
        Full input distributions for: p, t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2,
        t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, d_1, d_2
    dists_out: tuple[numpy.ndarray[float]]
        Full output distributions for the same parameters as intervals
    data_id: int, str
        Identification for the dataset used
    
    Returns
    -------
    None
    """
    data = np.column_stack((*dists_in, *dists_out))
    # file name just adds 'dists' at the end
    split_name = os.path.splitext(os.path.basename(file_name))
    target_id = split_name[0]  # the file name without extension
    fn_ext = split_name[1]
    file_name_2 = file_name.replace(fn_ext, '_dists' + fn_ext)
    description = 'Prior and posterior distributions (not MCMC).'
    hdr = (f'{target_id}, {data_id}, {description}\n'
           'p_vals, t_1_vals, t_2_vals, t_1_1_vals, t_1_2_vals, t_2_1_vals, t_2_2_vals, '
           't_b_1_1_vals, t_b_1_2_vals, t_b_2_1_vals, t_b_2_2_vals, d_1_vals, d_2_vals, '
           'e_vals, w_vals, i_vals, rsumsma_vals, rratio_vals, sbratio_vals')
    np.savetxt(file_name_2, data, delimiter=',', header=hdr)
    return None


def read_results_dists(file_name):
    """Read the distributions of the determination of orbital elements

    Parameters
    ----------
    file_name: str
        File name (including path) for saving the results.

    Returns
    -------
    dists_in: tuple[numpy.ndarray[float]]
        Full input distributions for: p, t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2,
        t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, d_1, d_2
    dists_out: tuple[numpy.ndarray[float]]
        Full output distributions for the same parameters as intervals
    """
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_2 = file_name.replace(fn_ext, '_dists' + fn_ext)
    all_dists = np.loadtxt(file_name_2, delimiter=',', unpack=True)
    dists_in = all_dists[:13]
    dists_out = all_dists[13:]
    return dists_in, dists_out


def save_inference_data(file_name, inf_data):
    """Save the inference data object from Arviz/PyMC3
    
        Parameters
    ----------
    file_name: str
        File name (including path) for saving the results.
    inf_data: object
        Arviz inference data object

    Returns
    -------
    None
    """
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_mc = file_name.replace(fn_ext, '_dists.nc4')
    inf_data.to_netcdf(file_name_mc)
    return None


def read_inference_data(file_name):
    """Read the inference data object from Arviz/PyMC3

    Parameters
    ----------
    file_name: str
        File name (including path) for saving the results.

    Returns
    -------
    inf_data: object
        Arviz inference data object
    """
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_mc = file_name.replace(fn_ext, '_dists.nc4')
    inf_data = az.from_netcdf(file_name_mc)
    return inf_data


def save_summary(target_id, save_dir, data_id='none'):
    """Create a summary file from the results of the analysis
    
    Parameters
    ----------
    target_id: int, str
        Target identifier
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    data_id: int, str
        Identification for the dataset used
    
    Returns
    -------
    None
    
    Notes
    -----
    Meant both as a quick overview of the results and to facilitate
    the compilation of a catalogue of a set of results
    """
    prew_par = -np.ones(5)
    timings_par = -np.ones(25)
    form_par = -np.ones(21)
    fit_par_init = -np.ones(6)
    fit_par = -np.ones(9)
    freqs_par = -np.ones(5, dtype=int)
    level_par = -np.ones(12)
    t_tot, t_mean = 0, 0
    # read results
    save_dir = os.path.join(save_dir, f'{target_id}_analysis')  # add subdir
    # get period from last prewhitening step
    file_name_3 = os.path.join(save_dir, f'{target_id}_analysis_3.hdf5')
    file_name_5 = os.path.join(save_dir, f'{target_id}_analysis_5.hdf5')
    if os.path.isfile(file_name_5):
        results = read_parameters_hdf5(file_name_5, verbose=False)
        _, _, _, _, ecl_mean, ecl_err, _, _, _, _, _, stats, _, _ = results
        p_orb, _, _, _, _, _, _, _, _, _, _, _ = ecl_mean
        p_err, _, _, _, _, _, _, _, _, _, _, _ = ecl_err
        t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level = stats
        prew_par = [p_orb, p_err, n_param, bic, noise_level]
    elif os.path.isfile(file_name_3):
        results = read_parameters_hdf5(file_name_3, verbose=False)
        _, _, _, _, ecl_mean, ecl_err, _, _, _, _, _, stats, _, _ = results
        p_orb, _, _, _, _, _, _, _, _, _, _, _ = ecl_mean
        p_err, _, _, _, _, _, _, _, _, _, _, _ = ecl_err
        t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level = stats
        prew_par = [p_orb, p_err, n_param, bic, noise_level]
    # load timing results (9)
    file_name = os.path.join(save_dir, f'{target_id}_analysis_9.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        _, _, _, _, ecl_mean, _, _, timings, timings_err, _, _, stats, _, _ = results
        _, t_zero, _, _, _, _, _, _, _, _, _, _ = ecl_mean
        t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level = stats
        _, _, p_t_corr = af.linear_regression_uncertainty(prew_par[0], t_tot, sigma_t=t_int)
        timings_par = [t_zero, *timings, *timings_err, p_t_corr, n_param, bic, noise_level]
    # load parameter results from formulae (10)
    file_name = os.path.join(save_dir, f'{target_id}_analysis_10.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        _, _, _, _, ecl_mean, ecl_err, ecl_hdi, _, _, _, _, _, _, _ = results
        _, _, ecosw, esinw, cosi, phi_0, r_rat, sb_rat, e, w, i, r_sum = ecl_mean
        _, _, sigma_ecosw, sigma_esinw, _, sigma_phi_0, _, _, sigma_e, sigma_w, _, sigma_r_sum = ecl_err
        _, _, ecosw_err, esinw_err, cosi_err, phi_0_err, r_rat_err, sb_rat_err = ecl_hdi[:8]
        e_err, w_err, i_err, r_sum_err = ecl_hdi[8:]
        elem = [e, e_err[1], e_err[0], sigma_e, w, w_err[1], w_err[0], sigma_w, i, i_err[1], i_err[0],
                r_sum, r_sum_err[1], r_sum_err[0], sigma_r_sum, r_rat, r_rat_err[1], r_rat_err[0],
                sb_rat, sb_rat_err[1], sb_rat_err[0]]
        form_par = elem
    # load parameter results from initial fit (11)
    file_name = os.path.join(save_dir, f'{target_id}_analysis_11.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        _, _, _, _, ecl_mean, _, _, _, _, _, _, _, _, _ = results
        _, _, ecosw, esinw, cosi, phi_0, r_rat, sb_rat, e, w, i, r_sum = ecl_mean
        fit_par_init = [e, w, i, r_sum, r_rat, sb_rat]
    # load parameter results from full fit (13)
    file_name = os.path.join(save_dir, f'{target_id}_analysis_13.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        _, _, _, _, ecl_mean, _, _, _, _, _, _, stats, _, _ = results
        _, t_zero, ecosw, esinw, cosi, phi_0, r_rat, sb_rat, e, w, i, r_sum = ecl_mean
        t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level = stats
        fit_par = [e, w, i, r_sum, r_rat, sb_rat, n_param, bic, noise_level]
    # include n_freqs/n_freqs_passed (14)
    file_name = os.path.join(save_dir, f'{target_id}_analysis_14.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        _, _, _, sin_select, _, _, _, _, _, _, _, _, _, _ = results
        passed_sigma, passed_snr, passed_both, passed_h = sin_select
        freqs_par = [len(passed_both), np.sum(passed_sigma), np.sum(passed_snr), np.sum(passed_both),
                     np.sum(passed_h)]
    # include variability stats (15)
    file_name = os.path.join(save_dir, f'{target_id}_analysis_15.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        _, _, _, _, _, _, _, _, _, _, var_stats, _, _, _ = results
        std_1, std_2, std_3, std_4, ratios_1, ratios_2, ratios_3, ratios_4 = var_stats
        level_par = [std_1, std_2, std_3, std_4, *ratios_1, *ratios_2, *ratios_3, *ratios_4]
    # file header with all variable names
    hdr = ['id', 'stage', 't_tot', 't_mean', 'period', 'p_err', 'n_param_prew', 'bic_prew', 'noise_level_prew',
           't_0', 't_1', 't_2', 't_1_1', 't_1_2', 't_2_1', 't_2_2', 't_b_1_1', 't_b_1_2', 't_b_2_1', 't_b_2_2',
           'depth_1', 'depth_2', 't_1_err', 't_2_err', 't_1_1_err', 't_1_2_err', 't_2_1_err', 't_2_2_err',
           'd_1_err', 'd_2_err', 'p_t_corr', 'n_param_cubics', 'bic_cubics', 'noise_level_cubics',
           'e_form', 'e_low', 'e_upp', 'e_sig', 'w_form', 'w_low', 'w_upp', 'w_sig', 'i_form', 'i_low', 'i_upp',
           'r_sum_form', 'r_sum_low', 'r_sum_upp', 'r_sum_sig', 'r_rat_form',
           'r_rat_low', 'r_rat_upp', 'sb_rat_form', 'sb_rat_low', 'sb_rat_upp',
           'e_fit_h', 'w_fit_h', 'i_fit_h', 'r_sum_fit_h', 'r_rat_fit_h', 'sb_rat_fit_h',
           'e_fit', 'w_fit', 'i_fit', 'r_sum_fit', 'r_rat_fit', 'sb_rat_fit',
           'n_param_fit', 'bic_fit', 'noise_level_fit',
           'total_freqs', 'passed_sigma', 'passed_snr', 'passed_both', 'passed_harmonics',
           'std_1', 'std_2', 'std_3', 'std_4', 'ratio_1_1', 'ratio_1_2', 'ratio_2_1', 'ratio_2_2',
           'ratio_3_1', 'ratio_3_2', 'ratio_4_1', 'ratio_4_2']
    # descriptions of all variables
    desc = ['target identifier', 'furthest stage the analysis reached', 'total time base of observations in days',
            'time series mean time reference point', 'orbital period in days', 'error in the orbital period',
            'number of free parameters after the prewhitening phase', 'BIC after the prewhitening phase',
            'noise level after the prewhitening phase', 'time of primary minimum with respect to the mean time',
            'time of primary minimum minus t_0', 'time of secondary minimum minus t_0',
            'time of primary first contact minus t_0', 'time of primary last contact minus t_0',
            'time of secondary first contact minus t_0', 'time of secondary last contact minus t_0',
            'start of (flat) eclipse bottom left of primary minimum',
            'end of (flat) eclipse bottom right of primary minimum',
            'start of (flat) eclipse bottom left of secondary minimum',
            'end of (flat) eclipse bottom right of secondary minimum',
            'depth of primary minimum', 'depth of secondary minimum',
            'error in time of primary minimum (t_1)', 'error in time of secondary minimum (t_2)',
            'error in time of primary first contact (t_1_1)', 'error in time of primary last contact (t_1_2)',
            'error in time of secondary first contact (t_2_1)', 'error in time of secondary last contact (t_2_2)',
            'error in depth of primary minimum', 'error in depth of secondary minimum',
            'correlation between period and t_zero', 'number of free parameters after the eclipse timing phase',
            'BIC after the eclipse timing phase', 'noise level after the eclipse timing phase',
            'eccentricity from timing formulae',
            'upper error estimate in e', 'lower error estimate in e', 'formal uncorrelated error in e',
            'argument of periastron (radians) from timing formulae',
            'upper error estimate in w', 'lower error estimate in w', 'formal uncorrelated error in w',
            'inclination (radians) from timing formulae',
            'upper error estimate in i', 'lower error estimate in i',
            'sum of radii divided by the semi-major axis of the relative orbit from timing formulae',
            'upper error estimate in r_sum', 'lower error estimate in r_sum',
            'formal uncorrelated error in r_sum',
            'upper error estimate in r_rat', 'lower error estimate in r_rat',
            'radius ratio r2/r1 from timing formulae',
            'surface brightness ratio sb2/sb1 from timing formulae',
            'upper error estimate in sb_rat', 'lower error estimate in sb_rat',
            'eccentricity fit 1 (harmonics)', 'argument of periastron fit 1 (harmonics)',
            'orbital inclination i (radians) fit 1 (harmonics)', 'sum of fractional radii (r1+r2)/a fit 1 (harmonics)',
            'radius ratio r2/r1 fit 1 (harmonics)', 'surface brightness ratio sb2/sb1 fit 1 (harmonics)',
            'eccentricity fit 1 (full)', 'argument of periastron fit 1 (full)',
            'inclination (radians) fit 1 (full)', 'sum of fractional radii fit 1 (full)',
            'radius ratio fit 1 (full)', 'surface brightness ratio fit 1 (full)',
            'number of parameters fit 1 (full)', 'BIC fit 1 (full)', 'noise level fit 1 (full)',
            'total number of frequencies', 'number of frequencies that passed the sigma test',
            'number of frequencies that passed the S/R test', 'number of frequencies that passed both tests',
            'number of hramonics that passed both tests',
            'Standard deviation of the residuals of the linear+sinusoid+eclipse model',
            'Standard deviation of the residuals of the linear+eclipse model',
            'Standard deviation of the residuals of the linear+harmonic 1 and 2+eclipse model',
            'Standard deviation of the residuals of the linear+non-harmonic sinusoid+eclipse model',
            'Ratio of the first eclipse depth to std_1', 'Ratio of the second eclipse depth to std_1',
            'Ratio of the first eclipse depth to std_2', 'Ratio of the second eclipse depth to std_2',
            'Ratio of the first eclipse depth to std_3', 'Ratio of the second eclipse depth to std_3',
            'Ratio of the first eclipse depth to std_4', 'Ratio of the second eclipse depth to std_4']
    # record the stage where the analysis finished
    stage = ''
    files_in_dir = []
    for root, dirs, files in os.walk(save_dir):
        for file_i in files:
            files_in_dir.append(os.path.join(root, file_i))
    for i in range(19, 0, -1):
        match_b = [fnmatch.fnmatch(file_i, f'*_analysis_{i}b*') for file_i in files_in_dir]
        if np.any(match_b):
            stage = str(i) + 'b'  # b variant
            break
        else:
            match_a = [fnmatch.fnmatch(file_i, f'*_analysis_{i}*') for file_i in files_in_dir]
            if np.any(match_a):
                stage = str(i)
                break
    stage = stage.rjust(3)  # make the string 3 long
    # compile all results
    obs_par = np.concatenate(([target_id], [stage], [t_tot], [t_mean], prew_par, timings_par, form_par, fit_par_init,
                              fit_par, freqs_par, level_par)).reshape((-1, 1))
    data = np.column_stack((hdr, obs_par, desc))
    file_hdr = f'{target_id}, {data_id}\nname, value'  # the actual header used for numpy savetxt
    save_name = os.path.join(save_dir, f'{target_id}_analysis_summary.csv')
    np.savetxt(save_name, data, delimiter=',', fmt='%s', header=file_hdr)
    return None


def sequential_plotting(times, signal, i_sectors, target_id, load_dir, save_dir=None, show=True):
    """Due to plotting not working under multiprocessing this function is
    made to make plots after running the analysis in parallel.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    target_id: int
        The TESS Input Catalog number for later reference
        Use any number (or even str) as reference if not available.
    load_dir: str
        Path to a directory for loading analysis results.
    save_dir: str, None
        Path to a directory for save the plots.
    show: bool
        Whether to show the plots or not.
    
    Returns
    -------
    None
    """
    load_dir = os.path.join(load_dir, f'{target_id}_analysis')  # add subdir
    save_dir = os.path.join(save_dir, f'{target_id}_analysis')  # add subdir
    # open all the data
    file_name = os.path.join(load_dir, f'{target_id}_analysis_1.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        sin_mean, _, _, _, _, _, _, _, _, _, _, _, _, _ = results
        const_1, slope_1, f_n_1, a_n_1, ph_n_1 = sin_mean
        model_linear = tsf.linear_curve(times, const_1, slope_1, i_sectors)
        model_sinusoid = tsf.sum_sines(times, f_n_1, a_n_1, ph_n_1)
        model_1 = model_linear + model_sinusoid
    else:
        const_1, slope_1, f_n_1, a_n_1, ph_n_1 = np.array([[], [], [], [], []])
        model_1 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'{target_id}_analysis_2.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        sin_mean, _, _, _, _, _, _, _, _, _, _, _, _, _ = results
        const_2, slope_2, f_n_2, a_n_2, ph_n_2 = sin_mean
        model_linear = tsf.linear_curve(times, const_2, slope_2, i_sectors)
        model_sinusoid = tsf.sum_sines(times, f_n_2, a_n_2, ph_n_2)
        model_2 = model_linear + model_sinusoid
    else:
        const_2, slope_2, f_n_2, a_n_2, ph_n_2 = np.array([[], [], [], [], []])
        model_2 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'{target_id}_analysis_3.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        sin_mean, _, _, _, ecl_mean, ecl_err, _, _, _, _, _, stats, _, _ = results
        const_3, slope_3, f_n_3, a_n_3, ph_n_3 = sin_mean
        p_orb_3, _, _, _, _, _, _, _, _, _, _, _ = ecl_mean
        p_err_3, _, _, _, _, _, _, _, _, _, _, _ = ecl_err
        model_linear = tsf.linear_curve(times, const_3, slope_3, i_sectors)
        model_sinusoid = tsf.sum_sines(times, f_n_3, a_n_3, ph_n_3)
        model_3 = model_linear + model_sinusoid
    else:
        const_3, slope_3, f_n_3, a_n_3, ph_n_3 = np.array([[], [], [], [], []])
        p_orb_3, p_err_3 = 0, 0
        model_3 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'{target_id}_analysis_4.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        sin_mean, _, _, _, _, _, _, _, _, _, _, stats, _, _ = results
        const_4, slope_4, f_n_4, a_n_4, ph_n_4 = sin_mean
        model_linear = tsf.linear_curve(times, const_4, slope_4, i_sectors)
        model_sinusoid = tsf.sum_sines(times, f_n_4, a_n_4, ph_n_4)
        model_4 = model_linear + model_sinusoid
    else:
        const_4, slope_4, f_n_4, a_n_4, ph_n_4 = np.array([[], [], [], [], []])
        model_4 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'{target_id}_analysis_5.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        sin_mean, _, _, _, ecl_mean, ecl_err, _, _, _, _, _, stats, _, _ = results
        const_5, slope_5, f_n_5, a_n_5, ph_n_5 = sin_mean
        p_orb_5, _, _, _, _, _, _, _, _, _, _, _ = ecl_mean
        p_err_5, _, _, _, _, _, _, _, _, _, _, _ = ecl_err
        t_tot, t_mean, t_mean_s, t_int, n_param_5, bic_5, noise_level_5 = stats
        model_linear = tsf.linear_curve(times, const_5, slope_5, i_sectors)
        model_sinusoid = tsf.sum_sines(times, f_n_5, a_n_5, ph_n_5)
        model_5 = model_linear + model_sinusoid
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_5, p_orb_5, f_tol=1e-9)
        f_h_5, a_h_5, ph_h_5 = f_n_5[harmonics], a_n_5[harmonics], ph_n_5[harmonics]
        inf_data_5 = read_inference_data(file_name)
    else:
        const_5, slope_5, f_n_5, a_n_5, ph_n_5 = np.array([[], [], [], [], []])
        p_orb_5, p_err_5 = 0, 0
        n_param_5, bic_5, noise_level_5 = 0, 0, 0
        model_5 = np.zeros(len(times))
        f_h_5, a_h_5, ph_h_5 = np.array([[], [], []])
    # stick together for sending to plot function
    models = [model_1, model_2, model_3, model_4, model_5]
    p_orb_i = [0, 0, p_orb_3, p_orb_3, p_orb_5]
    p_err_i = [0, 0, p_err_3, p_err_3, p_err_5]
    f_n_i = [f_n_1, f_n_2, f_n_3, f_n_4, f_n_5]
    a_n_i = [a_n_1, a_n_2, a_n_3, a_n_4, a_n_5]
    # get the low harmonics
    if (len(f_n_5) > 0):
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_5, p_orb_5, f_tol=1e-9)
        if (len(harmonics) > 0):
            low_h = (harmonic_n <= 20)  # restrict harmonics to avoid interference of ooe signal
            f_hl_5, a_hl_5, ph_hl_5 = f_n_5[harmonics[low_h]], a_n_5[harmonics[low_h]], ph_n_5[harmonics[low_h]]
    # load initial timing results (6)
    file_name = os.path.join(load_dir, f'{target_id}_analysis_6.hdf5')
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_2 = file_name.replace(fn_ext, '_ecl_indices.csv')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        _, _, _, _, ecl_mean, _, _, timings, timings_err, _, _, _, _, _ = results
        _, t_zero_6, _, _, _, _, _, _, _, _, _, _ = ecl_mean
        timings_6, depths_6 = timings[:10], timings[10:]
        timings_err_6, depths_err_6 = timings_err[:6], timings_err[6:]
        ecl_indices_6 = read_results_ecl_indices(file_name)
    elif os.path.isfile(file_name_2):
        ecl_indices_6 = read_results_ecl_indices(file_name)
    if os.path.isfile(file_name) | os.path.isfile(file_name_2):
        ecl_indices_6 = np.atleast_2d(ecl_indices_6)
        if (len(ecl_indices_6) == 0):
            del ecl_indices_6  # delete the empty array to not do the plot
    # load timing results (9)
    file_name = os.path.join(load_dir, f'{target_id}_analysis_9.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        sin_mean, _, _, _, ecl_mean, _, _, timings, timings_err, _, _, stats, _, _ = results
        const_9, slope_9, f_n_9, a_n_9, ph_n_9 = sin_mean
        _, t_zero_9, _, _, _, _, _, _, _, _, _, _ = ecl_mean
        timings_9, depths_9 = timings[:10], timings[10:]
        timings_err_9, depths_err_9 = timings_err[:6], timings_err[6:]
        t_tot, t_mean, t_mean_s, t_int, n_param_9, bic_9, noise_level_9 = stats
        _, _, p_t_corr = af.linear_regression_uncertainty(p_orb_5, t_tot, sigma_t=t_int)
    # load parameter results from formulae (10)
    file_name = os.path.join(load_dir, f'{target_id}_analysis_10.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        _, _, _, _, ecl_mean, ecl_err, ecl_hdi, _, _, _, _, _, _, _ = results
        _, _, ecosw_10, esinw_10, cosi_10, phi_0_10, r_rat_10, sb_rat_10, e_10, w_10, i_10, r_sum_10 = ecl_mean
        _, _, ecosw_err_10, esinw_err_10, cosi_err_10, phi_0_err_10, r_rat_err_10, sb_rat_err_10 = ecl_hdi[:8]
        e_err_10, w_err_10, i_err_10, r_sum_err_10 = ecl_hdi[8:]
        par_init_10 = [e_10, w_10, i_10, r_sum_10, r_rat_10, sb_rat_10]
        dists_in_10, dists_out_10 = read_results_dists(file_name)
        # intervals_w #? for when the interval is disjoint
    # load parameter results from initial fit (11)
    file_name = os.path.join(load_dir, f'{target_id}_analysis_11.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        _, _, _, _, ecl_mean, _, _, _, _, _, _, _, _, _ = results
        _, _, ecosw_11, esinw_11, cosi_11, phi_0_11, r_rat_11, sb_rat_11, e_11, w_11, i_11, r_sum_11 = ecl_mean
        par_opt_11 = [e_11, w_11, i_11, r_sum_11, r_rat_11, sb_rat_11]
    # load parameter results from full fit (13)
    file_name = os.path.join(load_dir, f'{target_id}_analysis_13.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        sin_mean, _, _, _, ecl_mean, _, _, _, _, _, _, stats, _, _ = results
        const_13, slope_13, f_n_13, a_n_13, ph_n_13 = sin_mean
        _, t_zero_13, ecosw_13, esinw_13, cosi_13, phi_0_13, r_rat_13, sb_rat_13, e_13, w_13, i_13, r_sum_13 = ecl_mean
        t_tot, t_mean, t_mean_s, t_int, n_param_13, bic_13, noise_level_13 = stats
        par_opt_13 = np.array([e_13, w_13, i_13, r_sum_13, r_rat_13, sb_rat_13])
        inf_data_13 = read_inference_data(file_name)
    # include n_freqs/n_freqs_passed (14)
    file_name = os.path.join(load_dir, f'{target_id}_analysis_14.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        _, _, _, sin_select, _, _, _, _, _, _, _, _, _, _ = results
        passed_sigma_14, passed_snr_14, passed_b_14, passed_h_14 = sin_select
    # frequency_analysis
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_pd_full.png')
        else:
            file_name = None
        vis.plot_pd_full_output(times, signal, models, p_orb_i, p_err_i, f_n_i, a_n_i, i_sectors,
                                save_file=file_name, show=show)
        if np.any([len(fs) != 0 for fs in f_n_i]):
            plot_nr = np.arange(1, len(f_n_i) + 1)[[len(fs) != 0 for fs in f_n_i]][-1]
            plot_data = [eval(f'const_{plot_nr}'), eval(f'slope_{plot_nr}'),
                         eval(f'f_n_{plot_nr}'), eval(f'a_n_{plot_nr}'), eval(f'ph_n_{plot_nr}')]
            if save_dir is not None:
                file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_lc_sinusoids_{plot_nr}.png')
            else:
                file_name = None
            vis.plot_lc_sinusoids(times, signal, *plot_data, i_sectors, save_file=file_name, show=show)
            if save_dir is not None:
                file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_pd_output_{plot_nr}.png')
            else:
                file_name = None
            plot_data = [p_orb_i[plot_nr - 1], p_err_i[plot_nr - 1]] + plot_data
            vis.plot_pd_single_output(times, signal, *plot_data, i_sectors, annotate=False,
                                      save_file=file_name, show=show)
            if save_dir is not None:
                file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_lc_harmonics_{plot_nr}.png')
            else:
                file_name = None
            vis.plot_lc_harmonics(times, signal, *plot_data, i_sectors, save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    except ValueError:
        pass  # no frequencies?
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_mcmc.png')
        else:
            file_name = None
        vis.plot_pair_harmonics(inf_data_5, p_orb_5, const_5, slope_5, f_h_5, a_h_5, ph_h_5,
                                save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    # eclipse_analysis
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_eclipse_analysis_derivatives_lh.png')
        else:
            file_name = None
        vis.plot_lc_derivatives(p_orb_5, f_h_5, a_h_5, ph_h_5, f_hl_5, a_hl_5, ph_hl_5, ecl_indices_6,
                                save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_eclipse_analysis_initial_timings_lh.png')
        else:
            file_name = None
        vis.plot_lc_eclipse_timestamps(times, signal, p_orb_5, t_zero_6, timings_6, depths_6, timings_err_6,
                                       depths_err_6, const_5, slope_5, f_n_5, a_n_5, ph_n_5, f_hl_5, a_hl_5,
                                       ph_hl_5, i_sectors, save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_eclipse_analysis_empirical_timings.png')
        else:
            file_name = None
        vis.plot_lc_empirical_model(times, signal, p_orb_5, t_zero_6, timings_6, depths_6, const_9, slope_9,
                                    f_n_9, a_n_9, ph_n_9, t_zero_9, timings_9, depths_9,
                                    timings_err_9, depths_err_9, i_sectors, save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_eclipse_analysis_timings_elements.png')
        else:
            file_name = None
        vis.plot_corner_eclipse_parameters(p_orb_5, timings_9, depths_9, *dists_in_10, e_10, w_10, i_10, r_sum_10,
                                           r_rat_10, sb_rat_10, *dists_out_10, save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    except ValueError:
        pass  # no dynamic range for some variable
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_eclipse_analysis_lc_physical_initial.png')
        else:
            file_name = None
        vis.plot_lc_light_curve_fit(times, signal, p_orb_5, t_zero_9, timings_9, const_5, slope_5,
                                    f_n_5, a_n_5, ph_n_5, par_init_10, par_opt_11, i_sectors,
                                    save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_eclipse_analysis_timings_elements_with_fit.png')
        else:
            file_name = None
        vis.plot_corner_lc_fit_pars(par_init_10, par_opt_11, dists_out_10, save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    except ValueError:
        pass  # no dynamic range for some variable
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_eclipse_analysis_lc_physical_eclipse.png')
        else:
            file_name = None
        vis.plot_lc_disentangled_freqs(times, signal, p_orb_5, t_zero_13, const_13, slope_13, f_n_13, a_n_13,
                                       ph_n_13, i_sectors, passed_b_14, par_opt_13, model='simple',
                                       save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_eclipse_analysis_lc_physical_eclipse_h.png')
        else:
            file_name = None
        vis.plot_lc_disentangled_freqs_h(times, signal, p_orb_5, t_zero_13, timings_9, const_13, slope_13,
                                         f_n_13, a_n_13, ph_n_13, i_sectors, passed_b_14, passed_h_14,
                                         par_opt_13, model='simple', save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_eclipse_analysis_pd_leftover_sinusoid.png')
        else:
            file_name = None
        vis.plot_pd_disentangled_freqs(times, signal, p_orb_5, t_zero_13, noise_level_5, const_13, slope_13,
                                       f_n_13, a_n_13, ph_n_13, passed_b_14, par_opt_13, i_sectors,
                                       model='simple', save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_eclipse_analysis_mcmc.png')
        else:
            file_name = None
        vis.plot_pair_eclipse(inf_data_13, t_zero_13, ecosw_13, esinw_13, cosi_13, phi_0_13, r_rat_13, sb_rat_13,
                              const_13, slope_13, f_n_13, a_n_13, ph_n_13, save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    # pulsation_analysis
    return None
