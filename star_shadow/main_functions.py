"""STAR SHADOW
Satellite Time series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This Python module contains the main functions that link together all functionality.

Code written by: Luc IJspeert
"""
import os
import time
import logging
import numpy as np
import functools as fct
import multiprocessing as mp

from . import timeseries_functions as tsf
from . import timeseries_fitting as tsfit
from . import mcmc_functions as mcf
from . import analysis_functions as af
from . import utility as ut


# initialize logger
logger = logging.getLogger(__name__)


def iterative_prewhitening(times, signal, signal_err, i_sectors, t_stats, file_name, data_id='none', overwrite=False,
                           verbose=False):
    """Iterative prewhitening of the input signal in the form of
    sine waves and a piece-wise linear curve

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    
    Notes
    -----
    After extraction, a final check is done to see whether some
    groups of frequencies are better replaced by one frequency.
    """
    t_a = time.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_parameters_hdf5(file_name, verbose=verbose)
        const, slope, f_n, a_n, ph_n = results['sin_mean']
        return const, slope, f_n, a_n, ph_n
    
    if verbose:
        print(f'Looking for frequencies')
    # extract all frequencies with the iterative scheme
    out_a = tsf.extract_sinusoids(times, signal, signal_err, i_sectors, verbose=verbose)
    # remove any frequencies that end up not making the statistical cut
    out_b = tsf.reduce_frequencies(times, signal, signal_err, 0, *out_a, i_sectors, verbose=verbose)
    const, slope, f_n, a_n, ph_n = out_b
    # main function done, do the rest for this step
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    n_param = 2 * len(const) + 3 * len(f_n)
    bic = tsf.calc_bic(resid / signal_err, n_param)
    noise_level = np.std(resid)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, a_n, i_sectors)
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    stats = (*t_stats, n_param, bic, noise_level)
    desc = 'Frequency extraction results.'
    ut.save_parameters_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, stats=stats, i_sectors=i_sectors,
                            description=desc, data_id=data_id)
    # print some useful info
    t_b = time.time()
    if verbose:
        print(f'\033[1;32;48mFrequency extraction complete.\033[0m')
        print(f'\033[0;32;48m{len(f_n)} frequencies, {n_param} free parameters. '
              f'BIC: {bic:1.2f}, time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return const, slope, f_n, a_n, ph_n


def optimise_sinusoid(times, signal, signal_err, const, slope, f_n, a_n, ph_n, i_sectors, t_stats, file_name,
                      method='sampler', data_id='none', overwrite=False, verbose=False):
    """Optimise the parameters of the sinusoid and linear model

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    method: str
        Method of optimization. Can be 'sampler' or 'fitter'.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    """
    t_a = time.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_parameters_hdf5(file_name, verbose=verbose)
        const, slope, f_n, a_n, ph_n = results['sin_mean']
        return const, slope, f_n, a_n, ph_n
    
    if verbose:
        print(f'Starting multi-sinusoid NL-LS optimisation.')
    t_tot, t_mean, t_mean_s, t_int = t_stats
    freq_res = 1.5 / t_tot  # Rayleigh criterion
    # use the chosen optimisation method
    inf_data, par_mean, par_hdi = None, None, None
    if method == 'fitter':
        par_mean = tsfit.fit_multi_sinusoid_per_group(times, signal, signal_err, const, slope, f_n, a_n, ph_n,
                                                       i_sectors, verbose=verbose)
    else:
        # make model including everything to calculate noise level
        model_lin = tsf.linear_curve(times, const, slope, i_sectors)
        model_sin = tsf.sum_sines(times, f_n, a_n, ph_n)
        resid = signal - (model_lin + model_sin)
        noise_level = np.std(resid)
        # formal linear and sinusoid parameter errors
        c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, a_n, i_sectors)
        # do not include those frequencies that have too big uncertainty
        include = (ph_n_err < 1 / np.sqrt(6))  # circular distribution for ph_n cannot handle these
        f_n, a_n, ph_n = f_n[include], a_n[include], ph_n[include]
        f_n_err, a_n_err, ph_n_err = f_n_err[include], a_n_err[include], ph_n_err[include]
        # Monte Carlo sampling of the model
        output = mcf.sample_sinusoid(times, signal, const, slope, f_n, a_n, ph_n, c_err, sl_err, f_n_err, a_n_err,
                                     ph_n_err, noise_level, i_sectors, verbose=verbose)
        inf_data, par_mean, par_hdi = output
    const, slope, f_n, a_n, ph_n = par_mean
    # main function done, do the rest for this step
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    n_param = 2 * len(const) + 3 * len(f_n)
    bic = tsf.calc_bic(resid / signal_err, n_param)
    noise_level = np.std(resid)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, a_n, i_sectors)
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    sin_hdi = par_hdi
    stats = (t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level)
    desc = 'Multi-sinusoid NL-LS optimisation results.'
    ut.save_parameters_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, sin_hdi=sin_hdi, stats=stats,
                            i_sectors=i_sectors, description=desc, data_id=data_id)
    ut.save_inference_data(file_name, inf_data)
    # print some useful info
    t_b = time.time()
    if verbose:
        print(f'\033[1;32;48mOptimisation complete.\033[0m')
        print(f'\033[0;32;48m{len(f_n)} frequencies, {n_param} free parameters. '
              f'BIC: {bic:1.2f}, time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return const, slope, f_n, a_n, ph_n


def find_orbital_period(times, signal, f_n, t_tot):
    """Find the most likely eclipse period from a sinusoid model
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    t_tot: float
        Total time base of observations
    
    Returns
    -------
    p_orb: float
        Orbital period of the eclipsing binary in days
    
    Notes
    -----
    Uses a combination of phase dispersion minimisation and
    Lomb-Scargle periodogram (see Saha & Vivas 2017), and some
    refining steps to get the best period.
    
    Period precision is 0.00001
    """
    freq_res = 1.5 / t_tot  # Rayleigh criterion
    f_nyquist = 1 / (2 * np.min(np.diff(times)))
    # first to get a global minimum do combined PDM and LS, at select frequencies
    periods, phase_disp = tsf.phase_dispersion_minimisation(times, signal, f_n, local=False)
    ampls = tsf.scargle_ampl(times, signal, 1 / periods)
    psi_measure = ampls / phase_disp
    # also check the number of harmonics at each period and include into best f
    n_harm, completeness, distance = af.harmonic_series_length(1 / periods, f_n, freq_res, f_nyquist)
    psi_h_measure = psi_measure * n_harm * completeness
    # select the best period, refine it and check double P
    base_p = periods[np.argmax(psi_h_measure)]
    # refine by using a dense sampling
    f_refine = np.arange(0.99 / base_p, 1.01 / base_p, 0.00001 / base_p)
    n_harm_r, completeness_r, distance_r = af.harmonic_series_length(f_refine, f_n, freq_res, f_nyquist)
    h_measure = n_harm_r * completeness_r  # compute h_measure for constraining a domain
    mask_peak = (h_measure > np.max(h_measure) / 1.5)  # constrain the domain of the search
    i_min_dist = np.argmin(distance_r[mask_peak])
    p_orb = 1 / f_refine[mask_peak][i_min_dist]
    # check twice the period as well
    base_p2 = base_p * 2
    # refine by using a dense sampling
    f_refine_2 = np.arange(0.99 / base_p2, 1.01 / base_p2, 0.00001 / base_p2)
    n_harm_r_2, completeness_r_2, distance_r_2 = af.harmonic_series_length(f_refine_2, f_n, freq_res, f_nyquist)
    h_measure_2 = n_harm_r_2 * completeness_r_2 # compute h_measure for constraining a domain
    mask_peak_2 = (h_measure_2 > np.max(h_measure_2) / 1.5)  # constrain the domain of the search
    i_min_dist_2 = np.argmin(distance_r_2[mask_peak_2])
    p_orb_2 = 1 / f_refine_2[mask_peak_2][i_min_dist_2]
    # compare the length and completeness to decide, using a threshold
    minimal_frac = 1.1  # empirically determined threshold
    frac_double = h_measure_2[mask_peak_2][i_min_dist_2] / h_measure[mask_peak][i_min_dist]
    if (frac_double > minimal_frac):
        p_orb = p_orb_2
    return p_orb


def couple_harmonics(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, t_stats, file_name,
                     data_id='none', overwrite=False, verbose=False):
    """Find the orbital period and couple harmonic frequencies to the orbital period

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    p_orb: float
        Orbital period of the eclipsing binary in days
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    
    Notes
    -----
    Performs a global period search, if the period is unknown.
    
    Removes theoretical harmonic candidate frequencies within the frequency
    resolution, then extracts a single harmonic at the theoretical location.
    
    Removes any frequencies that end up not making the statistical cut.
    """
    t_a = time.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_parameters_hdf5(file_name, verbose=verbose)
        const, slope, f_n, a_n, ph_n = results['sin_mean']
        p_orb, _ = results['ephem']
        return p_orb, const, slope, f_n, a_n, ph_n
    
    if verbose:
        print(f'Coupling the harmonic frequencies to the orbital frequency.')
    t_tot, t_mean, t_mean_s, t_int = t_stats
    # we use the input p_orb at face value if given
    if (p_orb == 0):
        p_orb = find_orbital_period(times, signal, f_n, t_tot)
    # if time series too short, or no harmonics found, log and warn and maybe cut off the analysis
    freq_res = 1.5 / t_tot  # Rayleigh criterion
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=freq_res / 2)
    if (t_tot / p_orb < 1.1):
        out_a = const, slope, f_n, a_n, ph_n  # return previous results
    elif (len(harmonics) < 2):
        out_a = const, slope, f_n, a_n, ph_n  # return previous results
    else:
        # couple the harmonics to the period. likely removes more frequencies that need re-extracting
        out_a = tsf.fix_harmonic_frequency(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors,
                                           verbose=verbose)
    # remove any frequencies that end up not making the statistical cut
    out_b = tsf.reduce_frequencies(times, signal, signal_err, p_orb, *out_a, i_sectors, verbose=verbose)
    const, slope, f_n, a_n, ph_n = out_b
    # main function done, do the rest for this step
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_param = 2 * len(const) + 1 + 2 * len(harmonics) + 3 * (len(f_n) - len(harmonics))
    bic = tsf.calc_bic(resid / signal_err, n_param)
    noise_level = np.std(resid)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, a_n, i_sectors)
    p_err, _, _ = af.linear_regression_uncertainty(p_orb, t_tot, sigma_t=t_int / 2)
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    ephem = np.array([p_orb, -1])
    ephem_err = np.array([p_err, -1])
    stats = [t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level]
    desc = 'Harmonic frequencies coupled.'
    ut.save_parameters_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, ephem=ephem, ephem_err=ephem_err,
                            stats=stats, i_sectors=i_sectors, description=desc, data_id=data_id)
    # print some useful info
    t_b = time.time()
    if verbose:
        print(f'\033[1;32;48mOrbital harmonic frequencies coupled. Period: {p_orb:2.4}\033[0m')
        print(f'\033[0;32;48m{len(f_n)} frequencies, {n_param} free parameters. '
              f'BIC: {bic:1.2f}, time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return p_orb, const, slope, f_n, a_n, ph_n


def add_sinusoids(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, t_stats, file_name,
                  data_id='none', overwrite=False, verbose=False):
    """Find and add more (harmonic and non-harmonic) frequencies if possible

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    p_orb: float
        Orbital period of the eclipsing binary in days
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves

    Notes
    -----
    First looks for additional harmonic frequencies at the integer multiples
    of the orbital frequency.

    Then looks for any additional non-harmonic frequencies taking into account
    the existing harmonics.

    Finally, removes any frequencies that end up not making the statistical cut.
    """
    t_a = time.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_parameters_hdf5(file_name, verbose=verbose)
        const, slope, f_n, a_n, ph_n = results['sin_mean']
        p_orb, _ = results['ephem']
        return p_orb, const, slope, f_n, a_n, ph_n
    
    if verbose:
        print(f'Looking for additional frequencies.')
    t_tot, t_mean, t_mean_s, t_int = t_stats
    n_f_init = len(f_n)
    # start by looking for more harmonics
    out_a = tsf.extract_harmonics(times, signal, signal_err, p_orb, i_sectors, f_n, a_n, ph_n, verbose=verbose)
    # look for any additional non-harmonics with the iterative scheme
    out_b = tsf.extract_sinusoids(times, signal, signal_err, i_sectors, p_orb, *out_a[2:], verbose=verbose)
    # remove any frequencies that end up not making the statistical cut
    out_c = tsf.reduce_frequencies(times, signal, signal_err, p_orb, *out_b, i_sectors, verbose=verbose)
    const, slope, f_n, a_n, ph_n = out_c
    # main function done, do the rest for this step
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_param = 2 * len(const) + 1 + 2 * len(harmonics) + 3 * (len(f_n) - len(harmonics))
    bic = tsf.calc_bic(resid / signal_err, n_param)
    noise_level = np.std(resid)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, a_n, i_sectors)
    p_err, _, _ = af.linear_regression_uncertainty(p_orb, t_tot, sigma_t=t_int / 2)
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    ephem = np.array([p_orb, -1])
    ephem_err = np.array([p_err, -1])
    stats = [t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level]
    desc = 'Additional non-harmonic extraction.'
    ut.save_parameters_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, ephem=ephem, ephem_err=ephem_err,
                            stats=stats, i_sectors=i_sectors, description=desc, data_id=data_id)
    # print some useful info
    t_b = time.time()
    if verbose:
        print(f'\033[1;32;48m{len(f_n) - n_f_init} additional frequencies added.\033[0m')
        print(f'\033[0;32;48m{len(f_n)} frequencies, {n_param} free parameters. '
              f'BIC: {bic:1.2f}, time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return p_orb, const, slope, f_n, a_n, ph_n


def optimise_sinusoid_h(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, t_stats, file_name,
                        method='sampler', data_id='none', overwrite=False, verbose=False):
    """Optimise the parameters of the sinusoid and linear model with coupled harmonics

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    method: str
        Method of optimization. Can be 'sampler' or 'fitter'.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    p_orb: float
        Orbital period of the eclipsing binary in days
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    """
    t_a = time.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_parameters_hdf5(file_name, verbose=verbose)
        const, slope, f_n, a_n, ph_n = results['sin_mean']
        p_orb, _ = results['ephem']
        return p_orb, const, slope, f_n, a_n, ph_n
    
    if verbose:
        print(f'Starting multi-sine NL-LS optimisation with harmonics.')
    t_tot, t_mean, t_mean_s, t_int = t_stats
    # use the chosen optimisation method
    inf_data, par_mean, sin_hdi, ephem_hdi = None, None, None, None
    if method == 'fitter':
        par_mean = tsfit.fit_multi_sinusoid_harmonics_per_group(times, signal, signal_err, p_orb, const, slope,
                                                                f_n, a_n, ph_n, i_sectors, verbose=verbose)
    else:
        # make model including everything to calculate noise level
        model_lin = tsf.linear_curve(times, const, slope, i_sectors)
        model_sin = tsf.sum_sines(times, f_n, a_n, ph_n)
        resid = signal - (model_lin + model_sin)
        noise_level = np.std(resid)
        # formal linear and sinusoid parameter errors
        c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, a_n, i_sectors)
        p_err, _, _ = af.linear_regression_uncertainty(p_orb, t_tot, sigma_t=t_int / 2)
        # do not include those frequencies that have too big uncertainty
        include = (ph_n_err < 1 / np.sqrt(6))  # circular distribution for ph_n cannot handle these
        f_n, a_n, ph_n = f_n[include], a_n[include], ph_n[include]
        f_n_err, a_n_err, ph_n_err = f_n_err[include], a_n_err[include], ph_n_err[include]
        # Monte Carlo sampling of the model
        output = mcf.sample_sinusoid_h(times, signal, p_orb, const, slope, f_n, a_n, ph_n, p_err, c_err, sl_err,
                                       f_n_err, a_n_err, ph_n_err, noise_level, i_sectors, verbose=verbose)
        inf_data, par_mean, par_hdi = output
        sin_hdi = par_hdi[1:]
        ephem_hdi = np.array([par_hdi[0], [-1, -1]])
    p_orb, const, slope, f_n, a_n, ph_n = par_mean
    # main function done, do the rest for this step
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    # calculate number of parameters, BIC and noise level
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_param = 2 * len(const) + 1 + 2 * len(harmonics) + 3 * (len(f_n) - len(harmonics))
    bic = tsf.calc_bic(resid / signal_err, n_param)
    noise_level = np.std(resid)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, a_n, i_sectors)
    p_err, _, _ = af.linear_regression_uncertainty(p_orb, t_tot, sigma_t=t_int / 2)
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    ephem = np.array([p_orb, -1])
    ephem_err = np.array([p_err, -1])
    stats = [t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level]
    desc = 'Multi-sine NL-LS optimisation results with coupled harmonics.'
    ut.save_parameters_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, sin_hdi=sin_hdi, ephem=ephem,
                            ephem_err=ephem_err, ephem_hdi=ephem_hdi, stats=stats, i_sectors=i_sectors,
                            description=desc, data_id=data_id)
    ut.save_inference_data(file_name, inf_data)
    # print some useful info
    t_b = time.time()
    if verbose:
        print(f'\033[1;32;48mOptimisation with fixed harmonics complete. Period: {p_orb:2.4}\033[0m')
        print(f'\033[0;32;48m{len(f_n)} frequencies, {n_param} free parameters. '
              f'BIC: {bic:1.2f}, time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return p_orb, const, slope, f_n, a_n, ph_n


def analyse_frequencies(times, signal, signal_err, i_sectors, p_orb, t_stats, target_id, save_dir, method='sampler',
                        data_id='none', overwrite=False, verbose=False):
    """Recipe for the extraction of sinusoids from EB light curves.

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    p_orb: float
        The orbital period. Set to 0 to search for the best period.
        If the orbital period is known with certainty beforehand, it can
        be provided as initial value and no new period will be searched.
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    target_id: int, str
        The TESS Input Catalog number for later reference.
        Use any number (or string) as reference if not available.
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    method: str
        Method of optimization. Can be 'sampler' or 'fitter'.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    p_orb_i: list[float]
        Orbital period at each stage of the analysis
    const_i: list[numpy.ndarray[float]]
        y-intercepts of a piece-wise linear curve for each stage of the analysis
    slope_i: list[numpy.ndarray[float]]
        slopes of a piece-wise linear curve for each stage of the analysis
    f_n_i: list[numpy.ndarray[float]]
        Frequencies of a number of sine waves for each stage of the analysis
    a_n_i: list[numpy.ndarray[float]]
        Amplitudes of a number of sine waves for each stage of the analysis
    ph_n_i: list[numpy.ndarray[float]]
        Phases of a number of sine waves for each stage of the analysis

    Notes
    -----
    The followed recipe is:
    
    1) Extract all frequencies
        We start by extracting the frequency with the highest amplitude one by one,
        directly from the Lomb-Scargle periodogram until the BIC does not significantly
        improve anymore. Some of these steps involve a final cleanup of the frequencies.
    
    2) First multi-sine NL-LS optimisation
        The sinusoid parameters are simultaneously optimised using Monte Carlo sampling.
    
    3) Measure the orbital period and couple the harmonic frequencies
        Global search done with combined phase dispersion, Lomb-Scargle and length/
        filling factor of the harmonic series in the list of frequencies.
        Then sets the frequencies of the harmonics to their new values, coupling them
        to the orbital period.
        [Note: it is possible to provide a fixed period if it is already well known.
        It will still be included as a free parameter in the final optimisation step]
    
    4) Attempt to extract additional frequencies
        The decreased number of free parameters (2 vs. 3), the BIC, which punishes for free
        parameters, may allow the extraction of more harmonics.
        It is also attempted to extract more frequencies like in step 1 again, now taking
        into account the presence of harmonics.
    
    5) Multi-sine NL-LS optimisation with coupled harmonics
        Optimise all frequencies simultaneously once again, including the orbital period and
        the coupled harmonics.
    """
    t_a = time.time()
    t_tot, t_mean, t_mean_s, t_int = t_stats
    freq_res = 1.5 / t_tot  # Rayleigh criterion
    signal_err = np.max(signal_err) * np.ones(len(times))  # likelihood assumes the same errors
    arg_dict = {'data_id': data_id, 'overwrite': overwrite, 'verbose': verbose}  # these stay the same
    # -------------------------------------------------------
    # --- [1] --- Initial iterative extraction of frequencies
    # -------------------------------------------------------
    file_name_1 = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_1.hdf5')
    out_1 = iterative_prewhitening(times, signal, signal_err, i_sectors, t_stats, file_name_1, **arg_dict)
    const_1, slope_1, f_n_1, a_n_1, ph_n_1 = out_1
    if (len(f_n_1) == 0):
        logger.info('No frequencies found.')
        p_orb_i = [0]
        const_i = [const_1]
        slope_i = [slope_1]
        f_n_i = [f_n_1]
        a_n_i = [a_n_1]
        ph_n_i = [ph_n_1]
        return p_orb_i, const_i, slope_i, f_n_i, a_n_i, ph_n_i
    # ----------------------------------------------------------------
    # --- [2] --- Multi-sinusoid non-linear least-squares optimisation
    # ----------------------------------------------------------------
    file_name_2 = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_2.hdf5')
    out_2 = optimise_sinusoid(times, signal, signal_err, const_1, slope_1, f_n_1, a_n_1, ph_n_1, i_sectors, t_stats,
                              file_name_2, method=method, **arg_dict)
    const_2, slope_2, f_n_2, a_n_2, ph_n_2 = out_2
    # --------------------------------------------------------------------------
    # --- [3] --- Measure the orbital period and couple the harmonic frequencies
    # --------------------------------------------------------------------------
    file_name_3 = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_3.hdf5')
    out_3 = couple_harmonics(times, signal, signal_err, p_orb, const_2, slope_2, f_n_2, a_n_2, ph_n_2, i_sectors,
                             t_stats, file_name_3, **arg_dict)
    p_orb_3, const_3, slope_3, f_n_3, a_n_3, ph_n_3 = out_3
    # save info and exit in the following cases (and log message)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_2, p_orb_3, f_tol=freq_res / 2)
    if (t_tot / p_orb_3 < 1.1):
        logger.info(f'Period over time-base is less than two: {t_tot / p_orb_3}; '
                    f'period (days): {p_orb_3}; time-base (days): {t_tot}')
    elif (len(harmonics) < 2):
        logger.info(f'Not enough harmonics found: {len(harmonics)}; '
                    f'period (days): {p_orb_3}; time-base (days): {t_tot}')
        # return previous results
    elif (t_tot / p_orb_3 < 2):
        logger.info(f'Period over time-base is less than two: {t_tot / p_orb_3}; '
                    f'period (days): {p_orb_3}; time-base (days): {t_tot}')
    if (t_tot / p_orb_3 < 1.1) | (len(harmonics) < 2):
        p_orb_i = [0, 0, p_orb_3]
        const_i = [const_1, const_2, const_2]
        slope_i = [slope_1, slope_2, slope_2]
        f_n_i = [f_n_1, f_n_2, f_n_2]
        a_n_i = [a_n_1, a_n_2, a_n_2]
        ph_n_i = [ph_n_1, ph_n_2, ph_n_2]
        return p_orb_i, const_i, slope_i, f_n_i, a_n_i, ph_n_i
    # -----------------------------------------------------
    # --- [4] --- Attempt to extract additional frequencies
    # -----------------------------------------------------
    file_name_4 = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_4.hdf5')
    out_4 = add_sinusoids(times, signal, signal_err, p_orb_3, const_3, slope_3, f_n_3, a_n_3, ph_n_3, i_sectors,
                          t_stats, file_name_4, **arg_dict)
    p_orb_4, const_4, slope_4, f_n_4, a_n_4, ph_n_4 = out_4  # p_orb_4 == p_orb_3
    # -----------------------------------------------
    # --- [5] --- Optimisation with coupled harmonics
    # -----------------------------------------------
    file_name_5 = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_5.hdf5')
    out_5 = optimise_sinusoid_h(times, signal, signal_err, p_orb_3, const_4, slope_4, f_n_4, a_n_4, ph_n_4, i_sectors,
                                t_stats, file_name_5, method=method, **arg_dict)
    p_orb_5, const_5, slope_5, f_n_5, a_n_5, ph_n_5 = out_5
    # save final freqs and linear curve in ascii format
    ut.convert_hdf5_to_ascii(file_name_5)
    # make lists for output
    p_orb_i = [0, 0, p_orb_3, p_orb_3, p_orb_5]
    const_i = [const_1, const_2, const_3, const_4, const_5]
    slope_i = [slope_1, slope_2, slope_3, slope_4, slope_5]
    f_n_i = [f_n_1, f_n_2, f_n_3, f_n_4, f_n_5]
    a_n_i = [a_n_1, a_n_2, a_n_3, a_n_4, a_n_5]
    ph_n_i = [ph_n_1, ph_n_2, ph_n_3, ph_n_4, ph_n_5]
    # final timing and message
    t_b = time.time()
    logger.info(f'Frequency extraction done. Total time elapsed: {t_b - t_a:1.1f}s.')
    return p_orb_i, const_i, slope_i, f_n_i, a_n_i, ph_n_i


def find_eclipse_timings(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, noise_level, file_name,
                         data_id='none', overwrite=False, verbose=False):
    """Finds the position of the eclipses using the prewhitened orbital harmonics

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    noise_level: float
        The noise level (standard deviation of the residuals)
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    timings: numpy.ndarray[float]
        Eclipse timings: minima, first/last contact points, internal tangency and depths,
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2 t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2
    timings_err: numpy.ndarray[float]
        Error estimates for the eclipse timings and depths,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err,
        t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err, depth_1_err, depth_2_err
    
    Notes
    -----
    First attempts to find the eclipses in the model up to the twentieth harmonic,
    reducing chances of confusion. If nothing is found, however, it will attempt
    again with the full model of harmonics.
    
    Times of eclispe are measured with respect to the mean time
    """
    t_a = time.time()
    # guard for existing file when not overwriting
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_2 = file_name.replace(fn_ext, '_ecl_indices.csv')
    if os.path.isfile(file_name) & os.path.isfile(file_name_2) & (not overwrite):
        results = ut.read_parameters_hdf5(file_name, verbose=verbose)
        timings = results['timings']
        timings_err = results['timings_err']
        return timings, timings_err
    elif (not os.path.isfile(file_name)) & os.path.isfile(file_name_2) & (not overwrite):
        if verbose:
            print('Not enough eclipses found last time (see log)')
        return (None,) * 2
    
    if verbose:
        print(f'Measuring eclipse time points and depths.')
    # find any gaps in phase coverage
    t_gaps = tsf.mark_folded_gaps(times, p_orb, p_orb / 100)
    t_gaps = np.vstack((t_gaps, t_gaps + p_orb))  # duplicate for interval [0, 2p]
    # we use the lowest harmonics
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    low_h = (harmonic_n <= 20)  # restrict harmonics to avoid interference of high frequencies
    f_h, a_h, ph_h = f_n[harmonics], a_n[harmonics], ph_n[harmonics]
    # measure eclipse timings - the deepest eclipse is put first in each measurement
    output_a = af.detect_eclipses(p_orb, f_h, a_h, ph_h, noise_level, t_gaps)
    t_1, t_2, t_contacts, t_tangency, depths, t_i_1_err, t_i_2_err, t_b_i_1_err, t_b_i_2_err, ecl_indices = output_a
    # account for not finding eclipses
    ut.save_results_ecl_indices(file_name, ecl_indices, data_id=data_id)  # always save the eclipse indices
    if np.all([item is None for item in output_a]):
        logger.info(f'No eclipse signatures found above the noise level of {noise_level}')
        # save only indices file
        return (None,) * 2
    elif np.any([item is None for item in output_a]):
        logger.info('No two eclipses found passing the criteria')
        # save only indices file
        return (None,) + (ecl_indices,)
    # error estimates/refinement
    timings = np.array([t_1, t_2, *t_contacts, *t_tangency, *depths])
    output_b = tsf.estimate_timing_errors(times, signal, p_orb, const, slope, f_n, a_n, ph_n, timings,
                                          noise_level, i_sectors, t_i_1_err, t_i_2_err, t_b_i_1_err, t_b_i_2_err)
    p_err, timings_ind_err, timings_err = output_b
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err = timings_err[:6]
    t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err, depth_1_err, depth_2_err = timings_err[6:]
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    ephem = np.array([p_orb, t_1])
    ephem_err = np.array([p_err, t_1_err])
    desc = 'Eclipse timings and depths.'
    ut.save_parameters_hdf5(file_name, sin_mean=sin_mean, ephem=ephem, ephem_err=ephem_err, timings=timings,
                            timings_err=timings_err, timings_indiv_err=timings_ind_err, description=desc,
                            data_id=data_id)
    # print some useful stuff
    t_b = time.time()
    if verbose:
        dur_1 = timings[3] - timings[2]  # t_b_1_2 - t_b_1_1
        dur_2 = timings[5] - timings[4]  # t_b_2_2 - t_b_2_1
        dur_1_err = np.sqrt(timings_err[2]**2 + timings_err[3]**2)
        dur_2_err = np.sqrt(timings_err[4]**2 + timings_err[5]**2)
        dur_b_1 = timings[7] - timings[6]  # t_b_1_2 - t_b_1_1
        dur_b_2 = timings[9] - timings[8]  # t_b_2_2 - t_b_2_1
        dur_b_1_err = np.sqrt(timings_err[6]**2 + timings_err[7]**2)
        dur_b_2_err = np.sqrt(timings_err[8]**2 + timings_err[9]**2)
        # determine decimals to print for two significant figures
        rnd_p_orb = max(ut.decimal_figures(p_err, 2), ut.decimal_figures(p_orb, 2))
        rnd_t_1 = max(ut.decimal_figures(timings_err[0], 2), ut.decimal_figures(timings[0], 2))
        rnd_t_2 = max(ut.decimal_figures(timings_err[1], 2), ut.decimal_figures(timings[1], 2))
        rnd_t_1_1 = max(ut.decimal_figures(timings_err[2], 2), ut.decimal_figures(timings[2], 2))
        rnd_t_1_2 = max(ut.decimal_figures(timings_err[3], 2), ut.decimal_figures(timings[3], 2))
        rnd_t_2_1 = max(ut.decimal_figures(timings_err[4], 2), ut.decimal_figures(timings[4], 2))
        rnd_t_2_2 = max(ut.decimal_figures(timings_err[5], 2), ut.decimal_figures(timings[5], 2))
        rnd_t_b_1_1 = max(ut.decimal_figures(timings_err[6], 2), ut.decimal_figures(timings[6], 2))
        rnd_t_b_1_2 = max(ut.decimal_figures(timings_err[7], 2), ut.decimal_figures(timings[7], 2))
        rnd_t_b_2_1 = max(ut.decimal_figures(timings_err[8], 2), ut.decimal_figures(timings[8], 2))
        rnd_t_b_2_2 = max(ut.decimal_figures(timings_err[9], 2), ut.decimal_figures(timings[9], 2))
        rnd_dur_1 = max(ut.decimal_figures(dur_1_err, 2), ut.decimal_figures(dur_1, 2))
        rnd_dur_2 = max(ut.decimal_figures(dur_2_err, 2), ut.decimal_figures(dur_2, 2))
        rnd_d_1 = max(ut.decimal_figures(timings_err[6], 2), ut.decimal_figures(depths[0], 2))
        rnd_d_2 = max(ut.decimal_figures(timings_err[7], 2), ut.decimal_figures(depths[1], 2))
        rnd_bot_1 = max(ut.decimal_figures(dur_1_err, 2), ut.decimal_figures(dur_b_1, 2))
        rnd_bot_2 = max(ut.decimal_figures(dur_2_err, 2), ut.decimal_figures(dur_b_2, 2))
        print(f'\033[1;32;48mMeasurements of timings and depths:\033[0m')
        print(f'\033[0;32;48mp_orb: {p_orb:.{rnd_p_orb}f} (+-{p_err:.{rnd_t_1}f}), '
              f't_1: {timings[0]:.{rnd_t_1}f} (+-{timings_err[0]:.{rnd_t_1}f}), '
              f't_2: {timings[1]:.{rnd_t_2}f} (+-{timings_err[1]:.{rnd_t_2}f}), \n'
              f't_1_1: {timings[2]:.{rnd_t_1_1}f} (+-{timings_err[2]:.{rnd_t_1_1}f}), '
              f't_1_2: {timings[3]:.{rnd_t_1_2}f} (+-{timings_err[3]:.{rnd_t_1_2}f}), \n'
              f't_2_1: {timings[4]:.{rnd_t_2_1}f} (+-{timings_err[4]:.{rnd_t_2_1}f}), '
              f't_2_2: {timings[5]:.{rnd_t_2_2}f} (+-{timings_err[5]:.{rnd_t_2_2}f}), \n'
              f'duration_1: {dur_1:.{rnd_dur_1}f} (+-{dur_1_err:.{rnd_dur_1}f}), '
              f'duration_2: {dur_2:.{rnd_dur_2}f} (+-{dur_2_err:.{rnd_dur_2}f}). \n'
              f't_b_1_1: {timings[6]:.{rnd_t_b_1_1}f} (+-{timings_err[6]:.{rnd_t_b_1_1}f}), '
              f't_b_1_2: {timings[7]:.{rnd_t_b_1_2}f} (+-{timings_err[7]:.{rnd_t_b_1_2}f}), \n'
              f't_b_2_1: {timings[8]:.{rnd_t_b_2_1}f} (+-{timings_err[8]:.{rnd_t_b_2_1}f}), '
              f't_b_2_2: {timings[9]:.{rnd_t_b_2_2}f} (+-{timings_err[9]:.{rnd_t_b_2_2}f}), \n'
              f'bottom_dur_1: {dur_b_1:.{rnd_bot_1}f} (+-{dur_b_1_err:.{rnd_bot_1}f}), '
              f'bottom_dur_2: {dur_b_2:.{rnd_bot_2}f} (+-{dur_b_2_err:.{rnd_bot_2}f}). \n'
              f'depth_1: {depths[0]:.{rnd_d_1}f} (+-{timings_err[6]:.{rnd_d_1}f}), '
              f'depth_2: {depths[1]:.{rnd_d_2}f} (+-{timings_err[7]:.{rnd_d_2}f}). \n'
              f'Time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return timings, timings_err


def convert_timings_to_elements(p_orb, timings, p_err, timings_err, p_t_corr, file_name, data_id='none',
                                overwrite=False, verbose=False):
    """Obtains orbital elements from the eclipse timings

    Parameters
    ----------
    p_orb: float
        Orbital period of the eclipsing binary in days
    timings: numpy.ndarray[float]
        Eclipse timings: minima, first/last contact points, internal tangency and depths,
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2 t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2
    p_err: float
        Error in the orbital period
    timings_err: numpy.ndarray[float]
        Error estimates for the eclipse timings and depths,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err,
        t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err, depth_1_err, depth_2_err
    p_t_corr: float
        Correlation between period and t_1
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit
    r_sum: float
        Sum of radii in units of the semi-major axis
    r_rat: float
        Radius ratio r_2/r_1
    sb_rat: float
        Surface brightness ratio sb_2/sb_1
    errors: tuple[numpy.ndarray[float]]
        The (non-symmetric) errors for the same parameters as intervals.
        These are computed from the intervals.
    intervals: tuple[numpy.ndarray[float]]
        The HDIs (hdi_prob=0.683) for the parameters:
        e, w, i, phi_0, psi_0, r_sum, r_dif_sma, r_rat,
        sb_rat, e*cos(w), e*sin(w), f_c, f_s
    formal_errors: tuple[float]
        Formal (symmetric) errors in the parameters:
        e, w, phi_0, r_sum, ecosw, esinw, f_c, f_s
    dists_in: tuple[numpy.ndarray[float]]
        Full input distributions for: t_1, t_2,
        tau_1_1, tau_1_2, tau_2_1, tau_2_2, d_1, d_2, bot_1, bot_2
    dists_out: tuple[numpy.ndarray[float]]
        Full output distributions for the same parameters as intervals
    """
    t_a = time.time()
    # guard for existing file when not overwriting
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_2 = file_name.replace(fn_ext, '_dists' + fn_ext)
    if os.path.isfile(file_name) & os.path.isfile(file_name_2) & (not overwrite):
        results = ut.read_parameters_hdf5(file_name, verbose=verbose)
        ecosw, esinw, cosi, phi_0, r_rat, sb_rat, e, w, i, r_sum = results['phys_mean']
        sigma_ecosw, sigma_esinw, _, sigma_phi_0, _, _, sigma_e, sigma_w, _, sigma_r_sum = results['phys_err']
        formal_errors = sigma_e, sigma_w, sigma_phi_0, sigma_r_sum, sigma_ecosw, sigma_esinw
        ecosw_err, esinw_err, cosi_err, phi_0_err, r_rat_err, sb_rat_err = results['phys_hdi'][:6]
        e_err, w_err, i_err, r_sum_err = results['phys_hdi'][6:]
        errors = e_err, w_err, i_err, r_sum_err, r_rat_err, sb_rat_err, ecosw_err, esinw_err, cosi_err, phi_0_err
        dists_in, dists_out = ut.read_results_dists(file_name)
        return e, w, i, r_sum, r_rat, sb_rat, errors, formal_errors, dists_in, dists_out
    
    if verbose:
        print('Determining eclipse parameters and error estimates.')
    # convert to durations
    tau_1_1 = timings[0] - timings[2]  # t_1 - t_1_1
    tau_1_2 = timings[3] - timings[0]  # t_1_2 - t_1
    tau_2_1 = timings[1] - timings[4]  # t_2 - t_2_1
    tau_2_2 = timings[5] - timings[1]  # t_2_2 - t_2
    timings_tau = np.array([timings[0], timings[1], tau_1_1, tau_1_2, tau_2_1, tau_2_2])
    tau_b_1_1 = timings[0] - timings[6]  # t_1 - t_b_1_1
    tau_b_1_2 = timings[7] - timings[0]  # t_b_1_2 - t_1
    tau_b_2_1 = timings[1] - timings[8]  # t_2 - t_b_2_1
    tau_b_2_2 = timings[9] - timings[1]  # t_b_2_2 - t_2
    timings_tau = np.append(timings_tau, [tau_b_1_1, tau_b_1_2, tau_b_2_1, tau_b_2_2])
    # minimisation procedure for parameters from formulae
    out_a = af.eclipse_parameters(p_orb, timings_tau, timings[10:], timings_err[:10], timings_err[10:], verbose=verbose)
    e, w, i, r_sum, r_rat, sb_rat = out_a
    ecosw, esinw = e * np.cos(w), e * np.sin(w)
    cosi = np.cos(i)
    phi_0 = af.phi_0_from_r_sum_sma(e, i, r_sum)
    # calculate the errors
    out_b = af.error_estimates_hdi(e, w, i, r_sum, r_rat, sb_rat, p_orb, timings[:10], timings[10:], p_err,
                                   timings_err[:10], timings_err[10:], p_t_corr, verbose=verbose)
    intervals, errors, dists_in, dists_out = out_b
    i_sym_err = max(errors[2])  # take the maximum as pessimistic estimate of the symmetric error
    formal_errors = af.formal_uncertainties(e, w, i, p_orb, *timings_tau[:6], p_err, i_sym_err, *timings_err[:6])
    # check physical result
    if (e > 0.99):
        logger.info(f'Unphysically large eccentricity found: {e}')
    # save the result
    e_err, w_err, i_err, r_sum_err, r_rat_err, sb_rat_err, ecosw_err, esinw_err, cosi_err, phi_0_err = errors
    sigma_e, sigma_w, sigma_phi_0, sigma_r_sum, sigma_ecosw, sigma_esinw = formal_errors
    ephem = np.array([p_orb, timings[0]])
    ephem_err = np.array([p_err, timings_err[0]])
    phys_mean = np.array([ecosw, esinw, cosi, phi_0, r_rat, sb_rat, e, w, i, r_sum])
    phys_err = np.array([sigma_ecosw, sigma_esinw, -1, sigma_phi_0, -1, -1, sigma_e, sigma_w, -1, sigma_r_sum])
    phys_hdi = np.array([ecosw_err, esinw_err, cosi_err, phi_0_err, r_rat_err, sb_rat_err,
                         e_err, w_err, i_err, r_sum_err])
    desc = 'Eclipse elements from timings.'
    ut.save_parameters_hdf5(file_name, ephem=ephem, ephem_err=ephem_err, phys_mean=phys_mean, phys_err=phys_err,
                            phys_hdi=phys_hdi, timings=timings, timings_err=timings_err, description=desc,
                            data_id=data_id)
    ut.save_results_dists(file_name, dists_in, dists_out, data_id=data_id)
    # print some useful stuff
    t_b = time.time()
    if verbose:
        # determine decimals to print for two significant figures
        rnd_e = max(ut.decimal_figures(min(e_err), 2), ut.decimal_figures(e, 2), 0)
        rnd_w = max(ut.decimal_figures(min(w_err) / np.pi * 180, 2), ut.decimal_figures(w / np.pi * 180, 2), 0)
        rnd_i = max(ut.decimal_figures(min(i_err) / np.pi * 180, 2), ut.decimal_figures(i / np.pi * 180, 2), 0)
        rnd_r_sum = max(ut.decimal_figures(min(r_sum_err), 2), ut.decimal_figures(r_sum, 2), 0)
        rnd_r_rat = max(ut.decimal_figures(min(r_rat_err), 2), ut.decimal_figures(r_rat, 2), 0)
        rnd_sb_rat = max(ut.decimal_figures(min(sb_rat_err), 2), ut.decimal_figures(sb_rat, 2), 0)
        rnd_ecosw = max(ut.decimal_figures(min(ecosw_err), 2), ut.decimal_figures(ecosw, 2), 0)
        rnd_esinw = max(ut.decimal_figures(min(esinw_err), 2), ut.decimal_figures(esinw, 2), 0)
        rnd_phi_0 = max(ut.decimal_figures(min(phi_0_err), 2), ut.decimal_figures(phi_0, 2), 0)
        print(f'\033[1;32;48mMeasurements and initial optimisation of the eclipse parameters complete.\033[0m')
        print(f'\033[0;32;48me: {e:.{rnd_e}f} (+{e_err[1]:.{rnd_e}f} -{e_err[0]:.{rnd_e}f}), \n'
              f'w: {w / np.pi * 180:.{rnd_w}f} '
              f'(+{w_err[1] / np.pi * 180:.{rnd_w}f} -{w_err[0] / np.pi * 180:.{rnd_w}f}) degrees, \n'
              f'i: {i / np.pi * 180:.{rnd_i}f} '
              f'(+{i_err[1] / np.pi * 180:.{rnd_i}f} -{i_err[0] / np.pi * 180:.{rnd_i}f}) degrees, \n'
              f'(r1+r2)/a: {r_sum:.{rnd_r_sum}f} '
              f'(+{r_sum_err[1]:.{rnd_r_sum}f} -{r_sum_err[0]:.{rnd_r_sum}f}), \n'
              f'r2/r1: {r_rat:.{rnd_r_rat}f} (+{r_rat_err[1]:.{rnd_r_rat}f} -{r_rat_err[0]:.{rnd_r_rat}f}), \n'
              f'sb2/sb1: {sb_rat:.{rnd_sb_rat}f} '
              f'(+{sb_rat_err[1]:.{rnd_sb_rat}f} -{sb_rat_err[0]:.{rnd_sb_rat}f}), \n'
              f'ecos(w): {ecosw:.{rnd_ecosw}f} (+{ecosw_err[1]:.{rnd_ecosw}f} -{ecosw_err[0]:.{rnd_ecosw}f}), \n'
              f'esin(w): {esinw:.{rnd_esinw}f} (+{esinw_err[1]:.{rnd_esinw}f} -{esinw_err[0]:.{rnd_esinw}f}), \n'
              f'phi_0: {phi_0:.{rnd_phi_0}f} (+{phi_0_err[1]:.{rnd_phi_0}f} -{phi_0_err[0]:.{rnd_phi_0}f}). \n'
              f'Time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return e, w, i, r_sum, r_rat, sb_rat, errors, formal_errors, dists_in, dists_out


def optimise_physical_elements(times, signal, signal_err, p_orb, t_zero, ecl_par, const, slope, f_n, a_n, ph_n,
                               phys_err, i_sectors, t_stats, file_name, method='sampler', data_id='none',
                               overwrite=False, verbose=False):
    """Optimise the parameters of the physical eclipse, sinusoid and linear model

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of the deepest minimum with respect to the mean time
    ecl_par: numpy.ndarray[float]
        Initial eclipse parameters to start the fit, consisting of:
        e, w, i, r_sum, r_rat, sb_rat
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    phys_err: numpy.ndarray[float]
        Errors in the initial eclipse parameters:
        e, w, i, r_sum, r_rat, sb_rat, ecosw, esinw, cosi, phi_0
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    method: str
        Method of optimization. Can be 'sampler' or 'fitter'.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    t_zero: float
        Time of the deepest minimum with respect to the mean time
    ecl_par: numpy.ndarray[float]
        Eclipse parameters, consisting of:
        e, w, i, r_sum, r_rat, sb_rat
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    
    Notes
    -----
    Eclipses are modelled by a simple physical model assuming
    spherical stars that have uniform brightness.
    
    The steps taken are:
    1) initial eclipse model fit on the non-harmonic sinusoids model subtracted light curve
    2) iterative prewhitening of the eclipse model subtracted light curve
    3) optimisation of the full model of eclipses and sinusoids using MCMC
    """
    t_a = time.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_parameters_hdf5(file_name, verbose=verbose)
        const, slope, f_n, a_n, ph_n = results['sin_mean']
        _, t_zero = results['ephem']
        ecosw, esinw, cosi, phi_0, r_rat, sb_rat, e, w, i, r_sum = results['phys_mean']
        ecl_par = (e, w, i, r_sum, r_rat, sb_rat)
        return t_zero, ecl_par, const, slope, f_n, a_n, ph_n
    
    if verbose:
        print(f'Starting multi-sine NL-LS optimisation with physical eclipse model.')
    t_tot, t_mean, t_mean_s, t_int = t_stats
    # convert some parameters and fit initial physical model
    out_a = tsfit.fit_eclipse_physical(times, signal, signal_err, p_orb, t_zero, ecl_par, phys_err, i_sectors,
                                       verbose=verbose)
    # extract the leftover signal from the residuals with the iterative scheme
    model_eclipse = tsfit.eclipse_physical_lc(times, p_orb, t_zero, *out_a[:6])
    resid_ecl = signal - model_eclipse
    out_b = tsf.extract_sinusoids(times, resid_ecl, signal_err, i_sectors, verbose=verbose)
    # remove any frequencies that end up not making the statistical cut
    out_c = tsf.reduce_frequencies(times, resid_ecl, signal_err, 0, *out_b, i_sectors, verbose=verbose)
    const, slope, f_n, a_n, ph_n = out_c
    # make model including everything to calculate noise level
    model_lin = tsf.linear_curve(times, const, slope, i_sectors)
    model_sin = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - (model_lin + model_sin + model_eclipse)
    noise_level = np.std(resid)
    # formal linear and sinusoid parameter errors
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, a_n, i_sectors)
    # do not include those frequencies that have too big uncertainty
    include = (ph_n_err < 1 / np.sqrt(6))  # circular distribution for ph_n cannot handle these
    f_n, a_n, ph_n = f_n[include], a_n[include], ph_n[include]
    f_n_err, a_n_err, ph_n_err = f_n_err[include], a_n_err[include], ph_n_err[include]
    # Monte Carlo sampling of full model
    inf_data, par_mean, sin_hdi, phys_hdi = None, None, None, None
    if method == 'fitter':
        out_d = tsfit.fit_eclipse_physical_sinusoid(times, signal, signal_err, p_orb, t_zero, out_a[:6], const, slope,
                                                    f_n, a_n, ph_n, i_sectors, model='simple', verbose=verbose)
        par_mean = list(out_d[:5]) + [*out_d[5]]
    else:
        out_d = mcf.sample_sinusoid_eclipse(times, signal, p_orb, t_zero, out_a[:6], const, slope, f_n, a_n, ph_n,
                                            phys_err, c_err, sl_err, f_n_err, a_n_err, ph_n_err, noise_level,
                                            i_sectors, verbose=verbose)
        inf_data, par_mean, par_hdi = out_d
        sin_hdi = par_hdi[:5]
        phys_hdi = np.array(par_hdi[5:])
    const, slope, f_n, a_n, ph_n = par_mean[:5]
    ecosw, esinw, cosi, phi_0, r_rat, sb_rat, e, w, i, r_sum = par_mean[5:]
    ecl_par = (e, w, i, r_sum, r_rat, sb_rat)
    # get theoretical timings and depths
    timings = af.eclipse_times(p_orb, t_zero, e, w, i, r_sum, r_rat)
    depths = af.eclipse_depths(e, w, i, r_sum, r_rat, sb_rat)
    # main function done, do the rest for this step
    model_lin = tsf.linear_curve(times, const, slope, i_sectors)
    model_sin = tsf.sum_sines(times, f_n, a_n, ph_n)
    model_eclipse = tsfit.eclipse_physical_lc(times, p_orb, t_zero, e, w, i, r_sum, r_rat, sb_rat)
    resid = signal - (model_lin + model_sin + model_eclipse)
    n_param = 2 + 6 + 2 * len(const) + 3 * len(f_n)
    bic = tsf.calc_bic(resid / signal_err, n_param)
    noise_level = np.std(resid)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, a_n, i_sectors)
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    ephem = np.array([p_orb, t_zero])
    phys_mean = np.array([ecosw, esinw, cosi, phi_0, r_rat, sb_rat, e, w, i, r_sum])
    timings = np.append(timings, depths)
    stats = [t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level]
    desc = 'Optimised linear + sinusoid + eclipse model.'
    ut.save_parameters_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, sin_hdi=sin_hdi, ephem=ephem,
                            phys_mean=phys_mean, phys_hdi=phys_hdi, timings=timings, stats=stats, i_sectors=i_sectors,
                            description=desc, data_id=data_id)
    ut.save_inference_data(file_name, inf_data)
    # print some useful stuff
    t_b = time.time()
    if verbose:
        print(f'\033[1;32;48mOptimisation complete.\033[0m')
        print(f'\033[0;32;48mNumber of frequencies: {len(f_n)}, \n'
              f'BIC of eclipse model plus sinusoids: {bic:1.2f}. \n'
              f'Time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return t_zero, ecl_par, const, slope, f_n, a_n, ph_n


def analyse_eclipses(times, signal, signal_err, i_sectors, t_stats, target_id, save_dir, method='sampler',
                     data_id='none', overwrite=False, verbose=False):
    """Part two of analysis recipe for analysis of EB light curves,
    to be chained after frequency_analysis

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_sectors = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    target_id: int
        The TESS Input Catalog number for later reference.
        Use any number (or string) as reference if not available.
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    method: str
        Method of optimization. Can be 'sampler' or 'fitter'.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    out_6: tuple
        output of find_eclipse_timings
    out_7: tuple
        output of convert_timings_to_elements
    out_8: tuple
        output of optimise_physical_elements
    
    Notes
    -----
    The followed recipe is:
    1) identify the eclipses and determine their timings
    2) convert the timings to physical parameters using formulae
    3) optimise the physical parameters using a physical model of the eclipses
    """
    signal_err = np.max(signal_err) * np.ones(len(times))  # likelihood assumes the same errors
    t_tot, t_mean, t_mean_s, t_int = t_stats
    arg_dict = {'data_id': data_id, 'overwrite': overwrite, 'verbose': verbose}
    # read in the frequency analysis results
    file_name = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_5.hdf5')
    if not os.path.isfile(file_name):
        if verbose:
            print('No eclipse analysis results found')
        return (None,) * 4
    results_5 = ut.read_parameters_hdf5(file_name, verbose=verbose)
    const_5, slope_5, f_n_5, a_n_5, ph_n_5 = results_5['sin_mean']
    p_orb_5, _ = results_5['ephem']
    p_err_5, _ = results_5['ephem_err']
    _, _, _, _, _, _, noise_level_5 = results_5['stats']
    # ------------------------------------
    # --- [6] --- Initial eclipse timings
    # ------------------------------------
    file_name = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_6.hdf5')
    out_6 = find_eclipse_timings(times, signal, p_orb_5, const_5, slope_5, f_n_5, a_n_5, ph_n_5, i_sectors,
                                 noise_level_5, file_name, **arg_dict)
    timings_6, timings_err_6 = out_6
    # perform checks for stopping the analysis
    if np.any([item is None for item in out_6]):
        return (None,) * 4  # could not find eclipses for some reason
    elif (timings_6[10] <= 0) | (timings_6[11] <= 0):
        return (None,) * 4  # negative depths
    # check for significance
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_t_b_1_2, t_b_2_1, t_b_2_2, d_1, d_2 = timings_6
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err = timings_err_6[:6]
    t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err, depth_1_err, depth_2_err = timings_err_6[6:]
    dur_1, dur_2 = (t_1_2 - t_1_1), (t_2_2 - t_2_1)
    dur_1_err, dur_2_err = np.sqrt(t_1_1_err**2 + t_1_2_err**2), np.sqrt(t_2_1_err**2 + t_2_2_err**2)
    dur_diff = (dur_1 < 0.001 * dur_2) | (dur_2 < 0.001 * dur_1)
    depth_insig = (d_1 < depth_1_err) | (d_2 < depth_2_err)
    dur_insig = (dur_1 < dur_1_err) | (dur_2 < dur_2_err)
    if dur_diff | depth_insig | dur_insig:
        if depth_insig:
            message = f'One of the eclipses too shallow, depths: {d_1}, {d_2}, err: {depth_1_err}, {depth_2_err}'
        elif dur_insig:
            message = f'One of the eclipses too narrow, durations: {dur_1}, {dur_2}, err: {dur_1_err}, {dur_2_err}'
        else:
            message = f'One of the eclipses too narrow compared to the other, durations: {dur_1}, {dur_2}'
        logger.info(message)
        return (None,) * 4  # likely unphysical parameters
    # ------------------------------------
    # --- [7] --- Initial orbital elements
    # ------------------------------------
    file_name = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_7.hdf5')
    _, _, p_t_corr = af.linear_regression_uncertainty(p_orb_5, np.ptp(times), sigma_t=t_int / 2)
    out_7 = convert_timings_to_elements(p_orb_5, timings_6, p_err_5, timings_err_6, p_t_corr, file_name, **arg_dict)
    e_7, w_7, i_7, r_sum_7, r_rat_7, sb_rat_7 = out_7[:6]
    errors_7, formal_errors_7, dists_in_7, dists_out_7 = out_7[6:]
    e_err, w_err, i_err, r_sum_err, r_rat_err, sb_rat_err, ecosw_err, esinw_err, cosi_err, phi_0_err = errors_7
    sigma_e, sigma_w, sigma_phi_0, sigma_r_sum, sigma_ecosw, sigma_esinw = formal_errors_7
    # for the mcmc, take maximum errors for prior model
    e_err_7 = max(e_err[0], e_err[1], sigma_e)
    w_err_7 = max(w_err[0], w_err[1], sigma_w)
    i_err_7 = max(i_err[0], i_err[1])
    r_sum_err_7 = max(r_sum_err[0], r_sum_err[1], sigma_r_sum)
    r_rat_err_7 = max(r_rat_err[0], r_rat_err[1])
    sb_rat_err_7 = max(sb_rat_err[0], sb_rat_err[1])
    ecosw_err_7 = max(ecosw_err[0], ecosw_err[1], sigma_ecosw)
    esinw_err_7 = max(esinw_err[0], esinw_err[1], sigma_esinw)
    cosi_err_7 = max(cosi_err[0], cosi_err[1])
    phi_0_err_7 = max(phi_0_err[0], phi_0_err[1], sigma_phi_0)
    phys_err_7 = np.array([e_err_7, w_err_7, i_err_7, r_sum_err_7, r_rat_err_7, sb_rat_err_7,
                           ecosw_err_7, esinw_err_7, cosi_err_7, phi_0_err_7])
    ecl_par_7 = (e_7, w_7, i_7, r_sum_7, r_rat_7, sb_rat_7)
    if (e_7 > 0.99):
        return (None,) * 4  # unphysical parameters
    # --------------------------------------------------
    # --- [8] --- Optimise elements with physical model
    # --------------------------------------------------
    file_name = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_8.hdf5')
    out_8 = optimise_physical_elements(times, signal, signal_err, p_orb_5, timings_6[0], ecl_par_7, const_5, slope_5,
                                       f_n_5, a_n_5, ph_n_5, phys_err_7, i_sectors, t_stats, file_name,
                                       method=method, **arg_dict)
    return out_6, out_7, out_8


def frequency_selection(times, signal, model_eclipse, p_orb, const, slope, f_n, a_n, ph_n, noise_level, i_sectors,
                        t_stats, file_name, data_id='none', overwrite=False, verbose=False):
    """Selects the credible frequencies from the given set,
    ignoring the harmonics
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    model_eclipse: numpy.ndarray[float]
        Model of the eclipses at the same times
    p_orb: float
        Orbital period of the eclipsing binary in days
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    noise_level: float
        The noise level (standard deviation of the residuals)
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    passed_sigma: numpy.ndarray[bool]
        Non-harmonic frequencies that passed the sigma check
    passed_snr: numpy.ndarray[bool]
        Non-harmonic frequencies that passed the signal-to-noise check
    passed_both: numpy.ndarray[bool]
        Non-harmonic frequencies that passed both checks

    """
    t_a = time.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_parameters_hdf5(file_name, verbose=verbose)
        passed_sigma, passed_snr, passed_both, passed_h = results['sin_select']
        return passed_sigma, passed_snr, passed_both, passed_h
    
    if verbose:
        print(f'Selecting credible frequencies')
    t_tot, t_mean, t_mean_s, t_int = t_stats
    n_points = len(times)
    # obtain the errors on the sine waves (dependends on residual and thus model)
    model_lin = tsf.linear_curve(times, const, slope, i_sectors)
    model_sin = tsf.sum_sines(times, f_n, a_n, ph_n)
    residuals = signal - (model_lin + model_sin + model_eclipse)
    errors = tsf.formal_uncertainties(times, residuals, a_n, i_sectors)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = errors
    # find the insignificant frequencies
    remove_sigma = af.remove_insignificant_sigma(f_n, f_n_err, a_n, a_n_err, sigma_a=3, sigma_f=3)
    # apply the signal-to-noise threshold
    noise_at_f = tsf.scargle_noise_at_freq(f_n, times, residuals)
    remove_snr = af.remove_insignificant_snr(a_n, noise_at_f, n_points)
    # frequencies that pass sigma criteria
    passed_sigma = np.ones(len(f_n), dtype=bool)
    passed_sigma[remove_sigma] = False
    # frequencies that pass S/N criteria
    passed_snr = np.ones(len(f_n), dtype=bool)
    passed_snr[remove_snr] = False
    # passing both
    passed_both = (passed_sigma & passed_snr)
    # candidate harmonic frequencies
    freq_res = 1.5 / t_tot  # Rayleigh criterion
    harmonics, harmonic_n = af.select_harmonics_sigma(f_n, f_n_err, p_orb, f_tol=freq_res / 2, sigma_f=3)
    passed_h = np.zeros(len(f_n), dtype=bool)
    passed_h[harmonics] = True
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    sin_select = [passed_sigma, passed_snr, passed_h]
    stats = [t_tot, t_mean, t_mean_s, t_int, -1, -1, noise_level]
    desc = 'Credible frequency selection'
    ut.save_parameters_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, sin_select=sin_select, stats=stats,
                            i_sectors=i_sectors, description=desc, data_id=data_id)
    # print some useful stuff
    t_b = time.time()
    if verbose:
        print(f'\033[1;32;48mNon-harmonic frequencies selected.\033[0m')
        print(f'\033[0;32;48mNumber of frequencies passed: {np.sum(passed_both)} of {len(f_n)}. '
              f'Candidate harmonics: {np.sum(passed_h)}. \nTime taken: {t_b - t_a:1.1f}s\033[0m\n')
    return passed_sigma, passed_snr, passed_both, passed_h


def variability_amplitudes(times, signal, model_eclipse, p_orb, const, slope, f_n, a_n, ph_n, depths, i_sectors,
                           t_stats, file_name, data_id='none', overwrite=False, verbose=False):
    """Determine several levels of variability

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    model_eclipse: numpy.ndarray[float]
        Model of the eclipses at the same times
    p_orb: float
        Orbital period of the eclipsing binary in days
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    depths: numpy.ndarray[float]
        Eclipse depth of the primary and secondary, depth_1, depth_2
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    std_1: float
        Standard deviation of the residuals of the
        linear, sinusoid and eclipse model
    std_2: float
        Standard deviation of the residuals of the
        linear and eclipse model
    std_3: float
        Standard deviation of the residuals of the
        linear, harmonic 1 and 2 and eclipse model
    std_4: float
        Standard deviation of the residuals of the
        linear, non-harmonic sinusoid and eclipse model
    ratios_1: numpy.ndarray[float]
        Ratios of the eclipse depths to std_1
    ratios_2: numpy.ndarray[float]
        Ratios of the eclipse depths to std_2
    ratios_3: numpy.ndarray[float]
        Ratios of the eclipse depths to std_3
    ratios_4: numpy.ndarray[float]
        Ratios of the eclipse depths to std_4
    """
    t_a = time.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_parameters_hdf5(file_name, verbose=verbose)
        std_1, std_2, std_3, std_4, ratios_1, ratios_2, ratios_3, ratios_4 = results['var_stats']
        return std_1, std_2, std_3, std_4, ratios_1, ratios_2, ratios_3, ratios_4
    
    if verbose:
        print(f'Determining variability levels')
    t_tot, t_mean, t_mean_s, t_int = t_stats
    freq_res = 1.5 / t_tot  # Rayleigh criterion
    # [maybe this can go in f select anyway, and then also compute std of passing sines]
    # make the linear and sinusoid models
    model_lin = tsf.linear_curve(times, const, slope, i_sectors)
    model_sin = tsf.sum_sines(times, f_n, a_n, ph_n)
    # get the 2 lowest harmonic frequencies
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=freq_res / 2)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    mask_low_h = (harmonic_n == 1) | (harmonic_n == 2)
    low_h = harmonics[mask_low_h]
    model_sin_lh = tsf.sum_sines(times, f_n[low_h], a_n[low_h], ph_n[low_h])
    model_sin_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    # add models to other models
    model_lin_ecl = model_lin + model_eclipse
    model_lin_lh_ecl = model_lin + model_sin_lh + model_eclipse
    model_lin_nh_ecl = model_lin + model_sin_nh + model_eclipse
    model_lin_sin_ecl = model_lin + model_sin + model_eclipse
    # determine amplitudes of leftover variability
    std_1 = np.std(signal - model_lin_sin_ecl)
    std_2 = np.std(signal - model_lin_ecl)
    std_3 = np.std(signal - model_lin_lh_ecl)
    std_4 = np.std(signal - model_lin_nh_ecl)
    # calculate some ratios with eclipse depths
    ratios_1 = depths / std_1
    ratios_2 = depths / std_2
    ratios_3 = depths / std_3
    ratios_4 = depths / std_4
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    var_stats = [std_1, std_2, std_3, std_4, ratios_1, ratios_2, ratios_3, ratios_4]
    stats = [t_tot, t_mean, t_mean_s, t_int, -1, -1, -1]
    desc = 'Variability versus eclipse depth statistics'
    ut.save_parameters_hdf5(file_name, sin_mean=sin_mean, var_stats=var_stats, stats=stats, i_sectors=i_sectors,
                            description=desc, data_id=data_id)
    # print some useful stuff
    t_b = time.time()
    if verbose:
        print(f'\033[1;32;48mVariability levels calculated.\033[0m')
        print(f'\033[0;32;48mRatios of eclipse depth to leftover variability: {ratios_3[0]:2.3}, {ratios_3[1]:2.3}. \n'
              f'Time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return std_1, std_2, std_3, std_4, ratios_1, ratios_2, ratios_3, ratios_4


def analyse_pulsations(times, signal, signal_err, i_sectors, t_stats, target_id, save_dir, data_id='none',
                       overwrite=False, verbose=False):
    """Part two of analysis recipe for analysis of EB light curves,
    to be chained after frequency_analysis

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_sectors = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    target_id: int
        The TESS Input Catalog number for later reference.
        Use any number (or string) as reference if not available.
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    out_9: tuple
        output of frequency_selection
    out_10: tuple
        output of variability_amplitudes
    """
    signal_err = np.max(signal_err) * np.ones(len(times))  # likelihood assumes the same errors
    arg_dict = {'data_id': data_id, 'overwrite': overwrite, 'verbose': verbose}
    # read in the eclipse depths
    file_name = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_6.hdf5')
    if not os.path.isfile(file_name):
        if verbose:
            print('No frequency analysis results found')
        return (None,) * 2
    results_6 = ut.read_parameters_hdf5(file_name, verbose=verbose)
    depths_6 = results_6['timings'][10:]
    # read in the eclipse analysis results
    file_name = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_8.hdf5')
    if not os.path.isfile(file_name):
        if verbose:
            print('No eclipse analysis results found')
        return (None,) * 2
    results_8 = ut.read_parameters_hdf5(file_name, verbose=verbose)
    const_8, slope_8, f_n_8, a_n_8, ph_n_8 = results_8['sin_mean']
    p_orb_8, t_zero_8 = results_8['ephem']
    _, _, _, _, r_rat_8, sb_rat_8, e_8, w_8, i_8, r_sum_8 = results_8['phys_mean']
    ecl_par_8 = (e_8, w_8, i_8, r_sum_8, r_rat_8, sb_rat_8)
    t_tot, t_mean, t_mean_s, t_int, n_param_8, bic_8, noise_level_8 = results_8['stats']
    model_ecl_8 = tsfit.eclipse_physical_lc(times, p_orb_8, t_zero_8, *ecl_par_8)
    # ---------------------------------------------
    # --- [9] --- Frequency and harmonic selection
    # ---------------------------------------------
    file_name = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_9.hdf5')
    out_9 = frequency_selection(times, signal, model_ecl_8, p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                                 noise_level_8, i_sectors, t_stats, file_name, **arg_dict)
    # pass_nh_sigma, pass_nh_snr, passed_nh_b = out_9
    # -----------------------------------
    # --- [10] --- Variability amplitudes
    # -----------------------------------
    file_name = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_10.hdf5')
    out_10 = variability_amplitudes(times, signal, model_ecl_8, p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                                    depths_6, i_sectors, t_stats, file_name, **arg_dict)
    # std_1, std_2, std_3, std_4, ratios_1, ratios_2, ratios_3, ratios_4 = out_10
    # ---------------------------------
    # --- [11] --- Amplitude modulation
    # ---------------------------------
    # use wavelet transform or smth to see which star is pulsating
    return out_9, out_10


def customize_logger(save_dir, target_id, verbose):
    """Create a custom logger for logging to file and to stdout
    
    Parameters
    ----------
    save_dir: str
        folder to save the log file
    target_id: str
        Identifier to use for the log file
    verbose: bool
        If set to True, information will be printed by the logger
    
    Returns
    -------
     : None
    """
    # customize the logger
    logger.setLevel(logging.INFO)  # set base activation level for logger
    # make formatters for the handlers
    s_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    f_format = logging.Formatter(fmt='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    # remove existing handlers to avoid duplicate messages
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # make stream handler
    if verbose:
        s_handler = logging.StreamHandler()  # for printing
        s_handler.setLevel(logging.INFO)  # print everything with level 20 or above
        s_handler.setFormatter(s_format)
        logger.addHandler(s_handler)
    # file handler
    logname = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}.log')
    f_handler = logging.FileHandler(logname, mode='a')  # for saving
    f_handler.setLevel(logging.INFO)  # save everything with level 20 or above
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    return None


def period_from_file(file_name, i_sectors=None, method='fitter', data_id='none', overwrite=False, verbose=False):
    """Do the global period search for a given light curve file

    Parameters
    ----------
    file_name: str
        Path to a file containing the light curve data, with
        timestamps, normalised flux, error values as the
        first three columns, respectively.
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_sectors = np.array([[0, len(times)]]).
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    None
    
    Notes
    -----
    Results are saved in the same directory as the given file
    
    The input files are expected to have three columns with in order:
    times, signal, signal_err
    And the timestamps should be in ascending order.
    """
    target_id = os.path.splitext(os.path.basename(file_name))[0]  # file name is used as target identifier
    save_dir = os.path.dirname(file_name)
    # for saving, make a folder if not there yet
    if not os.path.isdir(os.path.join(save_dir, f'{target_id}_analysis')):
        os.mkdir(os.path.join(save_dir, f'{target_id}_analysis'))  # create the subdir
    customize_logger(save_dir, target_id, verbose)  # log stuff to a file and/or stdout
    logger.info('Start of analysis')
    # load the data
    times, signal, signal_err = np.loadtxt(file_name, usecols=(0, 1, 2), unpack=True)
    # if sectors not given, take full length
    if i_sectors is None:
        i_sectors = np.array([[0, len(times)]])  # no sector information
    i_half_s = i_sectors  # in this case no differentiation between half or full sectors
    # calculate some parameters
    t_tot = np.ptp(times)  # total time base of observations
    t_mean = np.mean(times)  # mean time of observations
    t_mean_s = np.array([np.mean(times[s[0]:s[1]]) for s in i_sectors])  # mean time per observation sector
    t_int = np.median(np.diff(times))  # integration time, taken to be the median time step
    t_stats = [t_tot, t_mean, t_mean_s, t_int]
    kw_args = {'data_id': data_id, 'overwrite': overwrite, 'verbose': verbose}
    # do the prewhitening and frequency optimisation
    file_name = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_1.hdf5')
    out_1 = iterative_prewhitening(times, signal, signal_err, i_half_s, t_stats, file_name, **kw_args)
    const_1, slope_1, f_n_1, a_n_1, ph_n_1 = out_1
    if (len(f_n_1) == 0):
        logger.info('No frequencies found.')
        return -1
    file_name = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_2.hdf5')
    out_2 = optimise_sinusoid(times, signal, signal_err, const_1, slope_1, f_n_1, a_n_1, ph_n_1, i_sectors, t_stats,
                              file_name, method=method, **kw_args)
    const_2, slope_2, f_n_2, a_n_2, ph_n_2 = out_2
    # find orbital period
    p_orb = find_orbital_period(times, signal, f_n_2, t_tot)
    p_err, _, _ = af.linear_regression_uncertainty(p_orb, t_tot, sigma_t=t_int / 2)
    # save p_orb
    file_name = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_period.txt')
    col1 = ['period (days)', 'period error (days)', 'time-base (days)', 'number of frequencies']
    col2 = [p_orb, p_err, t_tot, len(f_n_1)]
    np.savetxt(file_name, np.column_stack((col1, col2)), fmt='%s', delimiter=',')
    logger.info('End of analysis')
    if verbose:
        print(f'P_orb = {p_orb}, done.')
    return p_orb


def analyse_eb(times, signal, signal_err, p_orb, i_sectors, target_id, save_dir, method='sampler', data_id='none',
               overwrite=False, verbose=False):
    """Do all steps of the analysis

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days (set zero if unkown)
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_sectors = np.array([[0, len(times)]]).
    target_id: int, str
        Target identifier
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    method: str
        Method of optimization. Can be 'sampler' or 'fitter'.
        Sampler gives better error estimates and more accurate results
        Fitter can be much faster on large datasets but compromises on accuracy
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    None
    """
    # for saving, make a folder if not there yet
    if not os.path.isdir(os.path.join(save_dir, f'{target_id}_analysis')):
        os.mkdir(os.path.join(save_dir, f'{target_id}_analysis'))  # create the subdir
    # create a log
    customize_logger(save_dir, target_id, verbose)  # log stuff to a file and/or stdout
    logger.info('Start of analysis')  # info to save to log
    # time series stats
    t_tot = np.ptp(times)  # total time base of observations
    t_mean = np.mean(times)  # mean time of observations
    t_mean_s = np.array([np.mean(times[s[0]:s[1]]) for s in i_sectors])  # mean time per observation sector
    t_int = np.median(np.diff(times))  # integration time, taken to be the median time step
    t_stats = [t_tot, t_mean, t_mean_s, t_int]
    # keyword arguments in common between some functions
    kw_args = {'save_dir': save_dir, 'data_id': data_id, 'overwrite': overwrite, 'verbose': verbose}
    # do the analysis
    out_a = analyse_frequencies(times, signal, signal_err, i_sectors, p_orb, t_stats, target_id, method=method,
                                **kw_args)
    # if not full output, stop
    if not (len(out_a[0]) < 5):
        out_b = analyse_eclipses(times, signal, signal_err, i_sectors, t_stats, target_id, method=method, **kw_args)
        if not np.any([item is None for item in out_b]):
            out_c = analyse_pulsations(times, signal, signal_err, i_sectors, t_stats, target_id, **kw_args)
    # create summary file
    ut.save_summary(target_id, save_dir, data_id=data_id)
    logger.info('End of analysis')  # info to save to log
    if verbose:
        print('done.')
    return None


def analyse_from_file(file_name, p_orb=0, i_sectors=None, method='sampler', data_id='none', overwrite=False,
                      verbose=False):
    """Do all steps of the analysis for a given light curve file

    Parameters
    ----------
    file_name: str
        Path to a file containing the light curve data, with
        timestamps, normalised flux, error values as the
        first three columns, respectively.
    p_orb: float
        Orbital period of the eclipsing binary in days
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_sectors = np.array([[0, len(times)]]).
    method: str
        Method of optimization. Can be 'sampler' or 'fitter'.
        Sampler gives better error estimates and more accurate results
        Fitter can be much faster on large datasets but compromises on accuracy
    data_id: int, str
        Identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    None

    Notes
    -----
    Results are saved in the same directory as the given file
    
    The input files are expected to have three columns with in order:
    times, signal, signal_err
    And the timestamps should be in ascending order.
    """
    target_id = os.path.splitext(os.path.basename(file_name))[0]  # file name is used as target identifier
    save_dir = os.path.dirname(file_name)
    # load the data
    times, signal, signal_err = np.loadtxt(file_name, usecols=(0, 1, 2), unpack=True)
    # if sectors not given, take full length
    if i_sectors is None:
        i_sectors = np.array([[0, len(times)]])  # no sector information
    i_half_s = i_sectors  # in this case no differentiation between half or full sectors
    # do the analysis
    analyse_eb(times, signal, signal_err, p_orb, i_half_s, target_id, save_dir, method=method, data_id=data_id,
               overwrite=overwrite, verbose=verbose)
    return None


def analyse_from_tic(tic, all_files, p_orb=0, method='sampler', data_id='none', save_dir=None, overwrite=False,
                     verbose=False):
    """Do all steps of the analysis for a given TIC number
    
    Parameters
    ----------
    tic: int
        The TESS Input Catalog (TIC) number for loading/saving the data
        and later reference.
    all_files: list[str]
        List of all the TESS data product '.fits' files. The files
        with the corresponding TIC number are selected.
    p_orb: float
        Orbital period of the eclipsing binary in days
    method: str
        Method of optimization. Can be 'sampler' or 'fitter'.
        Sampler gives better error estimates and more accurate results
        Fitter can be much faster on large datasets but compromises on accuracy
    data_id: int, str
        Identification for the dataset used
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    None
    """
    # load the data
    lc_data = ut.load_tess_lc(tic, all_files, apply_flags=True)
    times, sap_signal, signal, signal_err, sectors, t_sectors, crowdsap = lc_data
    i_sectors = ut.convert_tess_t_sectors(times, t_sectors)
    lc_processed = ut.stitch_tess_sectors(times, signal, signal_err, i_sectors)
    times, signal, signal_err, sector_medians, t_combined, i_half_s = lc_processed
    # do the analysis
    analyse_eb(times, signal, signal_err, p_orb, i_half_s, tic, save_dir, method=method, data_id=data_id,
               overwrite=overwrite, verbose=verbose)
    return None


def analyse_set(target_list, function='analyse_from_tic', n_threads=os.cpu_count() - 2, **kwargs):
    """Analyse a set of light curves in parallel
    
    Parameters
    ----------
    target_list: list[str], list[int]
        List of either file names or TIC identifiers to analyse
    function: str
        Name  of the function to use for the analysis
        Choose from [analyse_from_file, analyse_from_tic]
    n_threads: int
        Number of threads to use. Uses two less than the
        available amount by default.
    **kwargs: dict
        Extra arguments to 'function': refer to each function's
        documentation for a list of all possible arguments.
    
    Returns
    -------
    None
    """
    if 'p_orb' in kwargs.keys():
        # Use mp.Pool.starmap for this
        raise NotImplementedError('keyword p_orb found in kwargs: this functionality is not yet implemented')
    if 'i_sectors' in kwargs.keys():
        # Use mp.Pool.starmap for this
        raise NotImplementedError('keyword i_sectors found in kwargs: this functionality is not yet implemented')
    
    t1 = time.time()
    with mp.Pool(processes=n_threads) as pool:
        pool.map(fct.partial(eval(function), **kwargs), target_list, chunksize=1)
    t2 = time.time()
    print(f'Finished analysing set in: {(t2 - t1):1.2} s ({(t2 - t1) / 3600:1.2} h) for {len(target_list)} targets,\n'
          f'using {n_threads} threads ({(t2 - t1) * n_threads / len(target_list):1.2} s '
          f'average per target single threaded).')
    return None
