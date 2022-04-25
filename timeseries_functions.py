"""STAR SHADOW
Satellite Time-series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This Python module contains functions for time series analysis;
specifically for the analysis of stellar oscillations and eclipses.

Code written by: Luc IJspeert
"""

import os
import time
import numpy as np
import scipy as sp
import numba as nb
import itertools as itt

import utility as ut
import visualisation as vis
import timeseries_fitting as tsfit
import analysis_functions as af


@nb.njit(cache=True)
def fold_time_series(times, p_orb, zero=None):
    """Fold the given time series over the orbital period to transform to phase space.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    p_orb: float
        The orbital period with which the time series is folded
    zero: float, None
        Reference zero point in time where the phase equals zero
    
    Returns
    -------
    phases: numpy.ndarray[float]
        Phase array for all timestamps. Phases are between -0.5 and 0.5
    """
    if zero is None:
        zero = times[0]
    phases = ((times - zero) / p_orb + 0.5) % 1 - 0.5
    return phases


def bin_folded_signal(phases, signal, bins, midpoints=False, statistic='mean'):
    """Average the phase folded signal within a given number of bins.
    
    Parameters
    ----------
    phases: numpy.ndarray[float]
        The phase-folded timestamps of the time series, between -0.5 and 0.5.
    signal: numpy.ndarray[float]
        Measurement values of the time series
    bins: int, numpy.ndarray[float]
        Either the number of bins or a set of bin edges to be used
    midpoints: bool
        To return bins as midpoints instead of edges, set True
    statistic: str
        The statistic to calculate for each bin (see scipy.stats.binned_statistic)
        
    Returns
    -------
    bins: numpy.ndarray[float]
        The bin edges, or bin midpoints if midpoints=True
    binned: numpy.ndarray[float]
        The calculated statistic for each bin
    
    Notes
    -----
    Uses scipy.stats.binned_statistic for flexibility. For the use in number
    crunching, use a specialised function that can be jitted instead.
    """
    if not hasattr(bins, '__len__'):
        # use as number of bins, else use as bin edges
        bins = np.linspace(-0.5, 0.5, bins + 1)
    binned, edges, indices = sp.stats.binned_statistic(phases, signal, statistic=statistic, bins=bins)
    if midpoints:
        bins = (bins[1:] + bins[:-1]) / 2
    return bins, binned


@nb.njit()
def phase_dispersion(phases, signal, n_bins):
    """Phase dispersion, as in PDM, without overlapping bins.
    
    Parameters
    ----------
    phases: numpy.ndarray[float]
        The phase-folded timestamps of the time series, between -0.5 and 0.5.
    signal: numpy.ndarray[float]
        Measurement values of the time series
    n_bins: int
        The number of bins over the orbital phase
    
    Returns
    -------
    total_var/overall_var: float
        Phase dispersion, or summed variance over the bins divided by
        the variance of the signal
    
    Notes
    -----
    Intentionally does not make use of bin_folded_signal (which uses scipy)
    to enable jitting, which makes this considerably faster.
    """
    def var_no_avg(a):
        return np.sum(np.abs(a - np.mean(a))**2)  # if mean instead of sum, this is variance
    
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    # binned, edges, indices = sp.stats.binned_statistic(phases, signal, statistic=statistic, bins=bins)
    binned_var = np.zeros(n_bins)
    for i, (b1, b2) in enumerate(zip(edges[:-1], edges[1:])):
        bin_mask = (phases >= b1) & (phases < b2)
        if np.any(bin_mask):
            binned_var[i] = var_no_avg(signal[bin_mask])
        else:
            binned_var[i] = 0
    total_var = np.sum(binned_var) / len(signal)
    overall_var = np.var(signal)
    return total_var / overall_var


@nb.njit()
def phase_dispersion_minimisation(times, signal, f_n, local=False):
    """Determine the phase dispersion over a set of periods to find the minimum
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    
    Returns
    -------
    periods: numpy.ndarray[float]
        Periods at which the phase dispersion is calculated
    pd_all: numpy.ndarray[float]
        Phase dispersion at the given periods
    """
    # number of bins for dispersion calculation
    n_dpoints = len(times)
    if (n_dpoints / 10 > 1000):
        n_bins = 1000
    else:
        n_bins = n_dpoints // 10  # at least 10 data points per bin on average
    # determine where to look based on the frequencies, including fractions of the frequencies
    if local:
        periods = 1 / f_n
    else:
        periods = np.zeros(7 * len(f_n))
        for i, f in enumerate(f_n):
            periods[7*i:7*i+7] = np.arange(1, 8) / f
    # stay below the maximum
    periods = periods[periods < np.ptp(times)]
    # compute the dispersion measures
    pd_all = np.zeros(len(periods))
    for i, p in enumerate(periods):
        fold = fold_time_series(times, p, 0)
        pd_all[i] = phase_dispersion(fold, signal, n_bins)
    return periods, pd_all


# @nb.njit()  # not jitted for convolve with arg 'same'
def noise_spectrum(times, signal, window_width=1.):
    """Calculate the noise spectrum by a convolution with a flat window of a certain width.

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    window_width: float
        The width of the window used to compute the noise spectrum,
        in inverse unit of the times array (i.e. 1/d if time is in d).

    Returns
    -------
    noise: numpy.ndarray[float]
        The noise spectrum in the frequency interval of the periodogram,
        in the same units as ampls.
    """
    # calculate the periodogram
    freqs, ampls = scargle(times, signal)  # use defaults to get full amplitude spectrum
    # determine the number of points to extend the spectrum with for convolution
    n_points = int(np.ceil(window_width / np.abs(freqs[1] - freqs[0])))#.astype(int)
    window = np.full(n_points, 1 / n_points)
    # extend the array with mirrors for convolution
    ext_ampls = np.concatenate((ampls[(n_points - 1)::-1], ampls, ampls[:-(n_points + 1):-1]))
    ext_noise = np.convolve(ext_ampls, window, 'same')
    # cut back to original interval
    noise = ext_noise[n_points:-n_points]
    # extra correction to account for convolve mode='full' instead of 'same' (needed for jitting)
    # noise = noise[n_points//2 - 1:-n_points//2]
    return noise


# @nb.njit()  # not sped up
def noise_at_freq(fs, times, signal, window_width=0.5):
    """Calculate the noise at a given set of frequencies
    
    Parameters
    ----------
    fs: numpy.ndarray[float]
        The frequencies at which to calculate the noise
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    window_width: float
        The width of the window used to compute the noise spectrum,
        in inverse unit of the times array (i.e. 1/d if time is in d).

    Returns
    -------
    noise: numpy.ndarray[float]
        The noise level calculated from a window around the frequency in the periodogram
    """
    freqs, ampls = scargle(times, signal)  # use defaults to get full amplitude spectrum
    margin = window_width / 2
    noise = np.array([np.average(ampls[(freqs > f - margin) & (freqs <= f + margin)]) for f in fs])
    return noise


def spectral_window(times, freqs):
    """Computes the modulus square of the spectral window W_N(f) of a set of
    time points at the given frequencies.

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    freqs: numpy.ndarray[float]
        Frequency points to calculate the window. Inverse unit of 'times'
        
    Returns
    -------
    spec_win: numpy.ndarray[float]
        The spectral window at the given frequencies, |W(freqs)|^2
    
    Notes
    -----
    The spectral window is the Fourier transform of the window function
    w_N(t) = 1/N sum(Dirac(t - t_i))
    The time points do not need to be equidistant.
    The normalisation is such that 1.0 is returned at frequency 0.
    """
    n_time = len(times)
    cos_term = np.sum(np.cos(2.0 * np.pi * freqs * times.reshape(n_time, 1)), axis=0)
    sin_term = np.sum(np.sin(2.0 * np.pi * freqs * times.reshape(n_time, 1)), axis=0)
    winkernel = cos_term**2 + sin_term**2
    # Normalise such that winkernel(nu = 0.0) = 1.0
    spec_win = winkernel / n_time**2
    return spec_win


@nb.njit()
def scargle(times, signal, f0=0, fn=0, df=0, norm='amplitude'):
    """Scargle periodogram with no weights.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    f0: float
        Starting frequency of the periodogram.
        If left zero, default is f0 = 1/(100*T)
    fn: float
        Last frequency of the periodogram.
        If left zero, default is fn = 1/(2*np.min(np.diff(times))) = Nyquist frequency
    df: float
        Frequency sampling space of the periodogram
        If left zero, default is df = 1/(10*T) = oversampling factor of ten (recommended)
    norm: str
        Normalisation of the periodogram. Choose from:
        'amplitude', 'density' or 'distribution'
    
    Returns
    -------
    f1: numpy.ndarray[float]
        Frequencies at which the periodogram was calculated
    s1: numpy.ndarray[float]
        The periodogram spectrum in the chosen units
    
    Notes
    -----
    Translated from Fortran (and just as fast when JITted with Numba!)
        Computation of Scargles periodogram without explicit tau
        calculation, with iteration (Method Cuypers)
        (is this the same: https://ui.adsabs.harvard.edu/abs/1989ApJ...338..277P/abstract ?)
    Useful extra information: VanderPlas 2018,
        https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract
    """
    n = len(signal)
    t_tot = np.ptp(times)
    f0 = max(f0, 0.01 / t_tot)  # don't go lower than T/100
    if (df == 0):
        df = 0.1 / t_tot
    if (fn == 0):
        fn = 1 / (2 * np.min(times[1:] - times[:-1]))
    nf = int((fn - f0) / df + 0.001) + 1
    # preassign some memory
    ss = np.zeros(nf)
    sc = np.zeros(nf)
    ss2 = np.zeros(nf)
    sc2 = np.zeros(nf)
    # here is the actual calculation:
    two_pi = 2 * np.pi
    for i in range(n):
        t_f0 = (times[i] * two_pi * f0) % two_pi
        sin_f0 = np.sin(t_f0)
        cos_f0 = np.cos(t_f0)
        mc_1_a = 2 * sin_f0 * cos_f0
        mc_1_b = cos_f0 * cos_f0 - sin_f0 * sin_f0

        t_df = (times[i] * two_pi * df) % two_pi
        sin_df = np.sin(t_df)
        cos_df = np.cos(t_df)
        mc_2_a = 2 * sin_df * cos_df
        mc_2_b = cos_df * cos_df - sin_df * sin_df
        
        sin_f0_s = sin_f0 * signal[i]
        cos_f0_s = cos_f0 * signal[i]
        for j in range(nf):
            ss[j] = ss[j] + sin_f0_s
            sc[j] = sc[j] + cos_f0_s
            temp_cos_f0_s = cos_f0_s
            cos_f0_s = temp_cos_f0_s * cos_df - sin_f0_s * sin_df
            sin_f0_s = sin_f0_s * cos_df + temp_cos_f0_s * sin_df
            ss2[j] = ss2[j] + mc_1_a
            sc2[j] = sc2[j] + mc_1_b
            temp_mc_1_b = mc_1_b
            mc_1_b = temp_mc_1_b * mc_2_b - mc_1_a * mc_2_a
            mc_1_a = mc_1_a * mc_2_b + temp_mc_1_b * mc_2_a
    
    f1 = f0 + np.arange(nf) * df
    s1 = ((sc**2 * (n - sc2) + ss**2 * (n + sc2) - 2 * ss * sc * ss2) / (n**2 - sc2**2 - ss2**2))
    # conversion to amplitude spectrum (or power density or statistical distribution)
    if not np.isfinite(s1[0]):
        s1[0] = 0  # sometimes there can be a nan value
    # convert to the wanted normalisation
    if norm == 'distribution':  # statistical distribution
        s1 /= np.var(signal)
    elif norm == 'amplitude':  # amplitude spectrum
        s1 = np.sqrt(4 / n) * np.sqrt(s1)
    elif norm == 'density':  # power density
        s1 = (4 / n) * s1 * t_tot
    return f1, s1


@nb.njit()
def scargle_ampl_single(times, signal, f):
    """Amplitude at one frequency from the Scargle periodogram

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    f: float
        A single frequency
    
    Returns
    -------
    ampl: float
        Amplitude at the given frequency
    
    See Also
    --------
    scargle_ampl, scargle_phase, scargle_phase_single
    """
    two_pi = 2 * np.pi
    four_pi = 4 * np.pi
    # define tau
    cos_tau = 0
    sin_tau = 0
    for j in range(len(times)):
        cos_tau += np.cos(four_pi * f * times[j])
        sin_tau += np.sin(four_pi * f * times[j])
    tau = 1 / (four_pi * f) * np.arctan2(sin_tau, cos_tau)  # tau(f)
    # define the general cos and sin functions
    s_cos = 0
    cos_2 = 0
    s_sin = 0
    sin_2 = 0
    for j in range(len(times)):
        cos = np.cos(two_pi * f * (times[j] - tau))
        sin = np.sin(two_pi * f * (times[j] - tau))
        s_cos += signal[j] * cos
        cos_2 += cos**2
        s_sin += signal[j] * sin
        sin_2 += sin**2
    # final calculations
    a_cos_2 = s_cos**2 / cos_2
    b_sin_2 = s_sin**2 / sin_2
    # amplitude
    ampl = (a_cos_2 + b_sin_2) / 2
    ampl = np.sqrt(4 / len(times)) * np.sqrt(ampl)  #conversion to amplitude
    return ampl


@nb.njit()
def scargle_ampl(times, signal, fs):
    """Amplitude at one or a set of frequencies from the Scargle periodogram
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the tiScargle periodogram with no weights.

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    f0: float
        Starting frequency of the periodogram.
        If left zero, default is f0 = 1/(100*T)
    fn: float
        Last frequency of the periodogram.
        If left zero, default is fn = 1/(2*np.min(np.diff(times))) = Nyquist frequency
    df: float
        Frequency sampling space of the periodogram
        If left zero, default is df = 1/(10*T) = oversampling factor of ten (recommended)
    norm: str
        Normalisation of the periodogram. Choose from:
        'amplitude', 'density' or 'distribution'

    Returns
    -------
    f1: numpy.ndarray[float]
        Frequencies at which the periodogram was calculated
    s1: numpy.ndarray[float]
        The periodogram spectrum in the chosen unitsme series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    fs: numpy.ndarray[float]
        A set of frequencies
    
    Returns
    -------
    ampl: numpy.ndarray[float]
        Amplitude at the given frequencies
    
    See Also
    --------
    scargle_phase
    """
    two_pi = 2 * np.pi
    four_pi = 4 * np.pi
    fs = np.atleast_1d(fs)

    ampl = np.zeros(len(fs))
    for i in range(len(fs)):
        # define tau
        cos_tau = 0
        sin_tau = 0
        for j in range(len(times)):
            cos_tau += np.cos(four_pi * fs[i] * times[j])
            sin_tau += np.sin(four_pi * fs[i] * times[j])
        tau = 1 / (four_pi * fs[i]) * np.arctan2(sin_tau, cos_tau)  # tau(f)
        # define the general cos and sin functions
        s_cos = 0
        cos_2 = 0
        s_sin = 0
        sin_2 = 0
        for j in range(len(times)):
            cos = np.cos(two_pi * fs[i] * (times[j] - tau))
            sin = np.sin(two_pi * fs[i] * (times[j] - tau))
            s_cos += signal[j] * cos
            cos_2 += cos**2
            s_sin += signal[j] * sin
            sin_2 += sin**2
        # final calculations
        a_cos_2 = s_cos**2 / cos_2
        b_sin_2 = s_sin**2 / sin_2
        # amplitude
        ampl[i] = (a_cos_2 + b_sin_2) / 2
        ampl[i] = np.sqrt(4 / len(times)) * np.sqrt(ampl[i])  #conversion to amplitude
    return ampl


@nb.njit()
def scargle_phase_single(times, signal, f):
    """Phase at one frequency from the Scargle periodogram
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    f: float
        A single frequency
    
    Returns
    -------
    phi: float
        Phase at the given frequency
    
    See Also
    --------
    scargle_phase, scargle_ampl_single
    """
    two_pi = 2 * np.pi
    four_pi = 4 * np.pi
    # define tau
    cos_tau = 0
    sin_tau = 0
    for j in range(len(times)):
        cos_tau += np.cos(four_pi * f * times[j])
        sin_tau += np.sin(four_pi * f * times[j])
    tau = 1 / (four_pi * f) * np.arctan2(sin_tau, cos_tau)  # tau(f)
    # define the general cos and sin functions
    s_cos = 0
    cos_2 = 0
    s_sin = 0
    sin_2 = 0
    for j in range(len(times)):
        cos = np.cos(two_pi * f * (times[j] - tau))
        sin = np.sin(two_pi * f * (times[j] - tau))
        s_cos += signal[j] * cos
        cos_2 += cos**2
        s_sin += signal[j] * sin
        sin_2 += sin**2
    # final calculations
    a_cos = s_cos / cos_2**(1/2)
    b_sin = s_sin / sin_2**(1/2)
    # phase (radians)
    phi = np.pi/2 - np.arctan2(b_sin, a_cos) - two_pi * f * tau
    return phi


@nb.njit()
def scargle_phase(times, signal, fs):
    """Phase at one or a set of frequencies from the Scargle periodogram
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    fs: numpy.ndarray[float]
        A set of frequencies
    
    Returns
    -------
    phi: numpy.ndarray[float]
        Phase at the given frequencies
    
    Notes
    -----
    Uses a slightly modified version of the function in Hocke 1997
    ("Phase estimation with the Lomb-Scargle periodogram method")
    https://www.researchgate.net/publication/283359043_Phase_estimation_with_the_Lomb-Scargle_periodogram_method
    (only difference is an extra pi/2 for changing cos phase to sin phase)
    """
    two_pi = 2 * np.pi
    four_pi = 4 * np.pi
    fs = np.atleast_1d(fs)

    phi = np.zeros(len(fs))
    for i in range(len(fs)):
        # define tau
        cos_tau = 0
        sin_tau = 0
        for j in range(len(times)):
            cos_tau += np.cos(four_pi * fs[i] * times[j])
            sin_tau += np.sin(four_pi * fs[i] * times[j])
        tau = 1 / (four_pi * fs[i]) * np.arctan2(sin_tau, cos_tau)  # tau(f)
        # define the general cos and sin functions
        s_cos = 0
        cos_2 = 0
        s_sin = 0
        sin_2 = 0
        for j in range(len(times)):
            cos = np.cos(two_pi * fs[i] * (times[j] - tau))
            sin = np.sin(two_pi * fs[i] * (times[j] - tau))
            s_cos += signal[j] * cos
            cos_2 += cos**2
            s_sin += signal[j] * sin
            sin_2 += sin**2
        # final calculations
        a_cos = s_cos / cos_2**(1/2)
        b_sin = s_sin / sin_2**(1/2)
        # phase (radians)
        phi[i] = np.pi / 2 - np.arctan2(b_sin, a_cos) - two_pi * fs[i] * tau
    return phi


@nb.njit()
def calc_likelihood(residuals):
    """Natural logarithm of the likelihood function.
    
    Parameters
    ----------
    residuals: numpy.ndarray[float]
        Residual is signal - model
    
    Returns
    -------
    like: float
        Natural logarithm of the likelihood
    
    Notes
    -----
    Under the assumption that the errors are independent and identically distributed
    according to a normal distribution, the likelihood becomes:
    ln(L(θ)) = -n/2 (ln(2 pi σ^2) + 1)
    and σ^2 is estimated as σ^2 = sum((residuals)^2)/n
    """
    n = len(residuals)
    # like = -n / 2 * (np.log(2 * np.pi * np.sum(residuals**2) / n) + 1)
    # originally unjitted function, but for loop is quicker with numba
    sum_r_2 = 0
    for i, r in enumerate(residuals):
        sum_r_2 += r**2
    like = -n / 2 * (np.log(2 * np.pi * sum_r_2 / n) + 1)
    return like


@nb.njit()
def calc_bic(residuals, n_param):
    """Bayesian Information Criterion.
    
    Parameters
    ----------
    residuals: numpy.ndarray[float]
        Residual is signal - model
    n_param: int
        Number of free parameters in the model
    
    Returns
    -------
    bic: float
        Bayesian Information Criterion
    
    Notes
    -----
    BIC = −2 ln(L(θ)) + k ln(n)
    where L is the likelihood as function of the parameters θ, n the number of data points
    and k the number of free parameters.
    
    Under the assumption that the errors are independent and identically distributed
    according to a normal distribution, the likelihood becomes:
    ln(L(θ)) = -n/2 (ln(2 pi σ^2) + 1)
    and σ^2 is the error variance estimated as σ^2 = sum((residuals)^2)/n
    (residuals being data - model).
    
    Combining this gives:
    BIC = n ln(2 pi σ^2) + n + k ln(n)
    """
    n = len(residuals)
    # bic = n * np.log(2 * np.pi * np.sum(residuals**2) / n) + n + n_param * np.log(n)
    # originally jitted function, but with for loop is slightly quicker
    sum_r_2 = 0
    for i, r in enumerate(residuals):
        sum_r_2 += r**2
    bic = n * np.log(2 * np.pi * sum_r_2 / n) + n + n_param * np.log(n)
    return bic


@nb.njit()
def linear_curve(times, const, slope, i_sectors):
    """Returns a piece-wise linear curve for the given time points
    with slopes and y-intercepts.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    const: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    
    Returns
    -------
    curve: numpy.ndarray[float]
        The model time series of a (set of) straight line(s)
    """
    curve = np.zeros(len(times))
    for co, sl, s in zip(const, slope, i_sectors):
        curve[s[0]:s[1]] = co + sl * times[s[0]:s[1]]
    return curve


@nb.njit()
def linear_slope(times, signal, i_sectors):
    """Calculate the slope(s) and y-intercept(s) of a linear trend with the MLE.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    
    Returns
    -------
    y_inter: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    
    Notes
    -----
    Source: https://towardsdatascience.com/linear-regression-91eeae7d6a2e
    """
    y_inter = np.zeros(len(i_sectors))
    slope = np.zeros(len(i_sectors))
    for i, s in enumerate(i_sectors):
        t_avg = np.mean(times[s[0]:s[1]])
        s_avg = np.mean(signal[s[0]:s[1]])
        times_m_avg = (times[s[0]:s[1]] - t_avg)
        num = np.sum(times_m_avg * (signal[s[0]:s[1]] - s_avg))
        denom = np.sum(times_m_avg**2)
        slope[i] = num / denom
        y_inter[i] = s_avg - slope[i] * t_avg
    return y_inter, slope


@nb.njit()
def linear_slope_two_points(x1, y1, x2, y2):
    """Calculate the slope(s) and y-intercept(s) of a linear curve defined by two points.
    
    Parameters
    ----------
    x1: numpy.ndarray[float]
        The x-coordinate of the left point(s)
    y1: numpy.ndarray[float]
        The y-coordinate of the left point(s)
    x2: numpy.ndarray[float]
        The x-coordinate of the right point(s)
    y2: numpy.ndarray[float]
        The y-coordinate of the left point(s)
    
    Returns
    -------
    y_inter: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    """
    slope = (y2 - y1) / (x2 - x1)
    y_inter = y1 - slope * x1  # take point 1 to calculate y intercept
    return y_inter, slope


@nb.njit()
def sum_sines(times, f_n, a_n, ph_n):
    """A sum of sine waves at times t, given the frequencies, amplitudes and phases.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: list[float], numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: list[float], numpy.ndarray[float]
        The phases of a number of sine waves
    
    Returns
    -------
    model_sines: numpy.ndarray[float]
        Model time series of a sum of sine waves. Varies around 0.
    """
    model_sines = np.zeros(len(times))
    for f, a, ph in zip(f_n, a_n, ph_n):
        # model_sines += a * np.sin((2 * np.pi * f * times) + ph)
        # double loop runs a tad quicker when numba-jitted
        for i, t in enumerate(times):
            model_sines[i] += a * np.sin((2 * np.pi * f * t) + ph)
    return model_sines


@nb.njit()
def sum_sines_deriv(times, f_n, a_n, ph_n, deriv=1):
    """The derivative of a sum of sine waves at times t,
    given the frequencies, amplitudes and phases.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: list[float], numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: list[float], numpy.ndarray[float]
        The phases of a number of sine waves
    deriv: int
        Number of time derivatives taken (>= 1)
    
    Returns
    -------
    model_sines: numpy.ndarray[float]
        Model time series of a sum of sine wave derivatives. Varies around 0.
    """
    model_sines = np.zeros(len(times))
    mod_2 = deriv % 2
    mod_4 = deriv % 4
    ph_cos = (np.pi / 2) * mod_2  # alternate between cosine and sine
    sign = (-1)**((mod_4 - mod_2) // 2)  # (1, -1, -1, 1, 1, -1, -1... for deriv=1, 2, 3...)
    for f, a, ph in zip(f_n, a_n, ph_n):
        for i, t in enumerate(times):
            model_sines[i] += sign * (2 * np.pi * f)**deriv * a * np.sin((2 * np.pi * f * t) + ph + ph_cos)
    return model_sines


def sum_sines_damped(times, f_n, a_n, lifetimes, t_zeros):
    """A sum of damped sine waves at times t, given the frequencies, amplitudes,
    mode lifetimes and excitation times (t_zeros).

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    f_n: float, list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: float, list[float], numpy.ndarray[float]
        The amplitudes of a number of sine waves
    lifetimes: float, list[float], numpy.ndarray[float]
        The wave lifetimes of a number of sine waves
    t_zeros: float, list[float], numpy.ndarray[float]
        The starting (excitation) time of a number of sine waves
    
    Returns
    -------
    model_sines: numpy.ndarray[float]
        Model time series of a sum of damped sine waves. Varies around 0.
    """
    # with η1 the damping rate of the mode, which is the inverse of the mode lifetime.
    times = np.ascontiguousarray(np.atleast_1d(times)).reshape(-1, 1)  # reshape to enable the vector product in the sum
    f_n = np.atleast_1d(f_n)
    a_n = np.atleast_1d(a_n)
    lifetimes = np.atleast_1d(lifetimes)
    t_zeros = np.atleast_1d(t_zeros)
    eta = 1 / lifetimes  # η is the damping rate of the mode, which is the inverse of the mode lifetime
    t_shift = np.repeat(np.copy(times), len(eta), axis=1) - t_zeros  # make a separate matrix for the exponent
    mask = (t_shift < 0)  # now need to avoid positive exponent and make the wave zero before t_zero
    t_shift[mask] = 0
    exponent = np.exp(-eta * t_shift)
    exponent[mask] = 0
    model_sines = np.sum(a_n * np.sin((2 * np.pi * f_n * t_shift)) * exponent, axis=1)
    return model_sines


@nb.njit()
def formal_uncertainties(times, residuals, a_n, i_sectors):
    """Calculates the corrected uncorrelated (formal) uncertainties for the extracted
    parameters (constant, slope, frequencies, amplitudes and phases).
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    residuals: numpy.ndarray[float]
        Residual is signal - model
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    
    Returns
    -------
    sigma_const: numpy.ndarray[float]
        Uncertainty in the constant for each sector
    sigma_slope: numpy.ndarray[float]
        Uncertainty in the slope for each sector
    sigma_f: numpy.ndarray[float]
        Uncertainty in the frequency for each sine wave
    sigma_a: numpy.ndarray[float]
        Uncertainty in the amplitude for each sine wave (these are identical)
    sigma_ph: numpy.ndarray[float]
        Uncertainty in the phase for each sine wave
    
    Notes
    -----
    As in Aerts 2021, https://ui.adsabs.harvard.edu/abs/2021RvMP...93a5001A/abstract
    Errors in const and slope:
    https://pages.mtu.edu/~fmorriso/cm3215/UncertaintySlopeInterceptOfLeastSquaresFit.pdf
    """
    n_data = len(residuals)
    n_param = 2 + 3 * len(a_n)  # number of parameters in the model
    n_dof = n_data - n_param  # degrees of freedom
    # calculate the standard deviation of the residuals
    sum_r_2 = 0
    for r in residuals:
        sum_r_2 += r**2
    std = np.sqrt(sum_r_2 / n_dof)  # standard deviation of the residuals
    # calculate the D factor (square root of the average number of consecutive data points of the same sign)
    positive = (residuals > 0).astype(np.int_)
    indices = np.arange(n_data)
    zero_crossings = indices[1:][np.abs(positive[1:] - positive[:-1]).astype(np.bool_)]
    sss_i = np.concatenate((np.array([0]), zero_crossings, np.array([n_data])))  # same-sign sequence indices
    d_factor = np.sqrt(np.mean(np.diff(sss_i)))
    # uncertainty formulae for sinusoids
    sigma_f = d_factor * std * np.sqrt(6 / n_data) / (np.pi * a_n * np.ptp(times))
    sigma_a = d_factor * std * np.sqrt(2 / n_data)
    sigma_ph = d_factor * std * np.sqrt(2 / n_data) / a_n  # times 2 pi w.r.t. the paper
    # make an array of sigma_a (these are the same)
    sigma_a = np.full(len(a_n), sigma_a)
    # linear regression uncertainties
    sigma_const = np.zeros(len(i_sectors))
    sigma_slope = np.zeros(len(i_sectors))
    for i, s in enumerate(i_sectors):
        len_t = len(times[s[0]:s[1]])
        n_data = len(residuals[s[0]:s[1]])  # same as len_t, but just for the sake of clarity
        # standard deviation of the residuals but per sector
        sum_r_2 = 0
        for r in residuals[s[0]:s[1]]:
            sum_r_2 += r**2
        std = np.sqrt(sum_r_2 / n_dof)
        # some sums for the uncertainty formulae
        sum_t = 0
        for t in times[s[0]:s[1]]:
            sum_t += t
        ss_xx = 0
        for t in times[s[0]:s[1]]:
            ss_xx += (t - sum_t / len_t)**2
        sigma_const[i] = std * np.sqrt(1 / n_data + (sum_t / len_t)**2 / ss_xx)
        sigma_slope[i] = std / np.sqrt(ss_xx)
    return sigma_const, sigma_slope, sigma_f, sigma_a, sigma_ph


@nb.njit()
def formal_period_uncertainty(p_orb, f_n_err, harmonics, harmonic_n):
    """Calculates a formal error for the orbital period
    
    Parameters
    ----------
    p_orb: float
        The orbital period
    f_n_err: numpy.ndarray[float]
        Formal errors in the frequencies
    harmonics: numpy.ndarray[int]
        Indices of the orbital harmonics in the frequency list
    harmonic_n: numpy.ndarray[int]
        Integer indicating which harmonic each index in 'harmonics'
        points to. n=1 for the base frequency (=orbital frequency)
    
    Returns
    -------
    p_orb_err: float
        Uncertainty in the orbital period
    
    Notes
    -----
    Computes the error that one would obtain if the orbital period was calculated by
    the weighted average of the orbital harmonic frequencies.
    """
    # errors of the harmonics have to be scaled the same as the frequencies in a weighted average
    f_h_err = f_n_err[harmonics] / harmonic_n
    f_orb_err = np.sqrt(np.sum(1/f_h_err**2 / len(f_h_err)**2)) / np.sum(1/f_h_err**2/ len(f_h_err))
    # calculation of period error via relative error (same as p * f_err / f)
    p_orb_err = f_orb_err * p_orb**2
    return p_orb_err


def extract_single(times, signal, f0=0, fn=0, verbose=True):
    """Extract a single frequency from a time series using oversampling
    of the periodogram.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    f0: float
        Starting frequency of the periodogram.
        If left zero, default is f0 = 1/(100*T)
    fn: float
        Last frequency of the periodogram.
        If left zero, default is fn = 1/(2*np.min(np.diff(times))) = Nyquist frequency
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    f_final: float
        Frequency of the extracted sinusoid
    a_final: float
        Amplitude of the extracted sinusoid
    ph_final: float
        Phase of the extracted sinusoid
    
    See Also
    --------
    scargle, scargle_phase_single
    
    Notes
    -----
    The extracted frequency is based on the highest amplitude in the
    periodogram (over the interval where it is calculated). The highest
    peak is oversampled by a factor 10^4 to get a precise measurement.
    """
    df = 0.1 / np.ptp(times)
    freqs, ampls = scargle(times, signal, f0=f0, fn=fn, df=df)
    p1 = np.argmax(ampls)
    # check if we pick the boundary frequency
    if (p1 in [0, len(freqs) - 1]):
        if verbose:
            print(f'Edge of frequency range {freqs[p1]:1.6f} at position {p1} during extraction phase 1.')
        p1 = (p1 == 0) + (p1 == len(freqs) - 1) * (len(freqs) - 2)
    # now refine once by increasing the frequency resolution x100
    f_refine, a_refine = scargle(times, signal, f0=freqs[p1 - 1], fn=freqs[p1 + 1], df=df/100)
    p2 = np.argmax(a_refine)
    # check if we pick the boundary frequency
    if (p2 in [0, len(f_refine) - 1]):
        if verbose:
            print(f'Edge of frequency range {f_refine[p2]:1.6f} at position {p2} during extraction phase 2.')
        p2 = (p2 == 0) + (p2 == len(f_refine) - 1) * (len(f_refine) - 2)
    # now refine another time by increasing the frequency resolution x100 again
    f_refine_2, a_refine_2 = scargle(times, signal, f0=f_refine[p2 - 1], fn=f_refine[p2 + 1], df=df/10000)
    p3 = np.argmax(a_refine_2)
    # check if we pick the boundary frequency
    if (p3 in [0, len(f_refine_2) - 1]):
        if verbose:
            print(f'Edge of frequency range {f_refine_2[p3]:1.6f} at position {p3} during extraction phase 3.')
    f_final = f_refine_2[p3]
    a_final = a_refine_2[p3]
    # finally, compute the phase (and make sure it stays within + and - pi)
    ph_final = scargle_phase_single(times, signal, f_final)
    ph_final = (ph_final + np.pi) % (2 * np.pi) - np.pi
    return f_final, a_final, ph_final


def extract_single_harmonics(times, signal, p_orb, f0=0, fn=0, verbose=True):
    """Extract a single frequency from a time series using oversampling
    of the periodogram and avoiding harmonics.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    f0: float
        Starting frequency of the periodogram.
        If left zero, default is f0 = 1/(100*T)
    fn: float
        Last frequency of the periodogram.
        If left zero, default is fn = 1/(2*np.min(np.diff(times))) = Nyquist frequency
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    f_final: float
        Frequency of the extracted sinusoid
    a_final: float
        Amplitude of the extracted sinusoid
    ph_final: float
        Phase of the extracted sinusoid
    
    See Also
    --------
    scargle, scargle_phase_single
    
    Notes
    -----
    The extracted frequency is based on the highest amplitude in the
    periodogram (over the interval where it is calculated). The highest
    peak is oversampled by a factor 10^4 to get a precise measurement.
    """
    freq_res = 1.5 / np.ptp(times)
    avoid = freq_res / (np.ptp(times) / p_orb)  # avoidance zone around harmonics
    df = 0.1 / np.ptp(times)
    freqs, ampls = scargle(times, signal, f0=f0, fn=fn, df=df)
    mask = (freqs % p_orb > p_orb - avoid / 2) | (freqs % p_orb < avoid / 2)
    p1 = np.argmax(ampls[mask])
    # check if we pick the boundary frequency (does not take into account masked positions)
    if (p1 in [0, len(freqs[mask]) - 1]):
        if verbose:
            print(f'Edge of frequency range {freqs[mask][p1]:1.6f} at position {p1} during extraction phase 1.')
        p1 = (p1 == 0) + (p1 == len(freqs[mask]) - 1) * (len(freqs[mask]) - 2)
    # now refine once by increasing the frequency resolution x100
    f_refine, a_refine = scargle(times, signal, f0=freqs[mask][p1 - 1], fn=freqs[mask][p1 + 1], df=df/100)
    p2 = np.argmax(a_refine)
    # check if we pick the boundary frequency
    if (p2 in [0, len(f_refine) - 1]):
        if verbose:
            print(f'Edge of frequency range {f_refine[p2]:1.6f} at position {p2} during extraction phase 2.')
        p2 = (p2 == 0) + (p2 == len(f_refine) - 1) * (len(f_refine) - 2)
    # now refine another time by increasing the frequency resolution x100 again
    f_refine_2, a_refine_2 = scargle(times, signal, f0=f_refine[p2 - 1], fn=f_refine[p2 + 1], df=df/10000)
    p3 = np.argmax(a_refine_2)
    # check if we pick the boundary frequency
    if (p3 in [0, len(f_refine_2) - 1]):
        if verbose:
            print(f'Edge of frequency range {f_refine_2[p3]:1.6f} at position {p3} during extraction phase 3.')
    f_final = f_refine_2[p3]
    a_final = a_refine_2[p3]
    # finally, compute the phase (and make sure it stays within + and - pi)
    ph_final = scargle_phase_single(times, signal, f_final)
    ph_final = (ph_final + np.pi) % (2 * np.pi) - np.pi
    return f_final, a_final, ph_final


def refine_subset(times, signal, close_f, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Refine a subset of frequencies that are within the Rayleigh criterion of each other.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    close_f: list[int], numpy.ndarray[int]
        Indices of the subset of frequencies to be refined
    const: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: list[float], numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: list[float], numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    const: numpy.ndarray[float]
        Updated y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        Updated slope(s) of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        Updated frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        Updated amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        Updated phases of a number of sine waves
    
    See Also
    --------
    extract_all
    
    Notes
    -----
    Intended as a sub-loop within another extraction routine (extract_all),
    can work standalone too.
    """
    freq_res = 1.5 / np.ptp(times)  # frequency resolution
    n_sectors = len(i_sectors)
    n_f = len(f_n)
    # determine initial bic
    model = linear_curve(times, const, slope, i_sectors)  # the linear part of the model
    model += sum_sines(times, f_n, a_n, ph_n)  # the sinusoid part of the model
    resid = signal - model
    f_n_temp, a_n_temp, ph_n_temp = np.copy(f_n), np.copy(a_n), np.copy(ph_n)
    n_param = 2 * n_sectors + 3 * n_f
    bic_prev = np.inf
    bic = calc_bic(resid, n_param)
    # stop the loop when the BIC increases
    i = 0
    while (np.round(bic_prev - bic, 2) > 0):
        # last frequencies are accepted
        f_n, a_n, ph_n = np.copy(f_n_temp), np.copy(a_n_temp), np.copy(ph_n_temp)
        bic_prev = bic
        if verbose:
            print(f'Refining iteration {i}, {n_f} frequencies, BIC= {bic:1.2f}')
        # remove each frequency one at a time to re-extract them
        for j in close_f:
            model = linear_curve(times, const, slope, i_sectors)  # the linear part of the model
            model += sum_sines(times, np.delete(f_n_temp, j), np.delete(a_n_temp, j),
                                    np.delete(ph_n_temp, j))  # the sinusoid part of the model
            resid = signal - model
            f_j, a_j, ph_j = extract_single(times, resid, f0=f_n_temp[j] - freq_res, fn=f_n_temp[j] + freq_res,
                                            verbose=verbose)
            f_n_temp[j], a_n_temp[j], ph_n_temp[j] = f_j, a_j, ph_j
        # as a last model-refining step, redetermine the constant and slope
        model = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
        const, slope = linear_slope(times, signal - model, i_sectors)
        model += linear_curve(times, const, slope, i_sectors)  # the linear part of the model
        # now subtract all from the signal and calculate BIC before moving to the next iteration
        resid = signal - model
        bic = calc_bic(resid, n_param)
        i += 1
    if verbose:
        print(f'Refining terminated. Iteration {i} not included with BIC= {bic:1.2f}, '
              f'delta-BIC= {bic_prev - bic:1.2f}')
    # redo the constant and slope without the last iteration of changes
    resid = signal - sum_sines(times, f_n, a_n, ph_n)
    const, slope = linear_slope(times, resid, i_sectors)
    return const, slope, f_n, a_n, ph_n


def refine_subset_harmonics(times, signal, close_f, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Refine a subset of frequencies that are within the Rayleigh criterion of each other.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    close_f: list[int], numpy.ndarray[int]
        Indices of the subset of frequencies to be refined
    p_orb: float
        The orbital period
    const: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: list[float], numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: list[float], numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    const: numpy.ndarray[float]
        Updated y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        Updated slope(s) of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        Updated frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        Updated amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        Updated phases of a number of sine waves
    
    See Also
    --------
    extract_all
    
    Notes
    -----
    Intended as a sub-loop within another extraction routine (extract_all),
    can work standalone too.
    """
    freq_res = 1.5 / np.ptp(times)  # frequency resolution
    n_sectors = len(i_sectors)
    n_f = len(f_n)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    # determine initial bic
    model = linear_curve(times, const, slope, i_sectors)  # the linear part of the model
    model += sum_sines(times, f_n, a_n, ph_n)  # the sinusoid part of the model
    resid = signal - model
    f_n_temp, a_n_temp, ph_n_temp = np.copy(f_n), np.copy(a_n), np.copy(ph_n)
    n_param = 2 * n_sectors + 3 * n_f
    bic_prev = np.inf
    bic = calc_bic(resid, n_param)
    # stop the loop when the BIC increases
    i = 0
    while (np.round(bic_prev - bic, 2) > 0):
        # last frequencies are accepted
        f_n, a_n, ph_n = np.copy(f_n_temp), np.copy(a_n_temp), np.copy(ph_n_temp)
        bic_prev = bic
        if verbose:
            print(f'Refining iteration {i}, {n_f} frequencies, BIC= {bic:1.2f}')
        # remove each frequency one at a time to re-extract them
        for j in close_f:
            model = linear_curve(times, const, slope, i_sectors)  # the linear part of the model
            model += sum_sines(times, np.delete(f_n_temp, j), np.delete(a_n_temp, j),
                               np.delete(ph_n_temp, j))  # the sinusoid part of the model
            resid = signal - model
            # if f is a harmonic, don't shift the frequency
            if j in harmonics:
                # f_j = f_n_temp[j]
                # a_j = scargle_ampl_single(times, resid, f_j)
                # ph_j = scargle_phase_single(times, resid, f_j)
                f_j = f_n_temp[j]
                a_j = a_n_temp[j]
                ph_j = ph_n_temp[j]
            else:
                f_j, a_j, ph_j = extract_single(times, resid, f0=f_n_temp[j] - freq_res, fn=f_n_temp[j] + freq_res,
                                                verbose=verbose)
            f_n_temp[j], a_n_temp[j], ph_n_temp[j] = f_j, a_j, ph_j
        # as a last model-refining step, redetermine the constant and slope
        model = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
        const, slope = linear_slope(times, signal - model, i_sectors)
        model += linear_curve(times, const, slope, i_sectors)  # the linear part of the model
        # now subtract all from the signal and calculate BIC before moving to the next iteration
        resid = signal - model
        bic = calc_bic(resid, n_param)
        i += 1
    if verbose:
        print(f'Refining terminated. Iteration {i} not included with BIC= {bic:1.2f}, '
              f'delta-BIC= {bic_prev - bic:1.2f}')
    # redo the constant and slope without the last iteration of changes
    resid = signal - sum_sines(times, f_n, a_n, ph_n)
    const, slope = linear_slope(times, resid, i_sectors)
    return const, slope, f_n, a_n, ph_n


def extract_all(times, signal, i_sectors, verbose=True):
    """Extract all the frequencies from a periodic signal.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    const: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    
    Notes
    -----
    Spits out frequencies and amplitudes in the same units as the input,
    and phases that are measured with respect to the first time point.
    Also determines the signal average, so this does not have to be subtracted
    before input into this function.
    Note: does not perform a non-linear least-squares fit at the end,
    which is highly recommended! (In fact, no fitting is done at all).
    
    i_sectors is a 2D array with start and end indices of each (half) sector.
    This is used to model a piecewise-linear trend in the data.
    If you have no sectors like the TESS mission does, set
    i_sectors = np.array([[0, len(times)]])
    
    Exclusively uses the Lomb-Scargle periodogram (and an iterative parameter
    improvement scheme) to extract the frequencies.
    Uses a delta BIC > 2 stopping criterion.
    """
    times -= times[0]  # shift reference time to times[0]
    freq_res = 1.5 / np.ptp(times)  # frequency resolution
    n_sectors = len(i_sectors)
    # constant term (or y-intercept) and slope
    const, slope = linear_slope(times, signal, i_sectors)
    resid = signal - linear_curve(times, const, slope, i_sectors)
    f_n_temp, a_n_temp, ph_n_temp = np.array([[], [], []])
    n_param = 2 * n_sectors
    bic_prev = np.inf  # initialise previous BIC to infinity
    bic = calc_bic(resid, n_param)  # initialise current BIC to the mean (and slope) subtracted signal
    # stop the loop when the BIC decreases by less than 2 (or increases)
    i = 0
    while (bic_prev - bic > 2):
        # last frequency is accepted
        f_n, a_n, ph_n = f_n_temp, a_n_temp, ph_n_temp
        bic_prev = bic
        if verbose:
            print(f'Iteration {i}, {len(f_n)} frequencies, BIC= {bic:1.2f}')
        # attempt to extract the next frequency
        f_i, a_i, ph_i = extract_single(times, resid, verbose=verbose)
        f_n_temp, a_n_temp, ph_n_temp = np.append(f_n_temp, f_i), np.append(a_n_temp, a_i), np.append(ph_n_temp, ph_i)
        # now iterate over close frequencies (around f_i) a number of times to improve them
        close_f = af.f_within_rayleigh(i, f_n_temp, freq_res)
        if (i > 0) & (len(close_f) > 1):
            refine_out = refine_subset(times, signal, close_f, const, slope, f_n_temp, a_n_temp, ph_n_temp, i_sectors,
                                       verbose=verbose)
            refine_out = const, slope, f_n_temp, a_n_temp, ph_n_temp
        # as a last model-refining step, redetermine the constant and slope
        model = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
        const, slope = linear_slope(times, signal - model, i_sectors)
        model += linear_curve(times, const, slope, i_sectors)  # the linear part of the model
        # now subtract all from the signal and calculate BIC before moving to the next iteration
        resid = signal - model
        n_param = 2 * n_sectors + 3 * len(f_n_temp)
        bic = calc_bic(resid, n_param)
        i += 1
    if verbose:
        print(f'Extraction terminated. Iteration {i} not included with BIC= {bic:1.2f}, '
              f'delta-BIC= {bic_prev - bic:1.2f}')
    # redo the constant and slope without the last iteration frequencies
    resid = signal - sum_sines(times, f_n, a_n, ph_n)
    const, slope = linear_slope(times, resid, i_sectors)
    return const, slope, f_n, a_n, ph_n


def extract_additional_frequencies(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=True):
    """Extract additional frequencies starting from an existing set.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        The orbital period
    const: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: list[float], numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: list[float], numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    const: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    
    Notes
    -----
    Spits out frequencies and amplitudes in the same units as the input,
    and phases that are measured with respect to the first time point.
    Also determines the signal average, so this does not have to be subtracted
    before input into this function.
    Note: does not perform a non-linear least-squares fit at the end,
    which is highly recommended! (In fact, no fitting is done at all).
    
    i_sectors is a 2D array with start and end indices of each (half) sector.
    This is used to model a piecewise-linear trend in the data.
    If you have no sectors like the TESS mission does, set
    i_sectors = np.array([[0, len(times)]])
    
    Exclusively uses the Lomb-Scargle periodogram (and an iterative parameter
    improvement scheme) to extract the frequencies.
    Uses a delta BIC > 2 stopping criterion.
    """
    times -= times[0]  # shift reference time to times[0]
    freq_res = 1.5 / np.ptp(times)  # frequency resolution
    indices = np.arange(len(f_n))
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    n_sectors = len(i_sectors)
    n_harmonics = len(harmonics)
    # constant term (or y-intercept) and slope
    model = linear_curve(times, const, slope, i_sectors)
    model += sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model
    f_n_temp, a_n_temp, ph_n_temp = np.copy(f_n), np.copy(a_n), np.copy(ph_n)
    n_param = 2 * n_sectors + 1 + 2 * n_harmonics + 3 * (len(f_n) - n_harmonics)
    bic_prev = np.inf  # initialise previous BIC to infinity
    bic = calc_bic(resid, n_param)  # current BIC
    # stop the loop when the BIC decreases by less than 2 (or increases)
    i = 0
    while (bic_prev - bic > 2):
        # last frequency is accepted
        f_n, a_n, ph_n = f_n_temp, a_n_temp, ph_n_temp
        bic_prev = bic
        if verbose:
            print(f'Iteration {i}, {len(f_n)} frequencies, BIC= {bic:1.2f}')
        # attempt to extract the next frequency
        f_i, a_i, ph_i = extract_single_harmonics(times, resid, p_orb, verbose=verbose)
        f_n_temp, a_n_temp, ph_n_temp = np.append(f_n_temp, f_i), np.append(a_n_temp, a_i), np.append(ph_n_temp, ph_i)
        # now iterate over close frequencies (around f_i) a number of times to improve them
        close_f = af.f_within_rayleigh(i, f_n_temp, freq_res)
        if (i > 0) & (len(close_f) > 1):
            refine_out = refine_subset_harmonics(times, signal, close_f, p_orb, const, slope, f_n_temp, a_n_temp,
                                                 ph_n_temp, i_sectors, verbose=verbose)
            const, slope, f_n_temp, a_n_temp, ph_n_temp = refine_out
        # as a last model-refining step, redetermine the constant and slope
        model = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
        const, slope = linear_slope(times, signal - model, i_sectors)
        model += linear_curve(times, const, slope, i_sectors)  # the linear part of the model
        # now subtract all from the signal and calculate BIC before moving to the next iteration
        resid = signal - model
        n_param = 2 * n_sectors + 1 + 2 * n_harmonics + 3 * (len(f_n_temp) - n_harmonics)
        bic = calc_bic(resid, n_param)
        i += 1
    if verbose:
        print(f'Extraction terminated. Iteration {i} not included with BIC= {bic:1.2f}, '
              f'delta-BIC= {bic_prev - bic:1.2f}')
    # redo the constant and slope without the last iteration frequencies
    resid = signal - sum_sines(times, f_n, a_n, ph_n)
    const, slope = linear_slope(times, resid, i_sectors)
    return const, slope, f_n, a_n, ph_n


def extract_additional_harmonics(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Tries to extract more harmonics from the signal
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        The orbital period
    const: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: list[float], numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: list[float], numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    const: numpy.ndarray[float]
        (Updated) y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        (Updated) slope(s) of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        (Updated) frequencies of a (higher) number of sine waves
    a_n: numpy.ndarray[float]
        (Updated) amplitudes of a (higher) number of sine waves
    ph_n: numpy.ndarray[float]
        (Updated) phases of a (higher) number of sine waves
    
    See Also
    --------
    extract_harmonic_pattern, measure_harmonic_period, fix_harmonic_frequency
    
    Notes
    -----
    Looks for missing harmonics and checks whether adding them
    decreases the BIC sufficiently (by more than 2).
    Assumes the harmonics are already fixed multiples of 1/p_orb
    as achieved with the functions mentioned in the see also section.
    """
    f_max = 1 / (2 * np.min(times[1:] - times[:-1]))  # Nyquist freq
    # extract the harmonics using the period
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    if (len(harmonics) == 0):
        raise ValueError('No harmonic frequencies found')
    # make a list of not-present possible harmonics
    h_candidate = np.arange(1, p_orb * f_max, dtype=int)
    h_candidate = np.delete(h_candidate, harmonic_n - 1)  # harmonic_n minus one is the position
    # initial residuals
    model = linear_curve(times, const, slope, i_sectors)  # the linear part of the model
    model += sum_sines(times, f_n, a_n, ph_n)  # the sinusoid part of the model
    resid = signal - model
    n_param_orig = 3 * len(f_n) + 2 - len(harmonics) + 1  # harmonics have 1 less free parameter
    bic_prev = calc_bic(resid, n_param_orig)
    # loop over candidates and try to extract
    n_accepted = 0
    for h_c in h_candidate:
        f_c = h_c / p_orb
        a_c = scargle_ampl_single(times, resid, f_c)
        ph_c = scargle_phase_single(times, resid, f_c)
        # make sure the phase stays within + and - pi
        ph_c = np.mod(ph_c + np.pi, 2 * np.pi) - np.pi
        # add to temporary parameters
        f_n_temp, a_n_temp, ph_n_temp = np.append(f_n, f_c), np.append(a_n, a_c), np.append(ph_n, ph_c)
        # redetermine the constant and slope
        model = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)
        const, slope = linear_slope(times, signal - model, i_sectors)
        # determine new BIC and whether it improved
        model = linear_curve(times, const, slope, i_sectors)  # the linear part of the model
        model += sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
        resid = signal - model
        n_param = n_param_orig + 2 * (n_accepted + 1)
        bic = calc_bic(resid, n_param)
        if (np.round(bic_prev - bic, 2) > 2):
            # h_c is accepted, add it to the final list and continue
            bic_prev = bic
            f_n, a_n, ph_n = np.copy(f_n_temp), np.copy(a_n_temp), np.copy(ph_n_temp)
            n_accepted += 1
            if verbose:
                print(f'Succesfully extracted harmonic {h_c}, BIC= {bic:1.2f}')
        else:
            # h_c is rejected, revert to previous residual
            resid = signal - sum_sines(times, f_n, a_n, ph_n)
            const, slope = linear_slope(times, resid, i_sectors)
            model = linear_curve(times, const, slope, i_sectors)  # the linear part of the model
            model += sum_sines(times, f_n, a_n, ph_n)  # the sinusoid part of the model
            resid = signal - model
    return const, slope, f_n, a_n, ph_n


def extract_residual_harmonics(times, signal, p_orb, t_zero, const, slope, f_n, a_n, ph_n, i_sectors, ellc_par,
                               timings, verbose=False):
    """Tries to extract harmonics from the signal after subtraction of
    an eclipse model
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        The orbital period
    t_zero: float
        Time of deepest minimum modulo p_orb
    const: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: list[float], numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: list[float], numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    ellc_par: numpy.ndarray[float]
        Parameters of the best ellc model, consisting of:
        [sqrt(e)cos(w), sqrt(e)sin(w), i, (r1+r2)/a, r2/r1, sb2/sb1]
    verbose: bool
        If set to True, this function will print some information
    
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
    
    Notes
    -----
    Looks for missing harmonics and checks whether adding them
    decreases the BIC sufficiently (by more than 2).
    Assumes the harmonics are fixed multiples of 1/p_orb.
    """
    f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset = ellc_par
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_nh = sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_line = linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line
    # initial eclipse model
    model_ellc = tsfit.ellc_lc_simple(times, p_orb, t_zero, f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset)
    # extract residual harmonics
    f_max = 1 / (2 * np.min(times[1:] - times[:-1]))  # Nyquist freq
    # make a list of not-present possible harmonics
    h_candidate = np.arange(1, p_orb * f_max, dtype=int)
    # initial residuals
    resid_orig = ecl_signal - model_ellc
    resid_orig_mean = np.mean(resid_orig)  # mean difference between the ellc model and harmonics
    resid_orig = resid_orig - resid_orig_mean
    resid = np.copy(resid_orig)
    f_n_r, a_n_r, ph_n_r = np.array([[], [], []])
    n_param_orig = 7  # the eclipse parameters for ellc plus the mean of the residual
    bic_prev = calc_bic(resid, n_param_orig)
    # loop over candidates and try to extract
    n_accepted = 0
    for h_c in h_candidate:
        f_c = h_c / p_orb
        a_c = scargle_ampl_single(times, resid, f_c)
        ph_c = scargle_phase_single(times, resid, f_c)
        # make sure the phase stays within + and - pi
        ph_c = np.mod(ph_c + np.pi, 2 * np.pi) - np.pi
        # add to temporary parameters
        f_n_temp, a_n_temp, ph_n_temp = np.append(f_n_r, f_c), np.append(a_n_r, a_c), np.append(ph_n_r, ph_c)
        # determine new BIC and whether it improved
        model = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
        resid = resid_orig - model - np.mean(resid_orig - model)
        n_param = n_param_orig + 2 * (n_accepted + 1)
        bic = calc_bic(resid, n_param)
        if (np.round(bic_prev - bic, 2) > 2):
            # h_c is accepted, add it to the final list and continue
            bic_prev = bic
            f_n_r, a_n_r, ph_n_r = np.copy(f_n_temp), np.copy(a_n_temp), np.copy(ph_n_temp)
            n_accepted += 1
            if verbose:
                print(f'Succesfully extracted harmonic {h_c}, BIC= {bic:1.2f}')
        else:
            # h_c is rejected, revert to previous residual
            model = sum_sines(times, f_n_r, a_n_r, ph_n_r)  # the sinusoid part of the model
            resid = resid_orig - model - np.mean(resid_orig - model)
    const_r = np.mean(resid_orig - model) + resid_orig_mean  # return constant for last model plus initial diff
    return const_r, f_n_r, a_n_r, ph_n_r


def reduce_frequencies(times, signal, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Attempt to reduce the number of frequencies.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    const: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: list[float], numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: list[float], numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    const: numpy.ndarray[float]
        (Updated) y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        (Updated) slope(s) of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        (Updated) frequencies of a (lower) number of sine waves
    a_n: numpy.ndarray[float]
        (Updated) amplitudes of a (lower) number of sine waves
    ph_n: numpy.ndarray[float]
        (Updated) phases of a (lower) number of sine waves
    
    Notes
    -----
    Checks whether the BIC can be inproved by removing a frequency. Special attention
    is given to frequencies that are within the Rayleigh criterion of each other.
    It is attempted to replace these by a single frequency.
    """
    freq_res = 1.5 / np.ptp(times)  # frequency resolution
    n_freq = np.arange(len(f_n))
    n_sectors = len(i_sectors)
    # first check if any one frequency can be left out (after the fit, this may be possible)
    remove_single = np.zeros(0, dtype=int)  # single frequencies to remove
    # determine initial bic
    model = linear_curve(times, const, slope, i_sectors)  # the linear part of the model
    model += sum_sines(times, f_n, a_n, ph_n)  # the sinusoid part of the model
    n_param = 2 * n_sectors + 3 * len(f_n)
    bic_init = calc_bic(signal - model, n_param)
    bic_prev = bic_init
    n_prev = -1
    # while frequencies are added to the remove list, continue loop
    while (len(remove_single) > n_prev):
        n_prev = len(remove_single)
        for i in n_freq:
            if i not in remove_single:
                # temporary arrays for this iteration (remove freqs, remove current freq)
                remove = np.append(remove_single, i)
                f_n_temp = np.delete(f_n, remove)
                a_n_temp = np.delete(a_n, remove)
                ph_n_temp = np.delete(ph_n, remove)
                # make a model not including the freq of this iteration
                model = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
                const, slope = linear_slope(times, signal - model, i_sectors)  # redetermine const and slope
                model += linear_curve(times, const, slope, i_sectors)  # the linear part of the model
                n_param = 2 * n_sectors + 3 * len(f_n_temp)
                bic = calc_bic(signal - model, n_param)
                if (np.round(bic_prev - bic, 2) > 0):
                    # add to list of removed freqs
                    remove_single = np.append(remove_single, [i])
                    bic_prev = bic
    f_n = np.delete(f_n, remove_single)
    a_n = np.delete(a_n, remove_single)
    ph_n = np.delete(ph_n, remove_single)
    if verbose:
        print(f'Single frequencies removed: {len(remove_single)}. BIC= {bic_prev:1.2f}')
    # Now go on to trying to replace sets of frequencies that are close together (first without harmonics)
    # make an array of sets of frequencies to be investigated for replacement
    close_f_g = af.chains_within_rayleigh(f_n, freq_res)
    f_sets = [g[np.arange(p1, p2 + 1)] for g in close_f_g for p1 in range(len(g) - 1) for p2 in range(p1 + 1, len(g))]
    s_indices = np.arange(len(f_sets))
    remove_sets = np.zeros(0, dtype=int)  # sets of frequencies to replace (by 1 freq)
    used_sets = np.zeros(0, dtype=int)  # sets that are not to be examined anymore
    f_new, a_new, ph_new = np.zeros((3, 0))
    n_prev = -1
    # while frequencies are added to the remove list, continue loop
    while (len(remove_sets) > n_prev):
        n_prev = len(remove_sets)
        for i, set_i in enumerate(f_sets):
            if i not in used_sets:
                # temporary arrays for this iteration (remove combos, remove current set, add new freqs)
                remove = np.append([k for j in remove_sets for k in f_sets[j]], set_i).astype(int)
                f_n_temp = np.append(np.delete(f_n, remove), f_new)
                a_n_temp = np.append(np.delete(a_n, remove), a_new)
                ph_n_temp = np.append(np.delete(ph_n, remove), ph_new)
                # make a model not including the freqs of this iteration
                model = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
                const, slope = linear_slope(times, signal - model, i_sectors)  # redetermine const and slope
                model += linear_curve(times, const, slope, i_sectors)  # the linear part of the model
                # extract a single freq to try replacing the pair (set)
                edges = [min(f_n[set_i]) - freq_res, max(f_n[set_i]) + freq_res]
                f_i, a_i, ph_i = extract_single(times, signal - model, f0=edges[0], fn=edges[1], verbose=verbose)
                # make a model including the new freq
                model = sum_sines(times, np.append(f_n_temp, f_i), np.append(a_n_temp, a_i),
                                  np.append(ph_n_temp, ph_i))  # the sinusoid part of the model
                const, slope = linear_slope(times, signal - model, i_sectors)  # redetermine const and slope
                model += linear_curve(times, const, slope, i_sectors)  # the linear part of the model
                # calculate bic
                n_param = 2 * n_sectors + 3 * len(f_n_temp)
                bic = calc_bic(signal - model, n_param)
                if (np.round(bic_prev - bic, 2) > 0):
                    # add to list of removed sets
                    remove_sets = np.append(remove_sets, [i])
                    # do not look at sets with the same freqs as the just removed set anymore
                    overlap = s_indices[[np.any([j in set_i for j in subset]) for subset in f_sets]]
                    used_sets = np.unique(np.append(used_sets, [overlap]))
                    # remember the new frequency
                    f_new, a_new, ph_new = np.append(f_new, [f_i]), np.append(a_new, [a_i]), np.append(ph_new, [ph_i])
                    bic_prev = bic
    f_n = np.append(np.delete(f_n, [k for i in remove_sets for k in f_sets[i]]), f_new)
    a_n = np.append(np.delete(a_n, [k for i in remove_sets for k in f_sets[i]]), a_new)
    ph_n = np.append(np.delete(ph_n, [k for i in remove_sets for k in f_sets[i]]), ph_new)
    if verbose:
        n_f_removed = len([k for i in remove_sets for k in f_sets[i]])
        print(f'Frequency sets replaced by a single frequency: {len(remove_sets)} ({n_f_removed} frequencies). '
              f'BIC= {bic_prev:1.2f}')
    # lastly re-determine slope and const
    model = sum_sines(times, f_n, a_n, ph_n)  # the sinusoid part of the model
    const, slope = linear_slope(times, signal - model, i_sectors)
    return const, slope, f_n, a_n, ph_n


def reduce_frequencies_harmonics(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Attempt to reduce the number of frequencies taking into
    account harmonics.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        The orbital period
    const: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: list[float], numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: list[float], numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    const: numpy.ndarray[float]
        (Updated) y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        (Updated) slope(s) of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        (Updated) frequencies of a (lower) number of sine waves
    a_n: numpy.ndarray[float]
        (Updated) amplitudes of a (lower) number of sine waves
    ph_n: numpy.ndarray[float]
        (Updated) phases of a (lower) number of sine waves
    
    Notes
    -----
    Checks whether the BIC can be inproved by removing a frequency. Special attention
    is given to frequencies that are within the Rayleigh criterion of each other.
    It is attempted to replace these by a single frequency.
    Harmonics are not removed (amplitude/phase can be updated).
    """
    freq_res = 1.5 / np.ptp(times)  # frequency resolution
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    n_sectors = len(i_sectors)
    n_harm = len(harmonics)
    # first check if any one frequency can be left out (after the fit, this may be possible)
    remove_single = np.zeros(0, dtype=int)  # single frequencies to remove
    # determine initial bic
    model = linear_curve(times, const, slope, i_sectors)  # the linear part of the model
    model += sum_sines(times, f_n, a_n, ph_n)  # the sinusoid part of the model
    n_param = 2 * n_sectors + 1 + 2 * n_harm + 3 * (len(f_n) - n_harm)
    bic_init = calc_bic(signal - model, n_param)
    bic_prev = bic_init
    n_prev = -1
    # while frequencies are added to the remove list, continue loop
    while (len(remove_single) > n_prev):
        n_prev = len(remove_single)
        for i in non_harm:
            if i not in remove_single:
                # temporary arrays for this iteration (remove freqs, remove current freq)
                remove = np.append(remove_single, i)
                f_n_temp = np.delete(f_n, remove)
                a_n_temp = np.delete(a_n, remove)
                ph_n_temp = np.delete(ph_n, remove)
                # make a model not including the freq of this iteration
                model = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
                const, slope = linear_slope(times, signal - model, i_sectors)  # redetermine const and slope
                model += linear_curve(times, const, slope, i_sectors)  # the linear part of the model
                n_param = 2 * n_sectors + 1 + 2 * n_harm + 3 * (len(f_n_temp) - n_harm)
                bic = calc_bic(signal - model, n_param)
                if (np.round(bic_prev - bic, 2) > 0):
                    # add to list of removed freqs
                    remove_single = np.append(remove_single, [i])
                    bic_prev = bic
    f_n = np.delete(f_n, remove_single)
    a_n = np.delete(a_n, remove_single)
    ph_n = np.delete(ph_n, remove_single)
    if verbose:
        print(f'Single frequencies removed: {len(remove_single)}. BIC= {bic_prev:1.2f}')
    # Now go on to trying to replace sets of frequencies that are close together (first without harmonics)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    # make an array of sets of frequencies to be investigated for replacement
    close_f_g = af.chains_within_rayleigh(f_n[non_harm], freq_res)
    f_sets = [g[np.arange(p1, p2 + 1)] for g in close_f_g for p1 in range(len(g) - 1) for p2 in range(p1 + 1, len(g))]
    s_indices = np.arange(len(f_sets))
    remove_sets = np.zeros(0, dtype=int)  # sets of frequencies to replace (by 1 freq)
    used_sets = np.zeros(0, dtype=int)  # sets that are not to be examined anymore
    f_new, a_new, ph_new = np.zeros((3, 0))
    n_prev = -1
    # while frequencies are added to the remove list, continue loop
    while (len(remove_sets) > n_prev):
        n_prev = len(remove_sets)
        for i, set_i in enumerate(f_sets):
            if i not in used_sets:
                # temporary arrays for this iteration (remove combos, remove current set, add new freqs)
                remove = np.append([k for j in remove_sets for k in f_sets[j]], set_i).astype(int)
                f_n_temp = np.append(np.delete(f_n, remove), f_new)
                a_n_temp = np.append(np.delete(a_n, remove), a_new)
                ph_n_temp = np.append(np.delete(ph_n, remove), ph_new)
                # make a model not including the freqs of this iteration
                model = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
                const, slope = linear_slope(times, signal - model, i_sectors)  # redetermine const and slope
                model += linear_curve(times, const, slope, i_sectors)  # the linear part of the model
                # extract a single freq to try replacing the pair (set)
                edges = [min(f_n[set_i]) - freq_res, max(f_n[set_i]) + freq_res]
                f_i, a_i, ph_i = extract_single(times, signal - model, f0=edges[0], fn=edges[1], verbose=verbose)
                # make a model including the new freq
                model = sum_sines(times, np.append(f_n_temp, f_i), np.append(a_n_temp, a_i),
                                  np.append(ph_n_temp, ph_i))  # the sinusoid part of the model
                const, slope = linear_slope(times, signal - model, i_sectors)  # redetermine const and slope
                model += linear_curve(times, const, slope, i_sectors)  # the linear part of the model
                # calculate bic
                n_param = 2 * n_sectors + 1 + 2 * n_harm + 3 * (len(f_n_temp) - n_harm + 1)
                bic = calc_bic(signal - model, n_param)
                if (np.round(bic_prev - bic, 2) > 0):
                    # add to list of removed sets
                    remove_sets = np.append(remove_sets, [i])
                    # do not look at sets with the same freqs as the just removed set anymore
                    overlap = s_indices[[np.any([j in set_i for j in subset]) for subset in f_sets]]
                    used_sets = np.unique(np.append(used_sets, [overlap]))
                    # remember the new frequency
                    f_new, a_new, ph_new = np.append(f_new, [f_i]), np.append(a_new, [a_i]), np.append(ph_new, [ph_i])
                    bic_prev = bic
    f_n = np.append(np.delete(f_n, [k for i in remove_sets for k in f_sets[i]]), f_new)
    a_n = np.append(np.delete(a_n, [k for i in remove_sets for k in f_sets[i]]), a_new)
    ph_n = np.append(np.delete(ph_n, [k for i in remove_sets for k in f_sets[i]]), ph_new)
    if verbose:
        n_f_removed = len([k for i in remove_sets for k in f_sets[i]])
        print(f'Frequency sets replaced by a single frequency: {len(remove_sets)} ({n_f_removed} frequencies). '
              f'BIC= {bic_prev:1.2f}')
    # make an array of sets of frequencies to be investigated (now with harmonics)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    close_f_g = af.chains_within_rayleigh(f_n, freq_res)
    f_sets = [g[np.arange(p1, p2 + 1)] for g in close_f_g for p1 in range(len(g) - 1) for p2 in range(p1 + 1, len(g))
              if np.any([g_f in harmonics for g_f in g[np.arange(p1, p2 + 1)]])]
    s_indices = np.arange(len(f_sets))
    remove_sets = np.zeros(0, dtype=int)  # sets of frequencies to replace (by a harmonic)
    used_sets = np.zeros(0, dtype=int)  # sets that are not to be examined anymore
    f_new, a_new, ph_new = np.zeros((3, 0))
    n_prev = -1
    # while frequencies are added to the remove list, continue loop
    while (len(remove_sets) > n_prev):
        n_prev = len(remove_sets)
        for i, set_i in enumerate(f_sets):
            if i not in used_sets:
                # temporary arrays for this iteration (remove combos, remove current set, add new freqs)
                remove = np.append([k for j in remove_sets for k in f_sets[j]], set_i).astype(int)
                f_n_temp = np.append(np.delete(f_n, remove), f_new)
                a_n_temp = np.append(np.delete(a_n, remove), a_new)
                ph_n_temp = np.append(np.delete(ph_n, remove), ph_new)
                # make a model not including the freqs of this iteration
                model = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
                const, slope = linear_slope(times, signal - model, i_sectors)  # redetermine const and slope
                model += linear_curve(times, const, slope, i_sectors)  # the linear part of the model
                # extract the amplitude and phase of the harmonic(s)
                harm_i = [h for h in set_i if h in harmonics]
                f_i = f_n[harm_i]  # fixed f
                a_i = scargle_ampl(times, signal - model, f_n[harm_i])
                ph_i = scargle_phase(times, signal - model, f_n[harm_i])
                # make a model including the new freq
                model = sum_sines(times, np.append(f_n_temp, f_i), np.append(a_n_temp, a_i),
                                  np.append(ph_n_temp, ph_i))  # the sinusoid part of the model
                const, slope = linear_slope(times, signal - model, i_sectors)  # redetermine const and slope
                model += linear_curve(times, const, slope, i_sectors)  # the linear part of the model
                # calculate bic
                n_param = 2 * n_sectors + 1 + 2 * n_harm + 3 * (len(f_n_temp) + len(f_i) - n_harm)
                bic = calc_bic(signal - model, n_param)
                if (np.round(bic_prev - bic, 2) > 0):
                    # add to list of removed sets
                    remove_sets = np.append(remove_sets, [i])
                    # do not look at sets with the same freqs as the just removed set anymore
                    overlap = s_indices[[np.any([j in set_i for j in subset]) for subset in f_sets]]
                    used_sets = np.unique(np.append(used_sets, [overlap]))
                    # remember the new frequency
                    f_new, a_new, ph_new = np.append(f_new, [f_i]), np.append(a_new, [a_i]), np.append(ph_new, [ph_i])
                    bic_prev = bic
    f_n = np.append(np.delete(f_n, [k for i in remove_sets for k in f_sets[i]]), f_new)
    a_n = np.append(np.delete(a_n, [k for i in remove_sets for k in f_sets[i]]), a_new)
    ph_n = np.append(np.delete(ph_n, [k for i in remove_sets for k in f_sets[i]]), ph_new)
    if verbose:
        n_f_removed = len([k for i in remove_sets for k in f_sets[i]])
        print(f'Frequency sets replaced by just harmonic(s): {len(remove_sets)} ({n_f_removed} frequencies). '
              f'BIC= {bic_prev:1.2f}')
    # lastly re-determine slope and const
    model = sum_sines(times, f_n, a_n, ph_n)  # the sinusoid part of the model
    const, slope = linear_slope(times, signal - model, i_sectors)
    return const, slope, f_n, a_n, ph_n


def fix_harmonic_frequency(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors):
    """Fixes the frequecy of harmonics to the theoretical value, then
    re-determines the amplitudes and phases.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        The orbital period
    const: numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: list[float], numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: list[float], numpy.ndarray[float]
        The phases of a number of sine waves
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    
    Returns
    -------
    const: numpy.ndarray[float]
        (Updated) y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        (Updated) slope(s) of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        (Updated) frequencies of the same number of sine waves
    a_n: numpy.ndarray[float]
        (Updated) amplitudes of the same number of sine waves
    ph_n: numpy.ndarray[float]
        (Updated) phases of the same number of sine waves
    """
    # extract the harmonics using the period and determine some numbers
    freq_res = 1.5 / np.ptp(times)
    f_tolerance = min(freq_res / 2, 1 / (2 * p_orb))
    harmonics, harmonic_n = af.find_harmonics_tolerance(f_n, p_orb, f_tol=f_tolerance)
    if (len(harmonics) == 0):
        raise ValueError('No harmonic frequencies found')
    # go through the harmonics by harmonic number
    for n in np.unique(harmonic_n):
        harmonics, harmonic_n = af.find_harmonics_tolerance(f_n, p_orb, f_tol=f_tolerance)
        remove = np.arange(len(f_n))[harmonics][harmonic_n == n]
        f_n = np.delete(f_n, remove)
        a_n = np.delete(a_n, remove)
        ph_n = np.delete(ph_n, remove)
        # make a model excluding the 'n' harmonics
        model = linear_curve(times, const, slope, i_sectors)  # the linear part of the model
        model += sum_sines(times, f_n, a_n, ph_n)  # the sinusoid part of the model
        resid = signal - model
        f_n = np.append(f_n, [n / p_orb])
        a_n = np.append(a_n, [scargle_ampl_single(times, resid, n / p_orb)])
        ph_n = np.append(ph_n, [scargle_phase_single(times, resid, n / p_orb)])
        # make sure the phase stays within + and - pi
        ph_n[-1] = np.mod(ph_n[-1] + np.pi, 2 * np.pi) - np.pi
        # as a last model-refining step, redetermine the constant
        resid = signal - sum_sines(times, f_n, a_n, ph_n)
        const, slope = linear_slope(times, resid, i_sectors)
    # re-extract the non-harmonics
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    for i in non_harm:
        model = linear_curve(times, const, slope, i_sectors)  # the linear part of the model
        model += sum_sines(times, np.delete(f_n, i),
                           np.delete(a_n, i), np.delete(ph_n, i))  # the sinusoid part of the model
        fl, fr = f_n[i] - freq_res, f_n[i] + freq_res
        f_n[i], a_n[i], ph_n[i] = extract_single(times, signal - model, f0=fl, fn=fr, verbose=False)
        # make sure the phase stays within + and - pi
        ph_n[i] = np.mod(ph_n[i] + np.pi, 2 * np.pi) - np.pi
        # as a last model-refining step, redetermine the constant
        resid = signal - sum_sines(times, f_n, a_n, ph_n)
        const, slope = linear_slope(times, resid, i_sectors)
    return const, slope, f_n, a_n, ph_n


def frequency_analysis(tic, times, signal, i_sectors, p_orb=0., data_id=None, save_dir=None, verbose=False, plot=False):
    """Recipe for analysis of EB light curves.
    
    Parameters
    ----------
    tic: int
        The TESS Input Catalog number for later reference
        Use any number (or even str) as reference if not available.
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    p_orb: float
        The orbital period
        if the orbital period is known with certainty beforehand, it can
        be provided as initial value and no new period will be searched.
    data_id: int, str, None
        Identification for the dataset used
    save_dir: str, None
        Path to a directory for save the results. More information
        is saved than is returned by this function.
        Set None to save nothing.
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    p_orb_i: list[float]
        Orbital period at each stage of the analysis
    const_i: list[numpy.ndarray[float]]
        Y-intercept(s) of a piece-wise linear curve for each stage of the analysis
    slope_i: list[numpy.ndarray[float]]
        Slope(s) of a piece-wise linear curve for each stage of the analysis
    f_n_i: list[numpy.ndarray[float]]
        Frequencies of a number of sine waves for each stage of the analysis
    a_n_i: list[numpy.ndarray[float]]
        Amplitudes of a number of sine waves for each stage of the analysis
    ph_n_i: list[numpy.ndarray[float]]
        Phases of a number of sine waves for each stage of the analysis
    
    Notes
    -----
    1) Extract frequencies
        We start by extracting the frequency of highest amplitude one by one, directly from
        the Lomb-Scargle periodogram until the BIC does not significantly improve anymore.
        No fitting is performed yet.
    2) First multi-sine NL-LS fit
        To get the best results in the following steps, a fit is performed over sets of 10-15
        frequencies at a time. Fitting in groups is a trade-off between accuracy and
        drastically reduced time taken.
    3) Measure the orbital period and couple the harmonic frequencies
        Find the orbital period from the longest series of harmonics.
        Find the harmonics with the orbital period, measure a better period
        from the harmonics and set the frequencies of the harmonics to their new values.
    4) Attempt to extract a few more orbital harmonics
        With the decreased number of free parameters (2 vs. 3), the BIC, which punishes
        for free parameters, may allow the extraction of a few more harmonics.
    5) Additional non-harmonics may be found
        Step 3 involves removing frequencies close to the harmonics. These may have included
        actual non-harmonic frequencies. It is attempted to extract these again here.
    6) Multi-NL-LS fit with coupled harmonics
        Fit once again in (larger) groups of frequencies, including the orbital period and the coupled
        harmonics.
    7) Attempt to remove frequencies
        After fitting, it is possible that certain frequencies are better removed than kept.
        This also looks at replacing groups of close frequencies by a single frequency.
        All harmonics are kept at the same frequency.
    8) (only if frequencies removed) Multi-NL-LS fit with coupled harmonics
        If the previous step removed some frequencies, we need to fit one final time.
    """
    t_0a = time.time()
    n_sectors = len(i_sectors)
    freq_res = 1.5 / np.ptp(times)  # Rayleigh criterion
    if save_dir is not None:
        file_id = f'TIC {tic}'
        if not os.path.isdir(os.path.join(save_dir, f'tic_{tic}_analysis')):
            os.mkdir(os.path.join(save_dir, f'tic_{tic}_analysis'))  # create the subdir
    # ---------------------------------------------------
    # [1] --- initial iterative extraction of frequencies
    # ---------------------------------------------------
    if verbose:
        print(f'Starting initial frequency extraction')
    t1_a = time.time()
    const_1, slope_1, f_n_1, a_n_1, ph_n_1 = extract_all(times, signal, i_sectors, verbose=verbose)
    t1_b = time.time()
    # main function done, do the rest for this step
    model_1 = linear_curve(times, const_1, slope_1, i_sectors)
    model_1 += sum_sines(times, f_n_1, a_n_1, ph_n_1)
    n_param_1 = 2 * n_sectors + 3 * len(f_n_1)
    bic = calc_bic(signal - model_1, n_param_1)
    # now print some useful info and/or save the result
    if verbose:
        print(f'\033[1;32;48mInitial frequency extraction complete.\033[0m')
        print(f'\033[0;32;48m{len(f_n_1)} frequencies, {n_param_1} free parameters. '
              f'BIC: {bic:1.2f}, time taken: {t1_b - t1_a:1.1f}s\033[0m\n')
    if save_dir is not None:
        results = (0, const_1, slope_1, f_n_1, a_n_1, ph_n_1)
        f_errors = formal_uncertainties(times, signal - model_1, a_n_1, i_sectors)
        c_err_1, sl_err_1, f_n_err_1, a_n_err_1, ph_n_err_1 = f_errors
        errors = (-1, c_err_1, sl_err_1, f_n_err_1, a_n_err_1, ph_n_err_1)
        stats = (n_param_1, bic, np.std(signal - model_1))
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_1.hdf5')
        desc = '[1] Initial frequency extraction results.'
        ut.save_results(results, errors, stats, file_name, identifier=file_id, description=desc, dataset=data_id)
    # ---------------------------------------------------
    # [2] --- do a first multi-sine NL-LS fit (in chunks)
    # ---------------------------------------------------
    if verbose:
        print(f'Starting multi-sine NL-LS fit.')
    t_2a = time.time()
    f_groups = ut.group_fequencies_for_fit(a_n_1, g_min=10, g_max=15)
    out_2 = tsfit.multi_sine_NL_LS_fit_per_group(times, signal, const_1, slope_1, f_n_1, a_n_1, ph_n_1, i_sectors,
                                                 f_groups, verbose=verbose)
    t_2b = time.time()
    # main function done, do the rest for this step
    const_2, slope_2, f_n_2, a_n_2, ph_n_2 = out_2
    model_2 = linear_curve(times, const_2, slope_2, i_sectors)
    model_2 += sum_sines(times, f_n_2, a_n_2, ph_n_2)
    noise_level_2 = np.std(signal - model_2)
    f_errors = formal_uncertainties(times, signal - model_2, a_n_2, i_sectors)
    c_err_2, sl_err_2, f_n_err_2, a_n_err_2, ph_n_err_2 = f_errors
    n_param_2 = 2 * n_sectors + 3 * len(f_n_2)
    bic = calc_bic(signal - model_2, n_param_2)
    # now print some useful info and/or save the result
    if verbose:
        print(f'\033[1;32;48mFit complete.\033[0m')
        print(f'\033[0;32;48m{len(f_n_2)} frequencies, {n_param_2} free parameters. '
              f'BIC: {bic:1.2f}, time taken: {t_2b - t_2a:1.1f}s\033[0m\n')
    if save_dir is not None:
        results = (0, const_2, slope_2, f_n_2, a_n_2, ph_n_2)
        errors = (-1, c_err_2, sl_err_2, f_n_err_2, a_n_err_2, ph_n_err_2)
        stats = (n_param_2, bic, noise_level_2)
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_2.hdf5')
        desc = '[2] First multi-sine NL-LS fit results.'
        ut.save_results(results, errors, stats, file_name, identifier=file_id, description=desc, dataset=data_id)
    # -------------------------------------------------------------------------------
    # [3] --- measure the orbital period with pdm and couple the harmonic frequencies
    # -------------------------------------------------------------------------------
    if verbose:
        print(f'Coupling the harmonic frequencies to the orbital frequency.')
    t_3a = time.time()
    if (p_orb == 0):
        # first to get a global minimum, inform the pdm by the frequencies
        periods, phase_disp = phase_dispersion_minimisation(times, signal, f_n_2)
        base_p = periods[np.argmin(phase_disp)]
        # do a first test for once or twice the period
        p_best, p_test, opt = af.base_harmonic_check(f_n_2, f_n_err_2, a_n_2, base_p, np.ptp(times), f_tol=freq_res/2)
        # then refine by using a dense sampling
        f_refine = np.arange(0.99 / p_best, 1.01 / p_best, 0.0001 / p_best)
        periods, phase_disp = phase_dispersion_minimisation(times, signal, f_refine)
        p_orb_3 = periods[np.argmin(phase_disp)]
        # try to find out whether we need to double the period
        harmonics, harmonic_n = af.find_harmonics_tolerance(f_n_2, p_orb_3, f_tol=freq_res/2)
        model_h = sum_sines(times, f_n_2[harmonics], a_n_2[harmonics], ph_n_2[harmonics])
        sorted_model_h = model_h[np.argsort(fold_time_series(times, p_orb_3))]
        peaks, props = sp.signal.find_peaks(-sorted_model_h, height=noise_level_2, prominence=noise_level_2, width=9)
        if (len(peaks) == 1):
            p_orb_3 = 2 * p_orb_3
    else:
        # else we use the input p_orb at face value
        p_orb_3 = p_orb
    # if time series too short, cut off the analysis
    if (np.ptp(times) / p_orb_3  < 1.9):
        if verbose:
            print(f'Period over time-base is less than two: {np.ptp(times) / p_orb_3}')
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_3.txt')
            col1 = ['Period over time-base is less than two:', 'period', 'time-base']
            col2 = [np.ptp(times) / p_orb_3, p_orb_3, np.ptp(times)]
            np.savetxt(file_name, np.column_stack((col1, col2)))
        return [None], [None], [None], [None], [None], [None]
    # now couple the harmonics to the period. likely removes more frequencies that need re-extracting
    out_3 = fix_harmonic_frequency(times, signal, p_orb_3, const_2, slope_2, f_n_2, a_n_2, ph_n_2, i_sectors)
    t_3b = time.time()
    # main function done, do the rest for this step
    const_3, slope_3, f_n_3, a_n_3, ph_n_3 = out_3
    model_3 = linear_curve(times, const_3, slope_3, i_sectors)
    model_3 += sum_sines(times, f_n_3, a_n_3, ph_n_3)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_3, p_orb_3)
    n_param_3 = 2 * n_sectors + 1 + 2 * len(harmonics) + 3 * (len(f_n_3) - len(harmonics))
    bic = calc_bic(signal - model_3, n_param_3)
    # now print some useful info and/or save the result
    if verbose:
        print(f'\033[1;32;48mOrbital harmonic frequencies coupled. Period: {p_orb_3:2.4}\033[0m')
        print(f'\033[0;32;48m{len(f_n_3)} frequencies, {n_param_3} free parameters. '
              f'BIC: {bic:1.2f}, time taken: {t_3b - t_3a:1.1f}s\033[0m\n')
    if save_dir is not None:
        results = (p_orb_3, const_3, slope_3, f_n_3, a_n_3, ph_n_3)
        f_errors = formal_uncertainties(times, signal - model_3, a_n_3, i_sectors)
        c_err_3, sl_err_3, f_n_err_3, a_n_err_3, ph_n_err_3 = f_errors
        p_err_3 = formal_period_uncertainty(p_orb_3, f_n_err_3, harmonics, harmonic_n)
        errors = (p_err_3, c_err_3, sl_err_3, f_n_err_3, a_n_err_3, ph_n_err_3)
        stats = (n_param_3, bic, np.std(signal - model_3))
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_3.hdf5')
        desc = '[3] Harmonic frequencies coupled.'
        ut.save_results(results, errors, stats, file_name, identifier=file_id, description=desc, dataset=data_id)
    # ----------------------------------------------------------------------
    # [4] --- attempt to extract more harmonics knowing where they should be
    # ----------------------------------------------------------------------
    if verbose:
        print(f'Looking for additional harmonics.')
    t_4a = time.time()
    out_4 = extract_additional_harmonics(times, signal, p_orb_3, const_3, slope_3, f_n_3, a_n_3, ph_n_3,
                                         i_sectors, verbose=verbose)
    t_4b = time.time()
    # main function done, do the rest for this step
    const_4, slope_4, f_n_4, a_n_4, ph_n_4 = out_4
    model_4 = linear_curve(times, const_4, slope_4, i_sectors)
    model_4 += sum_sines(times, f_n_4, a_n_4, ph_n_4)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_4, p_orb_3)
    n_param_4 = 2 * n_sectors + 1 + 2 * len(harmonics) + 3 * (len(f_n_4) - len(harmonics))
    bic = calc_bic(signal - model_4, n_param_4)
    # now print some useful info and/or save the result
    if verbose:
        print(f'\033[1;32;48m{len(f_n_4) - len(f_n_3)} additional harmonics added.\033[0m')
        print(f'\033[0;32;48m{len(f_n_4)} frequencies, {n_param_4} free parameters. '
              f'BIC: {bic:1.2f}, time taken: {t_4b - t_4a:1.1f}s\033[0m\n')
    if save_dir is not None:
        results = (p_orb_3, const_4, slope_4, f_n_4, a_n_4, ph_n_4)
        f_errors = formal_uncertainties(times, signal - model_4, a_n_4, i_sectors)
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_4, p_orb_3)
        c_err_4, sl_err_4, f_n_err_4, a_n_err_4, ph_n_err_4 = f_errors
        p_err_4 = formal_period_uncertainty(p_orb_3, f_n_err_4, harmonics, harmonic_n)
        errors = (p_err_4, c_err_4, sl_err_4, f_n_err_4, a_n_err_4, ph_n_err_4 )
        stats = (n_param_4, bic, np.std(signal - model_4))
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_4.hdf5')
        desc = '[4] Additional harmonic extraction.'
        ut.save_results(results, errors, stats, file_name, identifier=file_id, description=desc, dataset=data_id)
    # ----------------------------------------------------------------
    # [5] --- attempt to extract additional non-harmonic frequencies
    # ----------------------------------------------------------------
    if verbose:
        print(f'Looking for additional frequencies.')
    t_5a = time.time()
    out_5 = extract_additional_frequencies(times, signal, p_orb_3, const_4, slope_4, f_n_4, a_n_4, ph_n_4,
                                           i_sectors, verbose=verbose)
    t_5b = time.time()
    # main function done, do the rest for this step
    # const_5, slope_5, f_n_5, a_n_5, ph_n_5 = out_4
    const_5, slope_5, f_n_5, a_n_5, ph_n_5 = out_5
    model_5 = linear_curve(times, const_5, slope_5, i_sectors)
    model_5 += sum_sines(times, f_n_5, a_n_5, ph_n_5)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_5, p_orb_3)
    n_param_5 = 2 * n_sectors + 1 + 2 * len(harmonics) + 3 * (len(f_n_5) - len(harmonics))
    bic = calc_bic(signal - model_5, n_param_5)
    # now print some useful info and/or save the result
    if verbose:
        print(f'\033[1;32;48m{len(f_n_5) - len(f_n_4)} additional frequencies added.\033[0m')
        print(f'\033[0;32;48m{len(f_n_5)} frequencies, {n_param_5} free parameters. '
              f'BIC: {bic:1.2f}, time taken: {t_5b - t_5a:1.1f}s\033[0m\n')
    if save_dir is not None:
        results = (p_orb_3, const_5, slope_5, f_n_5, a_n_5, ph_n_5)
        f_errors = formal_uncertainties(times, signal - model_5, a_n_5, i_sectors)
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_5, p_orb_3)
        c_err_5, sl_err_5, f_n_err_5, a_n_err_5, ph_n_err_5 = f_errors
        p_err_5 = formal_period_uncertainty(p_orb_3, f_n_err_5, harmonics, harmonic_n)
        errors = (p_err_5, c_err_5, sl_err_5, f_n_err_5, a_n_err_5, ph_n_err_5 )
        stats = (n_param_5, bic, np.std(signal - model_5))
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_5.hdf5')
        desc = '[5] Additional non-harmonic extraction.'
        ut.save_results(results, errors, stats, file_name, identifier=file_id, description=desc, dataset=data_id)
    # -----------------------------------------------------------------
    # [6] --- fit a second time but now with fixed harmonic frequencies
    # -----------------------------------------------------------------
    if verbose:
        print(f'Starting multi-sine NL-LS fit with harmonics.')
    t_6a = time.time()
    out_6 = tsfit.multi_sine_NL_LS_harmonics_fit_per_group(times, signal, p_orb_3, const_5, slope_5,
                                                           f_n_5, a_n_5, ph_n_5, i_sectors, verbose=verbose)
    t_6b = time.time()
    # main function done, do the rest for this step
    p_orb_6, const_6, slope_6, f_n_6, a_n_6, ph_n_6 = out_6
    model_6 = linear_curve(times, const_6, slope_6, i_sectors)
    model_6 += sum_sines(times, f_n_6, a_n_6, ph_n_6)
    bic = calc_bic(signal - model_6, n_param_5)
    # now print some useful info and/or save the result
    if verbose:
        print(f'\033[1;32;48mFit with fixed harmonics complete. Period: {p_orb_6:2.4}\033[0m')
        print(f'\033[0;32;48m{len(f_n_6)} frequencies, {n_param_5} free parameters. '
              f'BIC: {bic:1.2f}, time taken: {t_6b - t_6a:1.1f}s\033[0m\n')
    if save_dir is not None:
        results = (p_orb_6, const_6, slope_6, f_n_6, a_n_6, ph_n_6)
        f_errors = formal_uncertainties(times, signal - model_6, a_n_6, i_sectors)
        c_err_6, sl_err_6, f_n_err_6, a_n_err_6, ph_n_err_6 = f_errors
        p_err_6 = formal_period_uncertainty(p_orb_6, f_n_err_6, harmonics, harmonic_n)
        errors = (p_err_6, c_err_6, sl_err_6, f_n_err_6, a_n_err_6, ph_n_err_6)
        stats = (n_param_5, bic, np.std(signal - model_6))
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_6.hdf5')
        desc = '[6] Multi-sine NL-LS fit results with coupled harmonics.'
        ut.save_results(results, errors, stats, file_name, identifier=file_id, description=desc, dataset=data_id)
    # ----------------------------------------------------------------------
    # [7] --- try to reduce the number of frequencies after the fit was done
    # ----------------------------------------------------------------------
    if verbose:
        print(f'Attempting to reduce the number of frequencies.')
    t_7a = time.time()
    out_7 = reduce_frequencies_harmonics(times, signal, p_orb_6, const_6, slope_6, f_n_6, a_n_6, ph_n_6, i_sectors,
                                         verbose=verbose)
    t_7b = time.time()
    # main function done, do the rest for this step
    const_7, slope_7, f_n_7, a_n_7, ph_n_7 = out_7
    model_7 = linear_curve(times, const_7, slope_7, i_sectors)
    model_7 += sum_sines(times, f_n_7, a_n_7, ph_n_7)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_7, p_orb_6)
    n_param_7 = 2 * n_sectors + 1 + 2 * len(harmonics) + 3 * (len(f_n_7) - len(harmonics))
    bic = calc_bic(signal - model_7, n_param_7)
    # now print some useful info and/or save the result
    if verbose:
        print(f'\033[1;32;48mReducing frequencies complete.\033[0m')
        print(f'\033[0;32;48m{len(f_n_7)} frequencies, {n_param_7} free parameters. '
              f'BIC: {bic:1.2f}, time taken: {t_7b - t_7a:1.1f}s\033[0m\n')
    if save_dir is not None:
        results = (p_orb_6, const_7, slope_7, f_n_7, a_n_7, ph_n_7)
        f_errors = formal_uncertainties(times, signal - model_7, a_n_7, i_sectors)
        c_err_7, sl_err_7, f_n_err_7, a_n_err_7, ph_n_err_7 = f_errors
        p_err_7 = formal_period_uncertainty(p_orb_6, f_n_err_7, harmonics, harmonic_n)
        errors = (p_err_7, c_err_7, sl_err_7, f_n_err_7, a_n_err_7, ph_n_err_7)
        stats = (n_param_7, bic, np.std(signal - model_7))
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_7.hdf5')
        desc = '[7] Reduce frequency set.'
        ut.save_results(results, errors, stats, file_name, identifier=file_id, description=desc, dataset=data_id)
    # -------------------------------------------------------------------
    # [8] --- need to fit once more after the removal of some frequencies
    # -------------------------------------------------------------------
    if (len(f_n_6) > len(f_n_7)):
        if verbose:
            print(f'Starting second multi-sine NL-LS fit with harmonics.')
        t_8a = time.time()
        out_8 = tsfit.multi_sine_NL_LS_harmonics_fit_per_group(times, signal, p_orb_6, const_7, slope_7,
                                                               f_n_7, a_n_7, ph_n_7, i_sectors, verbose=verbose)
        t_8b = time.time()
        # main function done, do the rest for this step
        p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8 = out_8
        model_8 = linear_curve(times, const_8, slope_8, i_sectors)
        model_8 += sum_sines(times, f_n_8, a_n_8, ph_n_8)
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_8, p_orb_8)
        n_param_8 = 2 * n_sectors + 1 + 2 * len(harmonics) + 3 * (len(f_n_8) - len(harmonics))
        bic = calc_bic(signal - model_8, n_param_8)
        # now print some useful info and/or save the result
        if verbose:
            print(f'\033[1;32;48mFit with fixed harmonics complete. Period: {p_orb_8:2.4}\033[0m')
            print(f'\033[0;32;48m{len(f_n_8)} frequencies, {n_param_8} free parameters. '
                  f'BIC: {bic:1.2f}, time taken: {t_8b - t_8a:1.1f}s\033[0m\n')
    else:
        p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8 = out_6
        n_param_8 = n_param_5
        model_8 = np.copy(model_6)
        bic = calc_bic(signal - model_8, n_param_8)
        if verbose:
            print(f'\033[1;32;48mNo frequencies removed, so no additional fit needed.\033[0m')
    if save_dir is not None:
        results = (p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8)
        f_errors = formal_uncertainties(times, signal - model_8, a_n_8, i_sectors)
        c_err_8, sl_err_8, f_n_err_8, a_n_err_8, ph_n_err_8 = f_errors
        p_err_8 = formal_period_uncertainty(p_orb_8, f_n_err_8, harmonics, harmonic_n)
        errors = (p_err_8, c_err_8, sl_err_8, f_n_err_8, a_n_err_8, ph_n_err_8)
        stats = (n_param_8, bic, np.std(signal - model_8))
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_8.hdf5')
        desc = '[8] Second multi-sine NL-LS fit results with coupled harmonics.'
        ut.save_results(results, errors, stats, file_name, identifier=file_id, description=desc, dataset=data_id)
        # save final freqs and linear curve in ascii format
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_8_sinusoid.dat')
        data = np.column_stack((f_n_8, f_n_err_8, a_n_8, a_n_err_8, ph_n_8, ph_n_err_8))
        hdr = f'p_orb_8: {p_orb_8}, p_err_8: {p_err_8}\nf_n_8, f_n_err_8, a_n_8, a_n_err_8, ph_n_8, ph_n_err_8'
        np.savetxt(file_name, data, delimiter=',', header=hdr)
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_8_linear.dat')
        data = np.column_stack((const_8, c_err_8, slope_8, sl_err_8, i_sectors[:, 0], i_sectors[:, 1]))
        hdr = f'p_orb_8: {p_orb_8}, p_err_8: {p_err_8}\nconst_8, c_err_8, slope_8, sl_err_8, sector_start, sector_end'
        np.savetxt(file_name, data, delimiter=',', header=hdr)
    # final timing and message
    t_0b = time.time()
    if verbose:
        print(f'Frequency extraction done. Total time elapsed: {t_0b - t_0a:1.1f}s. Creating plots.\n')
    # make and/or save plots
    models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8]
    p_orb_i = [0, 0, p_orb_3, p_orb_3, p_orb_3, p_orb_6, p_orb_6, p_orb_8]
    const_i = [const_1, const_2, const_3, const_4, const_5, const_6, const_7, const_8]
    slope_i = [slope_1, slope_2, slope_3, slope_4, slope_5, slope_6, slope_7, slope_8]
    f_n_i = [f_n_1, f_n_2, f_n_3, f_n_4, f_n_5, f_n_6, f_n_7, f_n_8]
    a_n_i = [a_n_1, a_n_2, a_n_3, a_n_4, a_n_5, a_n_6, a_n_7, a_n_8]
    ph_n_i = [ph_n_1, ph_n_2, ph_n_3, ph_n_4, ph_n_5, ph_n_6, ph_n_7, ph_n_8]
    if (save_dir is not None) & plot:
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_frequency_analysis_full_pd.png')
        vis.plot_pd_full_output(times, signal, models, p_orb_i, f_n_i, a_n_i, i_sectors, save_file=file_name, show=False)
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_frequency_analysis_models.png')
        vis.plot_harmonic_output(times, signal, p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8, i_sectors,
                                 save_file=file_name, show=False)
    if plot:
        vis.plot_pd_full_output(times, signal, models, p_orb_i, f_n_i, a_n_i, i_sectors, save_file=None, show=True)
        vis.plot_harmonic_output(times, signal, p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8, i_sectors,
                                 save_file=None, show=True)
    return p_orb_i, const_i, slope_i, f_n_i, a_n_i, ph_n_i


def eclipse_analysis(tic, times, signal, signal_err, i_sectors, data_id=None, save_dir=None, verbose=False, plot=False):
    """Part two of analysis recipe for analysis of EB light curves,
    to be chained after frequency_analysis
    
    Parameters
    ----------
    tic: int
        The TESS Input Catalog number for later reference
        Use any number as reference if not available.
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_sectors = np.array([[0, len(times)]]).
    data_id: int, str, None
        Identification for the dataset used
    save_dir: str, None
        Path to a directory for save the results. Also used to load
        previous analysis results. If None, nothing is saved and the
        loading of previous results is attempted from a relative path.
    save_dir: str, None
        Path to a directory for save the results. More information
        is saved than is returned by this function.
        Set None to save nothing.
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    table: numpy.ndarray[str]
        Table containing all results and descriptions
    """
    # read in the frequency analysis results
    if save_dir is not None:
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_8.hdf5')
    else:
        file_name = os.path.join(f'tic_{tic}_analysis', f'tic_{tic}_analysis_8.hdf5')
    results, errors, stats = ut.read_results(file_name, verbose=verbose)
    p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8 = results
    p_orb_8 = p_orb_8[0]  # must be a float
    p_err_8, c_err_8, sl_err_8, f_n_err_8, a_n_err_8, ph_n_err_8 = errors
    n_param_8, bic_8, noise_level_8 = stats
    # ----------------------------------
    # [8] --- Eclipse timings and depths
    # ----------------------------------
    if verbose:
        print(f'Measuring eclipse time points and depths.')
    t_9a = time.time()
    # deepest eclipse is put first in each measurement
    out_9 = af.measure_eclipses_dt(p_orb_8, f_n_8, a_n_8, ph_n_8, noise_level_8)
    t_zero, t_1, t_2, t_contacts, depths, t_bottoms, t_i_1_err, t_i_2_err = out_9
    # measurements of the first/last contact points
    t_1_1, t_1_2, t_2_1, t_2_2 = t_contacts
    t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = t_bottoms
    # convert to durations
    tau_1_1 = t_1 - t_1_1
    tau_1_2 = t_1_2 - t_1
    tau_2_1 = t_2 - t_2_1
    tau_2_2 = t_2_2 - t_2
    timings = (t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2)
    timings_tau = (t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2)
    bottom_dur = (t_b_1_2 - t_b_1_1, t_b_2_2 - t_b_2_1)
    # define some errors (tau_1_1 error equals t_1_1 error - approximately)
    tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err = t_i_1_err[0], t_i_2_err[0], t_i_1_err[1], t_i_2_err[1]
    t_1_err = np.sqrt(tau_1_1_err**2 + tau_1_2_err**2 + p_err_8**2) / 3  # this is an estimate
    t_2_err = np.sqrt(tau_2_1_err**2 + tau_2_2_err**2 + p_err_8**2) / 3  # this is an estimate
    timing_errs = (t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err)
    bottom_dur_err = (4/3*tau_1_1_err, 4/3*tau_1_2_err)
    # depth errors from the noise levels at contact points and bottom of eclipse
    # sqrt(std(resid)**2/4+std(resid)**2/4+std(resid)**2)
    depths_err = np.array([np.sqrt(3/2 * noise_level_8**2), np.sqrt(3/2 * noise_level_8**2)])
    t_9b = time.time()
    if verbose:
        # determine decimals to print for two significant figures
        rnd_p_orb = max(ut.decimal_figures(p_err_8, 2), ut.decimal_figures(p_orb_8, 2))
        rnd_t_zero = max(ut.decimal_figures(t_1_err, 2), ut.decimal_figures(t_zero, 2))
        rnd_t_1 = max(ut.decimal_figures(t_1_err, 2), ut.decimal_figures(t_1, 2))
        rnd_t_2 = max(ut.decimal_figures(t_2_err, 2), ut.decimal_figures(t_2, 2))
        rnd_t_1_1 = max(ut.decimal_figures(tau_1_1_err, 2), ut.decimal_figures(t_1_1, 2))
        rnd_t_1_2 = max(ut.decimal_figures(tau_1_2_err, 2), ut.decimal_figures(t_1_2, 2))
        rnd_t_2_1 = max(ut.decimal_figures(tau_2_1_err, 2), ut.decimal_figures(t_2_1, 2))
        rnd_t_2_2 = max(ut.decimal_figures(tau_2_2_err, 2), ut.decimal_figures(t_2_2, 2))
        rnd_d_1 = max(ut.decimal_figures(depths_err[0], 2), ut.decimal_figures(depths[0], 2))
        rnd_d_2 = max(ut.decimal_figures(depths_err[1], 2), ut.decimal_figures(depths[1], 2))
        rnd_bot_1 = max(ut.decimal_figures(bottom_dur_err[0], 2), ut.decimal_figures(bottom_dur[0], 2))
        rnd_bot_2 = max(ut.decimal_figures(bottom_dur_err[1], 2), ut.decimal_figures(bottom_dur[1], 2))
        print(f'\033[1;32;48mMeasurements of timings and depths complete.\033[0m')
        print(f'\033[0;32;48mp_orb: {p_orb_8:.{rnd_p_orb}f} (+-{p_err_8:.{rnd_t_1}f}), '
              f't_zero: {t_zero:.{rnd_t_zero}f} (+-{t_1_err:.{rnd_t_zero}f}), \n'
              f't_1: {t_1:.{rnd_t_1}f} (+-{t_1_err:.{rnd_t_1}f}), '
              f't_2: {t_2:.{rnd_t_2}f} (+-{t_2_err:.{rnd_t_2}f}), \n'
              f't_1_1: {t_1_1:.{rnd_t_1_1}f}, tau_1_1: {tau_1_1:.{rnd_t_1_1}f}, (+-{tau_1_1_err:.{rnd_t_1_1}f}), \n'
              f't_1_2: {t_1_2:.{rnd_t_1_2}f}, tau_1_2: {tau_1_2:.{rnd_t_1_2}f}, (+-{tau_1_2_err:.{rnd_t_1_2}f}), \n'
              f't_2_1: {t_2_1:.{rnd_t_2_1}f}, tau_2_1: {tau_2_1:.{rnd_t_2_1}f}, (+-{tau_2_1_err:.{rnd_t_2_1}f}), \n'
              f't_2_2: {t_2_2:.{rnd_t_2_2}f}, tau_2_2: {tau_2_2:.{rnd_t_2_2}f}, (+-{tau_2_2_err:.{rnd_t_2_2}f}), \n'
              f'd_1: {depths[0]:.{rnd_d_1}f} (+-{depths_err[0]:.{rnd_d_1}f}), '
              f'd_2: {depths[1]:.{rnd_d_2}f} (+-{depths_err[1]:.{rnd_d_2}f}), \n'
              f'bottom_dur_1: {bottom_dur[0]:.{rnd_bot_1}f}, (+-{bottom_dur_err[0]:.{rnd_bot_1}f}), \n'
              f'bottom_dur_2: {bottom_dur[1]:.{rnd_bot_2}f}, (+-{bottom_dur_err[1]:.{rnd_bot_2}f}). \n'
              f'Time taken: {t_9b - t_9a:1.1f}s\033[0m\n')
    if save_dir is not None:
        table_1 = ut.save_results_9(tic, t_zero, timings, depths, t_bottoms, timing_errs, depths_err, save_dir, data_id)
    # ------------------------------------------
    # [10] --- Determination of orbital elements
    # ------------------------------------------
    if verbose:
        print(f'Determining eclipse parameters and error estimates.')
    t_10a = time.time()
    out_10 = af.eclipse_parameters(p_orb_8, timings_tau, depths, bottom_dur, timing_errs, depths_err)
    e, w, i, phi_0, psi_0, r_sum_sma, r_dif_sma, r_ratio, sb_ratio = out_10
    # calculate the errors
    out_10_1 = af.error_estimates_hdi(e, w, i, phi_0, psi_0, r_sum_sma, r_dif_sma, r_ratio, sb_ratio,
                                     p_orb_8, f_n_8, a_n_8, ph_n_8, t_zero, timings_tau, bottom_dur,
                                     timing_errs, depths_err, verbose=verbose)
    intervals, bounds, errors, dists_in, dists_out = out_10_1
    e_err, w_err, i_err, phi_0_err, psi_0_err, r_sum_sma_err, r_dif_sma_err, r_ratio_err, sb_ratio_err = errors[:9]
    ecosw_err, esinw_err, f_c_err, f_s_err = errors[9:]
    e_bds, w_bds, i_bds, phi_0_bds, psi_0_bds, r_sum_sma_bds, r_dif_sma_bds, r_ratio_bds, sb_ratio_bds = bounds[:9]
    ecosw_bds, esinw_bds, f_c_bds, f_s_bds = bounds[9:]
    i_sym_err = max(errors[2])  # take the maximum as pessimistic estimate of the symmetric error
    formal_errors = af.formal_uncertainties(e, w, i, phi_0, p_orb_8, *timings_tau, p_err_8, i_sym_err, *timing_errs)
    t_10b = time.time()
    if verbose:
        # determine decimals to print for two significant figures
        rnd_e = max(ut.decimal_figures(min(e_err), 2), ut.decimal_figures(e, 2))
        rnd_w = max(ut.decimal_figures(min(w_err)/np.pi*180, 2), ut.decimal_figures(w/np.pi*180, 2))
        rnd_i = max(ut.decimal_figures(min(i_err)/np.pi*180, 2), ut.decimal_figures(i/np.pi*180, 2))
        rnd_phi0 = max(ut.decimal_figures(min(phi_0_err), 2), ut.decimal_figures(phi_0, 2))
        rnd_rsumsma = max(ut.decimal_figures(min(r_sum_sma_err), 2), ut.decimal_figures(r_sum_sma, 2))
        rnd_rratio = max(ut.decimal_figures(min(r_ratio_err), 2), ut.decimal_figures(r_ratio, 2))
        rnd_sbratio = max(ut.decimal_figures(min(sb_ratio_err), 2), ut.decimal_figures(sb_ratio, 2))
        rnd_ecosw = max(ut.decimal_figures(min(ecosw_err), 2), ut.decimal_figures(e*np.cos(w), 2))
        rnd_esinw = max(ut.decimal_figures(min(esinw_err), 2), ut.decimal_figures(e*np.sin(w), 2))
        print(f'\033[1;32;48mMeasurements and initial optimisation of the eclipse parameters complete.\033[0m')
        print(f'\033[0;32;48me: {e:.{rnd_e}f} (+{e_err[1]:.{rnd_e}f} -{e_err[0]:.{rnd_e}f}), '
              f'bounds ({e_bds[0]:.{rnd_e}f}, {e_bds[1]:.{rnd_e}f}), \n'
              f'w: {w/np.pi*180:.{rnd_w}f} (+{w_err[1]/np.pi*180:.{rnd_w}f} -{w_err[0]/np.pi*180:.{rnd_w}f}) degrees, '
              f'bounds ({w_bds[0]/np.pi*180:.{rnd_w}f}, {w_bds[1]/np.pi*180:.{rnd_w}f}), \n'
              f'i: {i/np.pi*180:.{rnd_i}f} (+{i_err[1]/np.pi*180:.{rnd_i}f} -{i_err[0]/np.pi*180:.{rnd_i}f}) degrees, '
              f'bounds ({i_bds[0]/np.pi*180:.{rnd_i}f}, {i_bds[1]/np.pi*180:.{rnd_i}f}), \n'
              f'(r1+r2)/a: {r_sum_sma:.{rnd_rsumsma}f} '
              f'(+{r_sum_sma_err[1]:.{rnd_rsumsma}f} -{r_sum_sma_err[0]:.{rnd_rsumsma}f}), '
              f'bounds ({r_sum_sma_bds[0]:.{rnd_rsumsma}f}, {r_sum_sma_bds[1]:.{rnd_rsumsma}f}), \n'
              f'r2/r1: {r_ratio:.{rnd_rratio}f} (+{r_ratio_err[1]:.{rnd_rratio}f} -{r_ratio_err[0]:.{rnd_rratio}f}), '
              f'bounds ({r_ratio_bds[0]:.{rnd_rratio}f}, {r_ratio_bds[1]:.{rnd_rratio}f}), \n'
              f'sb2/sb1: {sb_ratio:.{rnd_sbratio}f} '
              f'(+{sb_ratio_err[1]:.{rnd_sbratio}f} -{sb_ratio_err[0]:.{rnd_sbratio}f}), '
              f'bounds ({sb_ratio_bds[0]:.{rnd_sbratio}f}, {sb_ratio_bds[1]:.{rnd_sbratio}f}), \n'
              f'ecos(w): {e*np.cos(w):.{rnd_ecosw}f} (+{ecosw_err[1]:.{rnd_ecosw}f} -{ecosw_err[0]:.{rnd_ecosw}f}), '
              f'bounds ({ecosw_bds[0]:.{rnd_ecosw}f}, {ecosw_bds[1]:.{rnd_ecosw}f}), \n'
              f'esin(w): {e*np.sin(w):.{rnd_esinw}f} (+{esinw_err[1]:.{rnd_esinw}f} -{esinw_err[0]:.{rnd_esinw}f}), '
              f'bounds ({esinw_bds[0]:.{rnd_esinw}f}, {esinw_bds[1]:.{rnd_esinw}f}). \n'
              f'Time taken: {t_10b - t_10a:1.1f}s\033[0m\n')
    if save_dir is not None:
        table_2 = ut.save_results_10(tic, e, w, i, phi_0, r_sum_sma, r_ratio, sb_ratio, errors, intervals, bounds,
                                     formal_errors, dists_in, dists_out, save_dir, data_id)
    # -------------------------------------------
    # [11] --- Fit for the light curve parameters
    # -------------------------------------------
    if verbose:
        print(f'Fitting for the light curve parameters.')
    t_11a = time.time()
    f_c = e**0.5 * np.cos(w)
    f_s = e**0.5 * np.sin(w)
    par_init = (f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio)
    par_bounds = (f_c_bds, f_s_bds, i_bds, r_sum_sma_bds, r_ratio_bds, sb_ratio_bds)
    # todo: test with ldc_1=0.5 and 1.0 on the synthetics
    out_11 = tsfit.fit_eclipse_ellc(times, signal, signal_err, p_orb_8, t_zero, timings, const_8, slope_8,
                                    f_n_8, a_n_8, ph_n_8, i_sectors, par_init, par_bounds, verbose=verbose)
    opt_f_c, opt_f_s, opt_i, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio, offset = out_11.x
    # get e and w from fitting parameters f_c and f_s
    opt_e = opt_f_c**2 + opt_f_s**2
    opt_w = np.arctan2(opt_f_s, opt_f_c) % (2 * np.pi)
    opt_teff_ratio = opt_sb_ratio**(1/4)
    # todo: think of a way to get errors?
    t_11b = time.time()
    if verbose:
        print(f'\033[1;32;48mOptimisation of the light curve parameters complete.\033[0m')
        print(f'\033[0;32;48me: {opt_e:2.4}, w: {opt_w/np.pi*180:2.4} deg, i: {opt_i/np.pi*180:2.4} deg, '
              f'(r1+r2)/a: {opt_r_sum_sma:2.4}, r2/r1: {opt_r_ratio:2.4}, sb2/sb1: {opt_sb_ratio:2.4}, '
              f'Teff2/Teff1: {opt_teff_ratio:2.4}. Time taken: {t_11b - t_11a:1.1f}s\033[0m\n')
    if save_dir is not None:
        table_3 = ut.save_results_11(tic, par_init, out_11.x, save_dir, data_id)
    # make and/or save plots
    if (save_dir is not None) & plot:
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_eclipse_analysis_timestamps.png')
        vis.plot_lc_eclipse_timestamps(times, signal, p_orb_8, t_zero, timings, depths, t_bottoms, timing_errs, depths_err,
                                       const_8, slope_8, f_n_8, a_n_8, ph_n_8, i_sectors, save_file=file_name, show=False)
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_eclipse_analysis_simple_lc.png')
        vis.plot_lc_eclipse_parameters_simple(times, signal, p_orb_8, t_zero, timings, const_8, slope_8, f_n_8,
                                              a_n_8, ph_n_8, i_sectors, (e, w, i, phi_0, r_sum_sma, r_ratio, sb_ratio),
                                              save_file=file_name, show=False)
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_eclipse_analysis_corner.png')
        vis.plot_corner_eclipse_parameters(timings_tau, depths, bottom_dur, *dists_in, e, w, i, phi_0, psi_0,
                                           r_sum_sma, r_dif_sma, r_ratio, sb_ratio, *dists_out,
                                           save_file=file_name, show=False)
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_eclipse_analysis_ellc_fit.png')
        vis.plot_lc_ellc_fit(times, signal, p_orb_8, t_zero, timings, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                             i_sectors, par_init, out_11.x, save_file=file_name, show=False)
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_eclipse_analysis_ellc_corner.png')
        vis.plot_corner_ellc_pars((f_c, f_s, i/np.pi*180, r_sum_sma, r_ratio, sb_ratio),
                                  (opt_f_c, opt_f_s, opt_i/np.pi*180, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio),
                                  *dists_out, save_file=file_name, show=False)
    if plot:
        vis.plot_lc_eclipse_timestamps(times, signal, p_orb_8, t_zero, timings, depths, t_bottoms, timing_errs, depths_err,
                                       const_8, slope_8, f_n_8, a_n_8, ph_n_8, i_sectors, save_file=None, show=True)
        vis.plot_lc_eclipse_parameters_simple(times, signal, p_orb_8, t_zero, timings, const_8, slope_8, f_n_8,
                                              a_n_8, ph_n_8, i_sectors, (e, w, i, phi_0, r_sum_sma, r_ratio, sb_ratio),
                                              save_file=None, show=True)
        vis.plot_dists_eclipse_parameters(timings_tau, depths, bottom_dur, *dists_in, e, w, i, phi_0, psi_0,
                                          r_sum_sma, r_dif_sma, r_ratio, sb_ratio, *dists_out)
        vis.plot_corner_eclipse_parameters(timings_tau, depths, bottom_dur, *dists_in, e, w, i, phi_0, psi_0,
                                           r_sum_sma, r_dif_sma, r_ratio, sb_ratio, *dists_out,
                                           save_file=None, show=True)
        vis.plot_lc_ellc_fit(times, signal, p_orb_8, t_zero, timings, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                             i_sectors, par_init, out_11.x, save_file=None, show=True)
        vis.plot_corner_ellc_pars((f_c, f_s, i/np.pi*180, r_sum_sma, r_ratio, sb_ratio),
                                  (opt_f_c, opt_f_s, opt_i/np.pi*180, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio),
                                  *dists_out, save_file=None, show=True)
    return table_1, table_2, table_3


def pulsation_analysis(tic, times, signal, i_sectors, data_id=None, save_dir=None, verbose=False, plot=False):
    """Part three of analysis recipe for analysis of EB light curves,
    to be chained after frequency_analysis and eclipse_analysis
    
    Parameters
    ----------
    tic: int
        The TESS Input Catalog number for later reference
        Use any number (or even str) as reference if not available.
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    data_id: int, str, None
        Identification for the dataset used
    save_dir: str, None
        Path to a directory for save the results. Also used to load
        previous analysis results. If None, nothing is saved and the
        loading of previous results is attempted from a relative path.
    save_dir: str, None
        Path to a directory for save the results. More information
        is saved than is returned by this function.
        Set None to save nothing.
    verbose: bool
        If set to True, this function will print some information
    """
    # read in the frequency analysis results
    if save_dir is not None:
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_8.hdf5')
    else:
        file_name = os.path.join(f'tic_{tic}_analysis', f'tic_{tic}_analysis_8.hdf5')
    results, errors, stats = ut.read_results(file_name, verbose=verbose)
    p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8 = results
    p_orb_8 = p_orb_8[0]  # must be a float
    p_err_8, c_err_8, sl_err_8, f_n_err_8, a_n_err_8, ph_n_err_8 = errors
    n_param_8, bic_8, noise_level_8 = stats
    # load t_zero from the timings file
    if save_dir is not None:
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_9.csv')
    else:
        file_name = os.path.join(f'tic_{tic}_analysis', f'tic_{tic}_analysis_9.csv')
    timings_all = np.loadtxt(file_name, usecols=(1,), delimiter=',')
    t_zero = timings_all[0]
    timings = timings_all[1:7]
    # open the orbital elements file
    if save_dir is not None:
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_analysis_11.csv')
    else:
        file_name = os.path.join(f'tic_{tic}_analysis', f'tic_{tic}_analysis_11.csv')
    params_ellc = np.loadtxt(file_name, usecols=(1,), delimiter=',')
    params_ellc = params_ellc[6:]
    f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset = params_ellc
    # -------------------------------------
    # [12] --- Frequency selection criteria
    # -------------------------------------
    if verbose:
        print(f'Selecting credible frequencies.')
    t_12a = time.time()
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_8, p_orb_8)
    non_harm = np.delete(np.arange(len(f_n_8)), harmonics)
    remove_sigma = af.remove_insignificant_sigma(f_n_8[non_harm], f_n_err_8[non_harm],
                                                 a_n_8[non_harm], a_n_err_8[non_harm], sigma_a=3., sigma_f=1.)
    remove_snr = af.remove_insignificant_snr(a_n_8[non_harm], noise_level_8, len(times))
    # make selections based on this
    remove = np.union1d(remove_sigma, remove_snr)
    passed_nh = np.delete(non_harm, remove)
    failed_nh = non_harm[remove]
    t_12b = time.time()
    if verbose:
        print(f'\033[1;32;48mFrequencies selected.\033[0m')
        print(f'\033[0;32;48mNumber of non-harmonic frequencies passed: {len(passed_nh)}, '
              f'total number of non-harmonic frequencies: {len(non_harm)}. Time taken: {t_12b - t_12a:1.1f}s\033[0m\n')
    if save_dir is not None:
        table_1 = ut.save_results_12(tic, f_n_8, a_n_8, ph_n_8, non_harm, remove_sigma, remove_snr, save_dir, data_id)
    # ------------------------------------
    # [13] --- Eclipse model disentangling
    # ------------------------------------
    if verbose:
        print(f'Disentangling eclipses from other harmonics.')
    t_13a = time.time()
    out_13 = extract_residual_harmonics(times, signal, p_orb_8, t_zero, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                                        i_sectors, params_ellc, timings, verbose=verbose)
    const_r, f_n_r, a_n_r, ph_n_r = out_13
    # make the models
    model_line = linear_curve(times, const_8, slope_8, i_sectors)
    model_h = sum_sines(times, f_n_8[harmonics], a_n_8[harmonics], ph_n_8[harmonics])
    model_nh = sum_sines(times, f_n_8[non_harm], a_n_8[non_harm], ph_n_8[non_harm])
    model_ellc_h = sum_sines(times, f_n_r, a_n_r, ph_n_r)
    model_ellc = tsfit.ellc_lc_simple(times, p_orb_8, t_zero, f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset)
    # resid of full sinusoid model and ellc + sinusoid model
    resid_sines = signal - model_line - model_nh - model_h
    resid_ellc = signal - model_line - model_nh - model_ellc - const_r - model_ellc_h
    bic_sines = calc_bic(resid_sines, 2 * len(const_8) + 3 * len(non_harm) + 2 * len(harmonics))
    bic_ellc = calc_bic(resid_ellc, 2 * len(const_8) + 3 * len(non_harm) + 7 + 2 * len(f_n_r))
    t_13b = time.time()
    if verbose:
        print(f'\033[1;32;48mHarmonic model disentangled.\033[0m')
        print(f'\033[0;32;48mNumber of harmonic frequencies before: {len(harmonics)}, '
              f'total number of harmonic frequencies after: {len(f_n_r)}. \n'
              f'BIC of full model of sinusoids: {bic_sines:1.2f}, '
              f'BIC of full model with ellc lc: {bic_ellc:1.2f}. '
              f'Time taken: {t_12b - t_12a:1.1f}s\033[0m\n')
    if save_dir is not None:
        table_2 = ut.save_results_13(tic, const_r, f_n_r, a_n_r, ph_n_r, save_dir, data_id)
    # todo: perhaps after this I can dig into the non-harmonic model and try to simplify (ltt effect, blocked light)
    # make and/or save plots
    if (save_dir is not None) & plot:
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_pulsation_analysis_pd.png')
        vis.plot_pd_pulsation_analysis(times, signal, p_orb_8, f_n_8, a_n_8, ph_n_8, noise_level_8, passed_nh,
                                       save_file=file_name, show=False)
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_pulsation_analysis_lc.png')
        vis.plot_lc_pulsation_analysis(times, signal, p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8, i_sectors,
                                       passed_nh, t_zero, const_r, f_n_r, a_n_r, ph_n_r, params_ellc, timings,
                                       save_file=file_name, show=False)
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_pulsation_analysis_ellc_lc.png')
        vis.plot_lc_ellc_harmonics(times, signal, p_orb_8, t_zero, timings, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                                   i_sectors, const_r, f_n_r, a_n_r, ph_n_r, params_ellc,
                                   save_file=file_name, show=False)
        file_name = os.path.join(save_dir, f'tic_{tic}_analysis', f'tic_{tic}_pulsation_analysis_ellc_pd.png')
        vis.plot_pd_ellc_harmonics(times, signal, p_orb_8, t_zero, const_8, slope_8, f_n_8, a_n_8, ph_n_8, i_sectors,
                                   noise_level_8, const_r, f_n_r, a_n_r, ph_n_r, params_ellc, timings,
                                   save_file=file_name, show=False)
    if plot:
        vis.plot_pd_pulsation_analysis(times, signal, p_orb_8, f_n_8, a_n_8, ph_n_8, noise_level_8, passed_nh,
                                       save_file=None, show=True)
        vis.plot_lc_pulsation_analysis(times, signal, p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8, i_sectors,
                                       passed_nh, t_zero, const_r, f_n_r, a_n_r, ph_n_r, params_ellc, timings,
                                       save_file=None, show=True)
        vis.plot_lc_ellc_harmonics(times, signal, p_orb_8, t_zero, timings, const_8, slope_8, f_n_8, a_n_8, ph_n_8,
                                   i_sectors, const_r, f_n_r, a_n_r, ph_n_r, params_ellc, save_file=None, show=True)
        vis.plot_pd_ellc_harmonics(times, signal, p_orb_8, t_zero, const_8, slope_8, f_n_8, a_n_8, ph_n_8, i_sectors,
                                   noise_level_8, const_r, f_n_r, a_n_r, ph_n_r, params_ellc, timings,
                                   save_file=None, show=True)
    return




























