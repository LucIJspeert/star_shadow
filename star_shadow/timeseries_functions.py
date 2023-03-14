"""STAR SHADOW
Satellite Time series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This Python module contains functions for time series analysis;
specifically for the analysis of stellar oscillations and eclipses.

Code written by: Luc IJspeert
"""

import numpy as np
import scipy as sp
import scipy.stats
import numba as nb
import astropy.timeseries as apy

from . import analysis_functions as af
from . import utility as ut


@nb.njit(cache=True)
def fold_time_series_phase(times, p_orb, zero=None):
    """Fold the given time series over the orbital period to transform to phase space.

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    p_orb: float
        The orbital period with which the time series is folded
    zero: float, None
        Reference zero point in time when the phase equals zero

    Returns
    -------
    phases: numpy.ndarray[float]
        Phase array for all timestamps. Phases are between -0.5 and 0.5
    """
    if zero is None:
        zero = times[0]
    phases = ((times - zero) / p_orb + 0.5) % 1 - 0.5
    return phases


@nb.njit(cache=True)
def fold_time_series(times, p_orb, t_zero, t_ext_1=0, t_ext_2=0):
    """Fold the given time series over the orbital period
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    p_orb: float
        The orbital period with which the time series is folded
    t_zero: float, None
        Reference zero point in time (with respect to the time series mean time)
        when the phase equals zero
    t_ext_1: float
        Negative time interval to extend the folded time series to the left.
    t_ext_2: float
        Positive time interval to extend the folded time series to the right.
    
    Returns
    -------
    t_extended: numpy.ndarray[float]
        Folded time series array for all timestamps (and possible extensions).
    ext_left: numpy.ndarray[bool]
        Mask of points to extend time series to the left (for if t_ext_1!=0)
    ext_right: numpy.ndarray[bool]
        Mask of points to extend time series to the right (for if t_ext_2!=0)
    """
    # reference time is the mean of the times array
    mean_t = np.mean(times)
    t_folded = (times - mean_t - t_zero) % p_orb
    # extend to both sides
    ext_left = (t_folded > p_orb + t_ext_1)
    ext_right = (t_folded < t_ext_2)
    t_extended = np.concatenate((t_folded[ext_left] - p_orb, t_folded, t_folded[ext_right] + p_orb))
    return t_extended, ext_left, ext_right


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
    crunching, use a specialised function that can be JIT-ted instead.
    """
    if not hasattr(bins, '__len__'):
        # use as number of bins, else use as bin edges
        bins = np.linspace(-0.5, 0.5, bins + 1)
    binned, edges, indices = sp.stats.binned_statistic(phases, signal, statistic=statistic, bins=bins)
    if midpoints:
        bins = (bins[1:] + bins[:-1]) / 2
    return bins, binned


@nb.njit(cache=True)
def mark_folded_gaps(times, p_orb, width):
    """Mark gaps in a folded series of time points.

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    p_orb: float
        The orbital period with which the time series is folded
    width: float
        Minimum width for a gap (in time units)

    Returns
    -------
    gaps: numpy.ndarray[float]
        Gap timestamps in pairs
    """
    # fold the time series
    t_fold_edges, _, _ = fold_time_series(times, p_orb, 0, t_ext_1=0, t_ext_2=0)
    if np.all(t_fold_edges > 0):
        t_fold_edges = np.append(0, t_fold_edges)
    if np.all(t_fold_edges < p_orb):
        t_fold_edges = np.append(t_fold_edges, p_orb)
    # mark the gaps
    t_sorted = np.sort(t_fold_edges)
    t_diff = t_sorted[1:] - t_sorted[:-1]  # np.diff(a)
    gaps = (t_diff > width)
    # get the timestamps
    t_left = t_sorted[:-1][gaps]
    t_right = t_sorted[1:][gaps]
    gaps = np.column_stack((t_left, t_right))
    return gaps


@nb.njit(cache=True)
def mask_timestamps(times, stamps):
    """Mask out everything except the parts between the given timestamps

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    stamps: numpy.ndarray[float]
        Pairs of timestamps

    Returns
    -------
    mask: numpy.ndarray[bool]
        Boolean mask that is True between the stamps
    """
    mask = np.zeros(len(times), dtype=np.bool_)
    for ts in stamps:
        mask = mask | ((times >= ts[0]) & (times <= ts[-1]))
    return mask


@nb.njit(cache=True)
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
    to enable JIT-ting, which makes this considerably faster.
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


@nb.njit(cache=True)
def phase_dispersion_minimisation(times, signal, f_n, local=False):
    """Determine the phase dispersion over a set of periods to find the minimum
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    local: bool
        If set True, only searches the given frequencies,
        else also fractions of the frequencies are searched
    
    Returns
    -------
    periods: numpy.ndarray[float]
        Periods at which the phase dispersion is calculated
    pd_all: numpy.ndarray[float]
        Phase dispersion at the given periods
    """
    # number of bins for dispersion calculation
    n_points = len(times)
    if (n_points / 10 > 1000):
        n_bins = 1000
    else:
        n_bins = n_points // 10  # at least 10 data points per bin on average
    # determine where to look based on the frequencies, including fractions of the frequencies
    if local:
        periods = 1 / f_n
    else:
        periods = np.zeros(7 * len(f_n))
        for i, f in enumerate(f_n):
            periods[7*i:7*i+7] = np.arange(1, 8) / f
    # stay below the maximum
    periods = periods[periods < np.ptp(times)]
    # and above the minimum
    periods = periods[periods > (2 * np.min(times[1:] - times[:-1]))]
    # compute the dispersion measures
    pd_all = np.zeros(len(periods))
    for i, p in enumerate(periods):
        fold = fold_time_series_phase(times, p, 0)
        pd_all[i] = phase_dispersion(fold, signal, n_bins)
    return periods, pd_all


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
    freqs, ampls = astropy_scargle(times, signal)  # use defaults to get full amplitude spectrum
    # determine the number of points to extend the spectrum with for convolution
    n_points = int(np.ceil(window_width / np.abs(freqs[1] - freqs[0])))  # .astype(int)
    window = np.full(n_points, 1 / n_points)
    # extend the array with mirrors for convolution
    ext_ampls = np.concatenate((ampls[(n_points - 1)::-1], ampls, ampls[:-(n_points + 1):-1]))
    ext_noise = np.convolve(ext_ampls, window, 'same')
    # cut back to original interval
    noise = ext_noise[n_points:-n_points]
    # extra correction to account for convolve mode='full' instead of 'same' (needed for JIT-ting)
    # noise = noise[n_points//2 - 1:-n_points//2]
    return noise


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
    freqs, ampls = astropy_scargle(times, signal)  # use defaults to get full amplitude spectrum
    margin = window_width / 2
    noise = np.array([np.average(ampls[(freqs > f - margin) & (freqs <= f + margin)]) for f in fs])
    return noise


# @nb.njit()  # not sped up (in this form)
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
    win_kernel = cos_term**2 + sin_term**2
    # Normalise such that win_kernel(nu = 0.0) = 1.0
    spec_win = win_kernel / n_time**2
    return spec_win


@nb.njit(cache=True)
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
    Translated from Fortran (and just as fast when JIT-ted with Numba!)
        Computation of Scargles periodogram without explicit tau
        calculation, with iteration (Method Cuypers)
        (is this the same: https://ui.adsabs.harvard.edu/abs/1989ApJ...338..277P/abstract ?)
    
    The times array is mean subtracted to reduce correlation between
    frequencies and phases. The signal array is mean subtracted to avoid
    a large peak at frequency equal to zero.
    
    Useful extra information: VanderPlas 2018,
    https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract
    """
    # times and signal are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(times)
    mean_s = np.mean(signal)
    times_ms = times - mean_t
    signal_ms = signal - mean_s
    # setup
    n = len(signal_ms)
    t_tot = np.ptp(times_ms)
    f0 = max(f0, 0.01 / t_tot)  # don't go lower than T/100
    if (df == 0):
        df = 0.1 / t_tot
    if (fn == 0):
        fn = 1 / (2 * np.min(times_ms[1:] - times_ms[:-1]))
    nf = int((fn - f0) / df + 0.001) + 1
    # pre-assign some memory
    ss = np.zeros(nf)
    sc = np.zeros(nf)
    ss2 = np.zeros(nf)
    sc2 = np.zeros(nf)
    # here is the actual calculation:
    two_pi = 2 * np.pi
    for i in range(n):
        t_f0 = (times_ms[i] * two_pi * f0) % two_pi
        sin_f0 = np.sin(t_f0)
        cos_f0 = np.cos(t_f0)
        mc_1_a = 2 * sin_f0 * cos_f0
        mc_1_b = cos_f0 * cos_f0 - sin_f0 * sin_f0

        t_df = (times_ms[i] * two_pi * df) % two_pi
        sin_df = np.sin(t_df)
        cos_df = np.cos(t_df)
        mc_2_a = 2 * sin_df * cos_df
        mc_2_b = cos_df * cos_df - sin_df * sin_df
        
        sin_f0_s = sin_f0 * signal_ms[i]
        cos_f0_s = cos_f0 * signal_ms[i]
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
        s1 /= np.var(signal_ms)
    elif norm == 'amplitude':  # amplitude spectrum
        s1 = np.sqrt(4 / n) * np.sqrt(s1)
    elif norm == 'density':  # power density
        s1 = (4 / n) * s1 * t_tot
    return f1, s1


@nb.njit(cache=True)
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
    
    Notes
    -----
    The times array is mean subtracted to reduce correlation between
    frequencies and phases. The signal array is mean subtracted to avoid
    a large peak at frequency equal to zero.
    """
    # times and signal are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(times)
    mean_s = np.mean(signal)
    times_ms = times - mean_t
    signal_ms = signal - mean_s
    # multiples of pi
    two_pi = 2 * np.pi
    four_pi = 4 * np.pi
    # define tau
    cos_tau = 0
    sin_tau = 0
    for j in range(len(times_ms)):
        cos_tau += np.cos(four_pi * f * times_ms[j])
        sin_tau += np.sin(four_pi * f * times_ms[j])
    tau = 1 / (four_pi * f) * np.arctan2(sin_tau, cos_tau)  # tau(f)
    # define the general cos and sin functions
    s_cos = 0
    cos_2 = 0
    s_sin = 0
    sin_2 = 0
    for j in range(len(times_ms)):
        cos = np.cos(two_pi * f * (times_ms[j] - tau))
        sin = np.sin(two_pi * f * (times_ms[j] - tau))
        s_cos += signal_ms[j] * cos
        cos_2 += cos**2
        s_sin += signal_ms[j] * sin
        sin_2 += sin**2
    # final calculations
    a_cos_2 = s_cos**2 / cos_2
    b_sin_2 = s_sin**2 / sin_2
    # amplitude
    ampl = (a_cos_2 + b_sin_2) / 2
    ampl = np.sqrt(4 / len(times_ms)) * np.sqrt(ampl)  # conversion to amplitude
    return ampl


@nb.njit(cache=True)
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
    fs: numpy.ndarray[float]
        A set of frequencies

    Returns
    -------
    f1: numpy.ndarray[float]
        Frequencies at which the periodogram was calculated
    s1: numpy.ndarray[float]
        The periodogram spectrum in the chosen units
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
    
    Notes
    -----
    The times array is mean subtracted to reduce correlation between
    frequencies and phases. The signal array is mean subtracted to avoid
    a large peak at frequency equal to zero.
    """
    # times and signal are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(times)
    mean_s = np.mean(signal)
    times_ms = times - mean_t
    signal_ms = signal - mean_s
    # multiples of pi
    two_pi = 2 * np.pi
    four_pi = 4 * np.pi
    fs = np.atleast_1d(fs)
    ampl = np.zeros(len(fs))
    for i in range(len(fs)):
        # define tau
        cos_tau = 0
        sin_tau = 0
        for j in range(len(times_ms)):
            cos_tau += np.cos(four_pi * fs[i] * times_ms[j])
            sin_tau += np.sin(four_pi * fs[i] * times_ms[j])
        tau = 1 / (four_pi * fs[i]) * np.arctan2(sin_tau, cos_tau)  # tau(f)
        # define the general cos and sin functions
        s_cos = 0
        cos_2 = 0
        s_sin = 0
        sin_2 = 0
        for j in range(len(times_ms)):
            cos = np.cos(two_pi * fs[i] * (times_ms[j] - tau))
            sin = np.sin(two_pi * fs[i] * (times_ms[j] - tau))
            s_cos += signal_ms[j] * cos
            cos_2 += cos**2
            s_sin += signal_ms[j] * sin
            sin_2 += sin**2
        # final calculations
        a_cos_2 = s_cos**2 / cos_2
        b_sin_2 = s_sin**2 / sin_2
        # amplitude
        ampl[i] = (a_cos_2 + b_sin_2) / 2
        ampl[i] = np.sqrt(4 / len(times_ms)) * np.sqrt(ampl[i])  # conversion to amplitude
    return ampl


@nb.njit(cache=True)
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
    
    Notes
    -----
    The times array is mean subtracted to reduce correlation between
    frequencies and phases. The signal array is mean subtracted to avoid
    a large peak at frequency equal to zero.
    """
    # times and signal are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(times)
    mean_s = np.mean(signal)
    times_ms = times - mean_t
    signal_ms = signal - mean_s
    # multiples of pi
    two_pi = 2 * np.pi
    four_pi = 4 * np.pi
    # define tau
    cos_tau = 0
    sin_tau = 0
    for j in range(len(times_ms)):
        cos_tau += np.cos(four_pi * f * times_ms[j])
        sin_tau += np.sin(four_pi * f * times_ms[j])
    tau = 1 / (four_pi * f) * np.arctan2(sin_tau, cos_tau)  # tau(f)
    # define the general cos and sin functions
    s_cos = 0
    cos_2 = 0
    s_sin = 0
    sin_2 = 0
    for j in range(len(times_ms)):
        cos = np.cos(two_pi * f * (times_ms[j] - tau))
        sin = np.sin(two_pi * f * (times_ms[j] - tau))
        s_cos += signal_ms[j] * cos
        cos_2 += cos**2
        s_sin += signal_ms[j] * sin
        sin_2 += sin**2
    # final calculations
    a_cos = s_cos / cos_2**(1/2)
    b_sin = s_sin / sin_2**(1/2)
    # sine phase (radians)
    phi = np.pi/2 - np.arctan2(b_sin, a_cos) - two_pi * f * tau
    return phi


@nb.njit(cache=True)
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
    
    Notes
    -----
    The times array is mean subtracted to reduce correlation between
    frequencies and phases. The signal array is mean subtracted to avoid
    a large peak at frequency equal to zero.
    """
    # times and signal are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(times)
    mean_s = np.mean(signal)
    times_ms = times - mean_t
    signal_ms = signal - mean_s
    # multiples of pi
    two_pi = 2 * np.pi
    four_pi = 4 * np.pi
    fs = np.atleast_1d(fs)
    phi = np.zeros(len(fs))
    for i in range(len(fs)):
        # define tau
        cos_tau = 0
        sin_tau = 0
        for j in range(len(times_ms)):
            cos_tau += np.cos(four_pi * fs[i] * times_ms[j])
            sin_tau += np.sin(four_pi * fs[i] * times_ms[j])
        tau = 1 / (four_pi * fs[i]) * np.arctan2(sin_tau, cos_tau)  # tau(f)
        # define the general cos and sin functions
        s_cos = 0
        cos_2 = 0
        s_sin = 0
        sin_2 = 0
        for j in range(len(times_ms)):
            cos = np.cos(two_pi * fs[i] * (times_ms[j] - tau))
            sin = np.sin(two_pi * fs[i] * (times_ms[j] - tau))
            s_cos += signal_ms[j] * cos
            cos_2 += cos**2
            s_sin += signal_ms[j] * sin
            sin_2 += sin**2
        # final calculations
        a_cos = s_cos / cos_2**(1/2)
        b_sin = s_sin / sin_2**(1/2)
        # sine phase (radians)
        phi[i] = np.pi / 2 - np.arctan2(b_sin, a_cos) - two_pi * fs[i] * tau
    return phi


def astropy_scargle(times, signal, f0=0, fn=0, df=0, norm='amplitude'):
    """Wrapper for the astropy Scargle periodogram.

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
    Approximation using fft, much faster than the other scargle in mode='fast'.
    Beware of computing narrower frequency windows, as there is inconsistency
    when doing this.
    
    Useful extra information: VanderPlas 2018,
    https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract
    
    The times array is mean subtracted to reduce correlation between
    frequencies and phases. The signal array is mean subtracted to avoid
    a large peak at frequency equal to zero.
    """
    # times and signal are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(times)
    mean_s = np.mean(signal)
    times_ms = times - mean_t
    signal_ms = signal - mean_s
    # setup
    n = len(signal)
    t_tot = np.ptp(times_ms)
    f0 = max(f0, 0.01 / t_tot)  # don't go lower than T/100
    if (df == 0):
        df = 0.1 / t_tot
    if (fn == 0):
        fn = 1 / (2 * np.min(times_ms[1:] - times_ms[:-1]))
    nf = int((fn - f0) / df + 0.001) + 1
    f1 = f0 + np.arange(nf) * df
    # use the astropy fast algorithm and normalise afterward
    ls = apy.LombScargle(times_ms, signal_ms, fit_mean=False, center_data=False)
    s1 = ls.power(f1, normalization='psd', method='fast')
    # convert to the wanted normalisation
    if norm == 'distribution':  # statistical distribution
        s1 /= np.var(signal_ms)
    elif norm == 'amplitude':  # amplitude spectrum
        s1 = np.sqrt(4 / n) * np.sqrt(s1)
    elif norm == 'density':  # power density
        s1 = (4 / n) * s1 * t_tot
    return f1, s1


@nb.njit(cache=True)
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
    # originally un-JIT-ted function, but for loop is quicker with numba
    sum_r_2 = 0
    for i, r in enumerate(residuals):
        sum_r_2 += r**2
    like = -n / 2 * (np.log(2 * np.pi * sum_r_2 / n) + 1)
    return like


@nb.njit(cache=True)
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
    # originally JIT-ted function, but with for loop is slightly quicker
    sum_r_2 = 0
    for i, r in enumerate(residuals):
        sum_r_2 += r**2
    bic = n * np.log(2 * np.pi * sum_r_2 / n) + n + n_param * np.log(n)
    return bic


@nb.njit(cache=True)
def linear_curve(times, const, slope, i_sectors, t_shift=True):
    """Returns a piece-wise linear curve for the given time points
    with slopes and y-intercepts.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    t_shift: bool
        Mean center the time axis
    
    Returns
    -------
    curve: numpy.ndarray[float]
        The model time series of a (set of) straight line(s)
    
    Notes
    -----
    Assumes the constants and slopes are determined with respect
    to the sector mean time as zero point.
    """
    
    curve = np.zeros(len(times))
    for co, sl, s in zip(const, slope, i_sectors):
        if t_shift:
            t_sector_mean = np.mean(times[s[0]:s[1]])
        else:
            t_sector_mean = 0
        curve[s[0]:s[1]] = co + sl * (times[s[0]:s[1]] - t_sector_mean)
    return curve


@nb.njit(cache=True)
def linear_pars(times, signal, i_sectors):
    """Calculate the slopes and y-intercepts of a linear trend with the MLE.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    
    Returns
    -------
    y_inter: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    
    Notes
    -----
    Source: https://towardsdatascience.com/linear-regression-91eeae7d6a2e
    Determines the constants and slopes with respect to the sector mean time
    as zero point to avoid correlations.
    """
    y_inter = np.zeros(len(i_sectors))
    slope = np.zeros(len(i_sectors))
    for i, s in enumerate(i_sectors):
        # mean and mean subtracted quantities
        x_m = np.mean(times[s[0]:s[1]])
        x_ms = (times[s[0]:s[1]] - x_m)
        y_m = np.mean(signal[s[0]:s[1]])
        y_ms = (signal[s[0]:s[1]] - y_m)
        # sums
        s_xx = np.sum(x_ms**2)
        s_xy = np.sum(x_ms * y_ms)
        # parameters
        slope[i] = s_xy / s_xx
        # y_inter[i] = y_m - slope[i] * x_m  # original non-mean-centered formula
        y_inter[i] = y_m  # mean-centered value
    return y_inter, slope


@nb.njit(cache=True)
def linear_pars_two_points(x1, y1, x2, y2):
    """Calculate the slope(s) and y-intercept(s) of a linear curve defined by two points.
    
    Parameters
    ----------
    x1: float, numpy.ndarray[float]
        The x-coordinate of the left point(s)
    y1: float, numpy.ndarray[float]
        The y-coordinate of the left point(s)
    x2: float, numpy.ndarray[float]
        The x-coordinate of the right point(s)
    y2: float, numpy.ndarray[float]
        The y-coordinate of the right point(s)
    
    Returns
    -------
    y_inter: float, numpy.ndarray[float]
        The y-intercept(s) of a piece-wise linear curve
    slope: float, numpy.ndarray[float]
        The slope(s) of a piece-wise linear curve
    
    Notes
    -----
    Determines the slope and y-intercept with respect to the center
    between the two x-values.
    """
    slope = (y2 - y1) / (x2 - x1)
    y_inter = (y1 + y2) / 2  # halfway point is y-intercept for mean-centered x
    return y_inter, slope


@nb.njit(cache=True)
def quadratic_curve(times, a, b, c):
    """Returns a parabolic curve for the given time points and parameters.

    Parameters
    ----------
    times: float, numpy.ndarray[float]
        Timestamps of the time series
    a: float
        The quadratic coefficient
    b: float
        The linear coefficient
    c: float
        The constant coefficient

    Returns
    -------
    curve: numpy.ndarray[float]
        The model time series of a (set of) straight line(s)
    
    Notes
    -----
    Assumes the parameters are determined with respect
    to the mean time as zero point.
    """
    mean_t = np.mean(times)
    t_mean_sub = times - mean_t  # mean subtracted times
    curve = a * t_mean_sub**2 + b * t_mean_sub + c
    return curve


@nb.njit(cache=True)
def quadratic_pars(times, signal):
    """Returns a parabolic curve for the given time points and parameters.

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series

    Returns
    -------
    a: float
        The quadratic coefficient
    b: float
        The linear coefficient
    c: float
        The constant coefficient
    
    Notes
    -----
    Determines the parameters with respect to the mean time
    as zero point to reduce correlations.
    """
    mean_t = np.mean(times)
    t_mean_sub = times - mean_t  # mean subtracted times
    # mean and mean subtracted quantities
    x_m = np.mean(t_mean_sub)
    x2_m = np.mean(t_mean_sub**2)
    x_ms = (t_mean_sub - x_m)
    x2_ms = (t_mean_sub**2 - x2_m)
    y_m = np.mean(signal)
    y_ms = (signal - y_m)
    # sums
    s_xx = np.sum(x_ms**2)
    s_x2x = np.sum(x2_ms * x_ms)
    s_x2x2 = np.sum(x2_ms**2)
    s_xy = np.sum(x_ms * y_ms)
    s_x2y = np.sum(x2_ms * y_ms)
    # parameters
    a = (s_x2y * s_xx - s_xy * s_x2x) / (s_x2x2 * s_xx - s_x2x**2)
    b = (s_xy - a * s_x2x) / s_xx
    c = y_m - a * x2_m - b * x_m
    return a, b, c


@nb.njit(cache=True)
def cubic_curve(times, a, b, c, d, t_zero):
    """Returns a parabolic curve for the given time points and parameters.

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    a: float
        The cubic coefficient
    b: float
        The quadratic coefficient
    c: float
        The linear coefficient
    d: float
        The constant coefficient
    t_zero: float
        Time zero point with respect to the mean time

    Returns
    -------
    curve: numpy.ndarray[float]
        The model time series of a (set of) straight line(s)

    Notes
    -----
    Assumes the parameters are determined with respect
    to the mean time as zero point.
    """
    mean_t = np.mean(times)
    t_mean_sub = times - mean_t - t_zero  # mean subtracted times
    curve = a * t_mean_sub**3 + b * t_mean_sub**2 + c * t_mean_sub + d
    return curve


@nb.njit(cache=True)
def cubic_pars(times, signal):
    """Returns a cubic curve for the given time points and parameters.

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series

    Returns
    -------
    a: float
        The cubic coefficient
    b: float
        The quadratic coefficient
    c: float
        The linear coefficient
    d: float
        The constant coefficient
    
    Notes
    -----
    Determines the parameters with respect to the mean time
    as zero point to reduce correlations.
    """
    mean_t = np.mean(times)
    t_mean_sub = times - mean_t  # mean subtracted times
    # mean and mean subtracted quantities
    x_m = np.mean(t_mean_sub)
    x2_m = np.mean(t_mean_sub**2)
    x3_m = np.mean(t_mean_sub**3)
    x_ms = (t_mean_sub - x_m)
    x2_ms = (t_mean_sub**2 - x2_m)
    x3_ms = (t_mean_sub**3 - x3_m)
    y_m = np.mean(signal)
    y_ms = (signal - y_m)
    # sums
    s_xx = np.sum(x_ms**2)
    s_x2x = np.sum(x2_ms * x_ms)
    s_x2x2 = np.sum(x2_ms**2)
    s_x3x = np.sum(x3_ms * x_ms)
    s_x3x2 = np.sum(x3_ms * x2_ms)
    s_x3x3 = np.sum(x3_ms**2)
    s_xy = np.sum(x_ms * y_ms)
    s_x2y = np.sum(x2_ms * y_ms)
    s_x3y = np.sum(x3_ms * y_ms)
    # parameters
    a = (s_x3y * (s_x2x2 * s_xx - s_x2x**2) - s_x2y * (s_x3x2 * s_xx - s_x3x * s_x2x)
         + s_xy * (s_x3x2 * s_x2x - s_x3x * s_x2x2))
    a = a / (s_x3x3 * (s_x2x2 * s_xx - s_x2x**2) - s_x3x2 * (s_x3x2 * s_xx - 2 * s_x3x * s_x2x) - s_x3x**2 * s_x2x2)
    b = (s_x2y * s_xx - s_xy * s_x2x - a * (s_x3x2 * s_xx - s_x3x * s_x2x)) / (s_x2x2 * s_xx - s_x2x**2)
    c = (s_xy - a * s_x3x - b * s_x2x) / s_xx
    d = y_m - a * x3_m - b * x2_m - c * x_m
    return a, b, c, d


@nb.njit(cache=True)
def cubic_pars_from_quadratic(x1, a_q, b_q, c_q):
    """Returns a cubic curve corresponding to an integrated quadratic.

    Parameters
    ----------
    x1: float, numpy.ndarray[float]
        The x-coordinate of one zero point in the quadratic
    a_q: float, numpy.ndarray[float]
        The quadratic coefficient(s)
    b_q: float, numpy.ndarray[float]
        The linear coefficient(s)
    c_q: float, numpy.ndarray[float]
        The constant coefficient(s)

    Returns
    -------
    a: float, numpy.ndarray[float]
        The cubic coefficient(s)
    b: float, numpy.ndarray[float]
        The quadratic coefficient(s)
    c: float, numpy.ndarray[float]
        The linear coefficient(s)
    d: float, numpy.ndarray[float]
        The constant coefficient(s)
    """
    a = a_q / 3
    b = b_q / 2
    c = c_q
    d = -(a_q / 3 * x1**3 + b_q / 2 * x1**2 + c_q * x1)
    return a, b, c, d


@nb.njit(cache=True)
def cubic_pars_two_points(x1, y1, x2, y2):
    """Calculate the parameters of a cubic formula defined by its extrema
     y = a*x**3 + b*x**2 + c*x + d

    Parameters
    ----------
    x1: float, numpy.ndarray[float]
        The x-coordinate of the local maximum point(s)
    y1: float, numpy.ndarray[float]
        The y-coordinate of the local maximum point(s)
    x2: float, numpy.ndarray[float]
        The x-coordinate of the local minimum point(s)
    y2: float, numpy.ndarray[float]
        The y-coordinate of the local minimum point(s)

    Returns
    -------
    a: float, numpy.ndarray[float]
        The quadratic coefficient(s)
    b: float, numpy.ndarray[float]
        The linear coefficient(s)
    c: float, numpy.ndarray[float]
        The constant coefficient(s)
    d: float, numpy.ndarray[float]
        The constant coefficient(s)
    
    Notes
    -----
    Determines the parameters with respect to the mean x
    as zero point to reduce correlations.
    """
    mean_x = (x1 + x2) / 2
    x1_ms = x1 - mean_x
    x2_ms = x2 - mean_x
    # parameter formulae
    cbc_dif = (x2_ms - x1_ms)**3
    a = -2 * (y2 - y1) / cbc_dif
    b = 3 * (y2 - y1) * (x2_ms + x1_ms) / cbc_dif
    c = -3 * a * x1_ms**2 - 2 * b * x1_ms
    d = y1 - a * x1_ms**3 - b * x1_ms**2 - c * x1_ms
    return a, b, c, d


@nb.njit(cache=True)
def sum_sines(times, f_n, a_n, ph_n, t_shift=True):
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
    t_shift: bool
        Mean center the time axis
    
    Returns
    -------
    model_sines: numpy.ndarray[float]
        Model time series of a sum of sine waves. Varies around 0.
    
    Notes
    -----
    Assumes the phases are determined with respect
    to the mean time as zero point.
    """
    if t_shift:
        mean_t = np.mean(times)
    else:
        mean_t = 0
    model_sines = np.zeros(len(times))
    for f, a, ph in zip(f_n, a_n, ph_n):
        # model_sines += a * np.sin((2 * np.pi * f * (times - mean_t)) + ph)
        # double loop runs a tad bit quicker when numba-JIT-ted
        for i, t in enumerate(times):
            model_sines[i] += a * np.sin((2 * np.pi * f * (t - mean_t)) + ph)
    return model_sines


@nb.njit(cache=True)
def sum_sines_deriv(times, f_n, a_n, ph_n, deriv=1, t_shift=True):
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
    t_shift: bool
        Mean center the time axis
    
    Returns
    -------
    model_sines: numpy.ndarray[float]
        Model time series of a sum of sine wave derivatives. Varies around 0.
    
    Notes
    -----
    Assumes the phases are determined with respect
    to the mean time as zero point.
    """
    if t_shift:
        mean_t = np.mean(times)
    else:
        mean_t = 0
    model_sines = np.zeros(len(times))
    mod_2 = deriv % 2
    mod_4 = deriv % 4
    ph_cos = (np.pi / 2) * mod_2  # alternate between cosine and sine
    sign = (-1)**((mod_4 - mod_2) // 2)  # (1, -1, -1, 1, 1, -1, -1... for deriv=1, 2, 3...)
    for f, a, ph in zip(f_n, a_n, ph_n):
        for i, t in enumerate(times):
            model_sines[i] += sign * (2 * np.pi * f)**deriv * a * np.sin((2 * np.pi * f * (t - mean_t)) + ph + ph_cos)
    return model_sines


@nb.njit(cache=True)
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
    i_sectors: numpy.ndarray[int]
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


@nb.njit(cache=True)
def measure_crossing_time(times, signal, p_orb, const, slope, f_n, a_n, ph_n, timings, noise_level, i_sectors):
    """Determine the noise level crossing time of the eclipse slopes

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
    timings: numpy.ndarray[float]
        Eclipse timings: minima, first/last contact points, internal tangency and depths,
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2
    noise_level: float
        The noise level (standard deviation of the residuals)
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).

    Returns
    -------
    t_1_i_err: float
        Noise crossing time of primary first/last contact
    t_2_i_err: float
        Noise crossing time of secondary first/last contact

    Notes
    -----
    The given sinusoids must be those from after disentangling,
    so that they don't contain the eclipses anymore.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2 = timings
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    model_sines = sum_sines(times, f_n, a_n, ph_n)
    model_line = linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_sines - model_line
    # use the eclipse model to find the derivative peaks
    t_folded, _, _ = fold_time_series(times, p_orb, t_1, t_ext_1=0, t_ext_2=0)
    mask_1_1 = (t_folded > t_1_1 - t_1 + p_orb) & (t_folded < t_b_1_1 - t_1 + p_orb)
    mask_1_2 = (t_folded > t_b_1_2 - t_1) & (t_folded < t_1_2 - t_1)
    mask_2_1 = (t_folded > t_2_1 - t_1) & (t_folded < t_b_2_1 - t_1)
    mask_2_2 = (t_folded > t_b_2_2 - t_1) & (t_folded < t_2_2 - t_1)
    # get timing error by dividing noise level by slopes
    if (np.sum(mask_1_1) > 2):
        y_inter, slope = linear_pars(t_folded[mask_1_1], ecl_signal[mask_1_1], np.array([[0, len(t_folded[mask_1_1])]]))
    else:
        slope = np.array([depth_1 / (t_b_1_1 - t_1_1)])
    t_1_1_err = abs(noise_level / slope[0])
    if (np.sum(mask_1_2) > 2):
        y_inter, slope = linear_pars(t_folded[mask_1_2], ecl_signal[mask_1_2], np.array([[0, len(t_folded[mask_1_2])]]))
    else:
        slope = np.array([depth_1 / (t_1_2 - t_b_1_2)])
    t_1_2_err = abs(noise_level / slope[0])
    if (np.sum(mask_2_1) > 2):
        y_inter, slope = linear_pars(t_folded[mask_2_1], ecl_signal[mask_2_1], np.array([[0, len(t_folded[mask_2_1])]]))
    else:
        slope = np.array([depth_2 / (t_b_2_1 - t_2_1)])
    t_2_1_err = abs(noise_level / slope[0])
    if (np.sum(mask_2_2) > 2):
        y_inter, slope = linear_pars(t_folded[mask_2_2], ecl_signal[mask_2_2], np.array([[0, len(t_folded[mask_2_2])]]))
    else:
        slope = np.array([depth_2 / (t_2_2 - t_b_2_2)])
    t_2_2_err = abs(noise_level / slope[0])
    # assume the slopes to be the same on both sides, so average them
    t_1_i_err = (t_1_1_err + t_1_2_err) / 2
    t_2_i_err = (t_2_1_err + t_2_2_err) / 2
    return t_1_i_err, t_2_i_err


@nb.njit(cache=True)
def measure_depth_error(times, signal, p_orb, const, slope, f_n, a_n, ph_n, timings, timings_err, noise_level,
                        i_sectors):
    """Estimate the error in the depth measurements based on the noise level.

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
    timings: numpy.ndarray[float]
        Eclipse timings: minima, first/last contact points, internal tangency,
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2
    timings_err: numpy.ndarray[float]
        Error estimates for the eclipse timings,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err
    noise_level: float
        The noise level (standard deviation of the residuals)
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).

    Returns
    -------
    depth_1_err: float
        Error in the depth of primary minimum
    depth_2_err: float
        Error in the depth of secondary minimum
    
    Notes
    -----
    The given sinusoids must be those from after disentangling,
    so that they don't contain the eclipses anymore.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err = timings_err
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    model_sines = sum_sines(times, f_n, a_n, ph_n)
    model_line = linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_sines - model_line
    # determine depth errors
    dur_b_1_err = np.sqrt(t_1_1_err**2 + t_1_2_err**2)
    dur_b_2_err = np.sqrt(t_2_1_err**2 + t_2_2_err**2)
    # use the full bottom if nonzero
    t_folded, _, _ = fold_time_series(times, p_orb, 0, t_ext_1=0, t_ext_2=0)
    if (t_b_1_2 - t_b_1_1 > dur_b_1_err):
        mask_b_1 = ((t_folded > t_b_1_1) & (t_folded < t_b_1_2))
        mask_b_1 = mask_b_1 | ((t_folded > p_orb + t_b_1_1) & (t_folded < p_orb + t_b_1_2))
    else:
        mask_b_1 = ((t_folded > t_1 - t_1_1_err) & (t_folded < t_1 + t_1_2_err))
        mask_b_1 = mask_b_1 | ((t_folded > p_orb + t_1 - t_1_1_err) & (t_folded < p_orb + t_1 + t_1_2_err))
    if (t_b_2_2 - t_b_2_1 > dur_b_2_err):
        mask_b_2 = ((t_folded > t_b_2_1) & (t_folded < t_b_2_2))
    else:
        mask_b_2 = ((t_folded > t_2 - t_2_1_err) & (t_folded < t_2 + t_2_2_err))
    # determine heights at the bottom and errors
    if (np.sum(mask_b_1) > 2):
        height_b_1_err = np.std(ecl_signal[mask_b_1])
    else:
        height_b_1_err = noise_level
    if (np.sum(mask_b_2) > 2):
        height_b_2_err = np.std(ecl_signal[mask_b_2])
    else:
        height_b_2_err = noise_level
    # heights at the edges
    height_1_1_err = noise_level
    height_1_2_err = noise_level
    height_2_1_err = noise_level
    height_2_2_err = noise_level
    # calculate depths
    depth_1_err = np.sqrt(height_1_1_err**2/4 + height_1_2_err**2/4 + height_b_1_err**2)
    depth_2_err = np.sqrt(height_2_1_err**2/4 + height_2_2_err**2/4 + height_b_2_err**2)
    return depth_1_err, depth_2_err


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
    peak is oversampled by a factor 100 to get a precise measurement.
    
    If and only if the full periodogram is calculated using the defaults
    for f0 and fn, the fast implementation of astropy scargle is used.
    It is accurate to a very high degree when used like this and gives
    a significant speed increase.
    """
    df = 0.1 / np.ptp(times)  # default frequency sampling is about 1/10 of frequency resolution
    # full LS periodogram
    if (f0 == 0) & (fn == 0):
        # inconsistency with astropy_scargle for small freq intervals, so only do the full pd
        freqs, ampls = astropy_scargle(times, signal, f0=f0, fn=fn, df=df)
    else:
        freqs, ampls = scargle(times, signal, f0=f0, fn=fn, df=df)
    p1 = np.argmax(ampls)
    # check if we pick the boundary frequency
    if (p1 in [0, len(freqs) - 1]):
        if verbose:
            print(f'Edge of frequency range {freqs[p1]} at position {p1} during extraction phase 1.')
    # now refine once by increasing the frequency resolution x100
    f_left = max(freqs[p1] - df, df / 10)  # may not get too low
    f_right = freqs[p1] + df
    f_refine, a_refine = scargle(times, signal, f0=f_left, fn=f_right, df=df/100)
    p2 = np.argmax(a_refine)
    # check if we pick the boundary frequency
    if (p2 in [0, len(f_refine) - 1]):
        if verbose:
            print(f'Edge of frequency range {f_refine[p2]} at position {p2} during extraction phase 2.')
    f_final = f_refine[p2]
    a_final = a_refine[p2]
    # finally, compute the phase (and make sure it stays within + and - pi)
    ph_final = scargle_phase_single(times, signal, f_final)
    ph_final = (ph_final + np.pi) % (2 * np.pi) - np.pi
    return f_final, a_final, ph_final


@nb.njit(cache=True)
def extract_single_narrow(times, signal, f0=0, fn=0, verbose=True):
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
    peak is oversampled by a factor 100 to get a precise measurement.

    If and only if the full periodogram is calculated using the defaults
    for f0 and fn, the fast implementation of astropy scargle is used.
    It is accurate to a very high degree when used like this and gives
    a significant speed increase.
    
    Same as extract_single, but meant for narrow frequency ranges. Much
    slower on the full frequency range, even though JIT-ted.
    """
    df = 0.1 / np.ptp(times)  # default frequency sampling is about 1/10 of frequency resolution
    # full LS periodogram (over a narrow range)
    freqs, ampls = scargle(times, signal, f0=f0, fn=fn, df=df)
    p1 = np.argmax(ampls)
    # check if we pick the boundary frequency
    if (p1 in [0, len(freqs) - 1]):
        if verbose:
            print(f'Edge of frequency range {ut.float_to_str(freqs[p1], dec=2)} at position {p1} '
                  f'during extraction phase 1.')
    # now refine once by increasing the frequency resolution x100
    f_left = max(freqs[p1] - df, df / 10)  # may not get too low
    f_right = freqs[p1] + df
    f_refine, a_refine = scargle(times, signal, f0=f_left, fn=f_right, df=df/100)
    p2 = np.argmax(a_refine)
    # check if we pick the boundary frequency
    if (p2 in [0, len(f_refine) - 1]):
        if verbose:
            print(f'Edge of frequency range {ut.float_to_str(f_refine[p2], dec=2)} at position {p2} '
                  f'during extraction phase 2.')
    f_final = f_refine[p2]
    a_final = a_refine[p2]
    # finally, compute the phase (and make sure it stays within + and - pi)
    ph_final = scargle_phase_single(times, signal, f_final)
    ph_final = (ph_final + np.pi) % (2 * np.pi) - np.pi
    return f_final, a_final, ph_final


def refine_subset(times, signal, signal_err, close_f, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Refine a subset of frequencies that are within the Rayleigh criterion of each other,
    taking into account (and not changing the frequencies of) harmonics if present.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    close_f: list[int], numpy.ndarray[int]
        Indices of the subset of frequencies to be refined
    p_orb: float
        Orbital period of the eclipsing binary in days (may be 0)
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
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    const: numpy.ndarray[float]
        Updated y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        Updated slopes of a piece-wise linear curve
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
    n_g = len(close_f)  # number of frequencies being updated
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_harm = len(harmonics)
    # determine initial bic
    model_sinusoid_ncf = sum_sines(times, np.delete(f_n, close_f), np.delete(a_n, close_f), np.delete(ph_n, close_f))
    cur_resid = signal - (model_sinusoid_ncf + sum_sines(times, f_n[close_f], a_n[close_f], ph_n[close_f]))
    resid = cur_resid - linear_curve(times, const, slope, i_sectors)
    f_n_temp, a_n_temp, ph_n_temp = np.copy(f_n), np.copy(a_n), np.copy(ph_n)
    n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_f - n_harm)
    bic_prev = calc_bic(resid / signal_err, n_param)
    bic_init = bic_prev
    # stop the loop when the BIC increases
    accept = True
    while accept:
        accept = False
        # remove each frequency one at a time to re-extract them
        model_linear = linear_curve(times, const, slope, i_sectors)
        for j in close_f:
            cur_resid += sum_sines(times, np.array([f_n_temp[j]]), np.array([a_n_temp[j]]), np.array([ph_n_temp[j]]))
            resid = cur_resid - model_linear
            # if f is a harmonic, don't shift the frequency
            if j in harmonics:
                f_j = f_n_temp[j]
                a_j = scargle_ampl_single(times, resid, f_j)
                ph_j = scargle_phase_single(times, resid, f_j)
            else:
                f0 = f_n_temp[j] - freq_res
                fn = f_n_temp[j] + freq_res
                f_j, a_j, ph_j = extract_single_narrow(times, resid, f0=f0, fn=fn, verbose=verbose)
            f_n_temp[j], a_n_temp[j], ph_n_temp[j] = f_j, a_j, ph_j
            cur_resid -= sum_sines(times, np.array([f_j]), np.array([a_j]), np.array([ph_j]))
        # as a last model-refining step, redetermine the constant and slope
        const, slope = linear_pars(times, cur_resid, i_sectors)
        resid = cur_resid - linear_curve(times, const, slope, i_sectors)
        # calculate BIC before moving to the next iteration
        bic = calc_bic(resid / signal_err, n_param)
        d_bic = bic_prev - bic
        if (np.round(d_bic, 2) > 0):
            # adjust the shifted frequencies
            f_n[close_f], a_n[close_f], ph_n[close_f] = f_n_temp[close_f], a_n_temp[close_f], ph_n_temp[close_f]
            bic_prev = bic
            accept = True
        if verbose:
            print(f'N_f= {n_f}, BIC= {bic:1.2f} (delta= {d_bic:1.2f}, total= {bic_init - bic:1.2f}) '
                  f'- N_refine= {n_g}, f= {f_j:1.6f}, a= {a_j:1.6f}', end='\r')
    if verbose:
        print(f'N_f= {len(f_n)}, BIC= {bic_prev:1.2f} (total= {bic_init - bic_prev:1.2f}) - end refinement', end='\r')
    # redo the constant and slope without the last iteration of changes
    resid = signal - (model_sinusoid_ncf + sum_sines(times, f_n[close_f], a_n[close_f], ph_n[close_f]))
    const, slope = linear_pars(times, resid, i_sectors)
    return const, slope, f_n, a_n, ph_n


def extract_all(times, signal, signal_err, i_sectors, verbose=True):
    """Extract all the frequencies from a periodic signal.

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
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
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

    [Author's note] Although it is my belief that doing a non-linear
    multi-sinusoid fit at each iteration of the prewhitening is the
    best approach, it is also a very (very!) time-consuming one and this
    algorithm aims to be fast while approaching the optimal solution.
    """
    freq_res = 1.5 / np.ptp(times)  # frequency resolution
    n_sectors = len(i_sectors)
    # determine the initial bic
    cur_resid = np.copy(signal)
    const, slope = linear_pars(times, signal, i_sectors)
    resid = signal - linear_curve(times, const, slope, i_sectors)
    f_n, a_n, ph_n = np.array([[], [], []])
    n_param = 2 * n_sectors
    bic_prev = calc_bic(resid / signal_err, n_param)  # initialise current BIC to the mean (and slope) subtracted signal
    bic_init = bic_prev
    if verbose:
        print(f'N_f= {len(f_n)}, BIC= {bic_init:1.2f} (delta= N/A) - start extraction')
    # stop the loop when the BIC decreases by less than 2 (or increases)
    n_prev = -1
    while (len(f_n) > n_prev):
        n_prev = len(f_n)
        # attempt to extract the next frequency
        f_i, a_i, ph_i = extract_single(times, resid, verbose=verbose)
        # now iterate over close frequencies (around f_i) a number of times to improve them
        f_n_temp, a_n_temp, ph_n_temp = np.append(f_n, f_i), np.append(a_n, a_i), np.append(ph_n, ph_i)
        close_f = af.f_within_rayleigh(n_prev, f_n_temp, freq_res)
        model_sinusoid_r = sum_sines(times, f_n_temp[close_f], a_n_temp[close_f], ph_n_temp[close_f])
        model_sinusoid_r -= sum_sines(times, np.array([f_i]), np.array([a_i]), np.array([ph_i]))
        if (len(close_f) > 1):
            refine_out = refine_subset(times, signal, signal_err, close_f, 0, const, slope, f_n_temp, a_n_temp,
                                       ph_n_temp, i_sectors, verbose=verbose)
            const, slope, f_n_temp, a_n_temp, ph_n_temp = refine_out
        # as a last model-refining step, redetermine the constant and slope
        model_sinusoid_n = sum_sines(times, f_n_temp[close_f], a_n_temp[close_f], ph_n_temp[close_f])
        cur_resid -= (model_sinusoid_n - model_sinusoid_r)  # add the changes to the sinusoid residuals
        const, slope = linear_pars(times, cur_resid, i_sectors)
        resid = cur_resid - linear_curve(times, const, slope, i_sectors)
        # calculate BIC before moving to the next iteration
        n_param = 2 * n_sectors + 3 * (n_prev + 1)
        bic = calc_bic(resid / signal_err, n_param)
        d_bic = bic_prev - bic
        if (np.round(d_bic, 2) > 2):
            # accept the new frequency
            f_n, a_n, ph_n = np.append(f_n, f_i), np.append(a_n, a_i), np.append(ph_n, ph_i)
            # adjust the shifted frequencies
            f_n[close_f], a_n[close_f], ph_n[close_f] = f_n_temp[close_f], a_n_temp[close_f], ph_n_temp[close_f]
            bic_prev = bic
        if verbose:
            print(f'N_f= {len(f_n)}, BIC= {bic:1.2f} (delta= {d_bic:1.2f}, total= {bic_init - bic:1.2f}) - '
                  f'f= {f_i:1.6f}, a= {a_i:1.6f}', end='\r')
    if verbose:
        print(f'N_f= {len(f_n)}, BIC= {bic_prev:1.2f} (delta= {bic_init - bic_prev:1.2f}) - end extraction')
    # lastly re-determine slope and const
    cur_resid += (model_sinusoid_n - model_sinusoid_r)  # undo last change
    const, slope = linear_pars(times, cur_resid, i_sectors)
    return const, slope, f_n, a_n, ph_n


def extract_additional(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=True):
    """Extract additional frequencies starting from an existing set
    taking into account any harmonics if present.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days (may be 0)
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
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
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
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_sectors = len(i_sectors)
    n_harm = len(harmonics)
    # constant term (or y-intercept) and slope
    model_linear = linear_curve(times, const, slope, i_sectors)
    model_sinusoid = sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    f_n_temp, a_n_temp, ph_n_temp = np.copy(f_n), np.copy(a_n), np.copy(ph_n)
    n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (len(f_n) - n_harm)
    bic_prev = np.inf  # initialise previous BIC to infinity
    bic = calc_bic(resid / signal_err, n_param)  # current BIC
    bic_init = bic
    d_bic = bic_prev - bic  # delta-BIC
    if verbose:
        print(f'N_f= {len(f_n_temp)}, BIC= {bic_init:1.2f} (delta= N/A) - start extraction')
    # stop the loop when the BIC decreases by less than 2 (or increases)
    i = 0
    while (bic_prev - bic > 2):
        # last frequency is accepted
        f_n, a_n, ph_n = f_n_temp, a_n_temp, ph_n_temp
        bic_prev = bic
        # attempt to extract the next frequency
        f_i, a_i, ph_i = extract_single(times, resid, verbose=verbose)
        f_n_temp, a_n_temp, ph_n_temp = np.append(f_n_temp, f_i), np.append(a_n_temp, a_i), np.append(ph_n_temp, ph_i)
        # now iterate over close frequencies (around f_i) a number of times to improve them
        close_f = af.f_within_rayleigh(i, f_n_temp, freq_res)
        if (i > 0) & (len(close_f) > 1):
            refine_out = refine_subset(times, signal, signal_err, close_f, p_orb, const, slope, f_n_temp,
                                       a_n_temp, ph_n_temp, i_sectors, verbose=verbose)
            const, slope, f_n_temp, a_n_temp, ph_n_temp = refine_out
        # as a last model-refining step, redetermine the constant and slope
        model_sinusoid = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)
        const, slope = linear_pars(times, signal - model_sinusoid, i_sectors)
        model_linear = linear_curve(times, const, slope, i_sectors)
        # now subtract all from the signal and calculate BIC before moving to the next iteration
        resid = signal - model_linear - model_sinusoid
        n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (len(f_n_temp) - n_harm)
        bic = calc_bic(resid / signal_err, n_param)
        d_bic = bic_prev - bic  # delta-BIC
        i += 1
        if verbose:
            print(f'N_f= {len(f_n_temp)}, BIC= {bic:1.2f} (delta= {d_bic:1.2f}, total= {bic_init - bic:1.2f}) - '
                  f'f= {f_i:1.6f}, a= {a_i:1.6f}', end='\r')
    if verbose:
        print(f'N_f= {len(f_n)}, BIC= {bic_prev:1.2f} (delta= {bic_init - bic_prev:1.2f}) - end extraction')
    # redo the constant and slope without the last iteration frequencies
    model_sinusoid = sum_sines(times, f_n, a_n, ph_n)
    const, slope = linear_pars(times, signal - model_sinusoid, i_sectors)
    return const, slope, f_n, a_n, ph_n


def extract_additional_harmonics(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors,
                                 verbose=False):
    """Tries to extract more harmonics from the signal
    
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
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    const: numpy.ndarray[float]
        (Updated) y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        (Updated) slopes of a piece-wise linear curve
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
    # extract the existing harmonics using the period
    if (len(f_n) > 0):
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    else:
        harmonics, harmonic_n = np.array([], dtype=int), np.array([], dtype=int)
    # make a list of not-present possible harmonics
    h_candidate = np.arange(1, p_orb * f_max, dtype=int)
    h_candidate = np.delete(h_candidate, harmonic_n - 1)  # harmonic_n minus one is the position
    # initial residuals
    model_linear = linear_curve(times, const, slope, i_sectors)
    model_sinusoid = sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    n_param_orig = 3 * len(f_n) + 2 * len(harmonics) + 1  # harmonics have 1 less free parameter
    bic_prev = calc_bic(resid / signal_err, n_param_orig)
    # loop over candidates and try to extract
    n_accepted = 0
    n_h_acc = []
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
        const, slope = linear_pars(times, signal - model, i_sectors)
        # determine new BIC and whether it improved
        model_linear = linear_curve(times, const, slope, i_sectors)
        model_sinusoid = sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)
        resid = signal - model_linear - model_sinusoid
        n_param = n_param_orig + 2 * (n_accepted + 1)
        bic = calc_bic(resid / signal_err, n_param)
        if (np.round(bic_prev - bic, 2) > 2):
            # h_c is accepted, add it to the final list and continue
            bic_prev = bic
            f_n, a_n, ph_n = np.copy(f_n_temp), np.copy(a_n_temp), np.copy(ph_n_temp)
            n_accepted += 1
            n_h_acc.append(h_c)
        else:
            # h_c is rejected, revert to previous residual
            model_sinusoid = sum_sines(times, f_n, a_n, ph_n)
            const, slope = linear_pars(times, signal - model_sinusoid, i_sectors)
            model_linear = linear_curve(times, const, slope, i_sectors)
            resid = signal - model_linear - model_sinusoid
    if verbose:
        print(f'N_f= {len(f_n)}, BIC= {bic_prev:1.2f} - Successfully extracted harmonics {n_h_acc}')
    return const, slope, f_n, a_n, ph_n


def fix_harmonic_frequency(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Fixes the frequency of harmonics to the theoretical value, then
    re-determines the amplitudes and phases.

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
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).

    Returns
    -------
    const: numpy.ndarray[float]
        (Updated) y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        (Updated) slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        (Updated) frequencies of the same number of sine waves
    a_n: numpy.ndarray[float]
        (Updated) amplitudes of the same number of sine waves
    ph_n: numpy.ndarray[float]
        (Updated) phases of the same number of sine waves
    """
    # extract the harmonics using the period and determine some numbers
    freq_res = 1.5 / np.ptp(times)
    harmonics, harmonic_n = af.find_harmonics_tolerance(f_n, p_orb, f_tol=freq_res / 2)
    if (len(harmonics) == 0):
        raise ValueError('No harmonic frequencies found')
    
    n_sectors = len(i_sectors)
    n_freq = len(f_n)
    n_harm_init = len(harmonics)
    # indices of harmonic candidates to remove
    remove_harm_c = np.zeros(0, dtype=np.int_)
    f_new, a_new, ph_new = np.zeros((3, 0))
    # determine initial bic
    model_linear = linear_curve(times, const, slope, i_sectors)
    model_sinusoid = sum_sines(times, f_n, a_n, ph_n)
    cur_resid = signal - model_sinusoid  # the residual after subtracting the model of sinusoids
    resid = cur_resid - model_linear
    n_param = 2 * n_sectors + 1 + 2 * n_harm_init + 3 * (n_freq - n_harm_init)
    bic_init = calc_bic(resid / signal_err, n_param)
    # go through the harmonics by harmonic number and re-extract them (removing all duplicate n's in the process)
    for n in np.unique(harmonic_n):
        remove = np.arange(len(f_n))[harmonics][harmonic_n == n]
        # make a model of the removed sinusoids and subtract it from the full sinusoid residual
        model_sinusoid_r = sum_sines(times, f_n[remove], a_n[remove], ph_n[remove])
        cur_resid += model_sinusoid_r
        const, slope = linear_pars(times, resid, i_sectors)  # redetermine const and slope
        model_linear = linear_curve(times, const, slope, i_sectors)
        resid = cur_resid - model_linear
        # calculate the new harmonic
        f_i = n / p_orb  # fixed f
        a_i = scargle_ampl_single(times, resid, f_i)
        ph_i = scargle_phase_single(times, resid, f_i)
        ph_i = np.mod(ph_i + np.pi, 2 * np.pi) - np.pi  # make sure the phase stays within + and - pi
        # make a model of the new sinusoid and add it to the full sinusoid residual
        model_sinusoid_n = sum_sines(times, np.array([f_i]), np.array([a_i]), np.array([ph_i]))
        cur_resid -= model_sinusoid_n
        # add to freq list and removal list
        f_new, a_new, ph_new = np.append(f_new, f_i), np.append(a_new, a_i), np.append(ph_new, ph_i)
        remove_harm_c = np.append(remove_harm_c, remove)
    # lastly re-determine slope and const
    const, slope = linear_pars(times, cur_resid, i_sectors)
    # finally, remove all the designated sinusoids from the lists and add the new ones
    f_n = np.append(np.delete(f_n, remove_harm_c), f_new)
    a_n = np.append(np.delete(a_n, remove_harm_c), a_new)
    ph_n = np.append(np.delete(ph_n, remove_harm_c), ph_new)
    # re-extract the non-harmonics
    n_freq = len(f_n)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(n_freq), harmonics)
    n_harm = len(harmonics)
    for i in non_harm:
        # make a model of the removed sinusoid and subtract it from the full sinusoid residual
        model_sinusoid_r = sum_sines(times, np.array([f_n[i]]), np.array([a_n[i]]), np.array([ph_n[i]]))
        cur_resid += model_sinusoid_r
        const, slope = linear_pars(times, cur_resid, i_sectors)  # redetermine const and slope
        model_linear = linear_curve(times, const, slope, i_sectors)
        resid = cur_resid - model_linear
        # extract the updated frequency
        fl, fr = f_n[i] - freq_res, f_n[i] + freq_res
        f_n[i], a_n[i], ph_n[i] = extract_single_narrow(times, resid, f0=fl, fn=fr, verbose=verbose)
        ph_n[i] = np.mod(ph_n[i] + np.pi, 2 * np.pi) - np.pi  # make sure the phase stays within + and - pi
        # make a model of the new sinusoid and add it to the full sinusoid residual
        model_sinusoid_n = sum_sines(times, np.array([f_n[i]]), np.array([a_n[i]]), np.array([ph_n[i]]))
        cur_resid -= model_sinusoid_n
    # lastly re-determine slope and const
    const, slope = linear_pars(times, cur_resid, i_sectors)
    
    # todo: refine subset
    # out = refine_subset(times, signal, signal_err, non_harm, p_orb, const, slope, f_n, a_n, ph_n, i_sectors,
    #                     verbose=verbose)
    # const, slope, f_n, a_n, ph_n = out
    if verbose:
        model_linear = linear_curve(times, const, slope, i_sectors)
        resid = cur_resid - model_linear
        n_param = 2 * n_sectors + 1 + 2 * n_harm + 3 * (n_freq - n_harm)
        bic = calc_bic(resid / signal_err, n_param)
        print(f'Candidate harmonics replaced: {n_harm_init} ({n_harm} left). ')
        print(f'N_f= {len(f_n)}, BIC= {bic:1.2f} (delta= {bic_init - bic:1.2f})')
    return const, slope, f_n, a_n, ph_n


@nb.njit(cache=True)
def remove_frequencies_single(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Attempt the removal of individual frequencies
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days (may be 0)
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
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    const: numpy.ndarray[float]
        (Updated) y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        (Updated) slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        (Updated) frequencies of a (lower) number of sine waves
    a_n: numpy.ndarray[float]
        (Updated) amplitudes of a (lower) number of sine waves
    ph_n: numpy.ndarray[float]
        (Updated) phases of a (lower) number of sine waves
    
    Notes
    -----
    Checks whether the BIC can be improved by removing a frequency.
    Harmonics are taken into account.
    """
    n_sectors = len(i_sectors)
    n_freq = len(f_n)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(n_freq), harmonics)
    n_harm = len(harmonics)
    # indices of single frequencies to remove
    remove_single = np.zeros(0, dtype=np.int_)
    # determine initial bic
    model_linear = linear_curve(times, const, slope, i_sectors)
    model_sinusoid = sum_sines(times, f_n, a_n, ph_n)
    cur_resid = signal - model_sinusoid  # the residual after subtracting the model of sinusoids
    resid = cur_resid - model_linear
    n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_freq - n_harm)
    bic_prev = calc_bic(resid / signal_err, n_param)
    bic_init = bic_prev
    n_prev = -1
    # while frequencies are added to the remove list, continue loop
    while (len(remove_single) > n_prev):
        n_prev = len(remove_single)
        for i in range(n_freq):
            if i in remove_single:
                continue
            
            # make a model of the removed sinusoids and subtract it from the full sinusoid model
            model_sinusoid_r = sum_sines(times, np.array([f_n[i]]), np.array([a_n[i]]), np.array([ph_n[i]]))
            resid = cur_resid + model_sinusoid_r
            const, slope = linear_pars(times, resid, i_sectors)  # redetermine const and slope
            model_linear = linear_curve(times, const, slope, i_sectors)
            resid -= model_linear
            # number of parameters and bic
            n_harm_i = n_harm - len([h for h in remove_single if h in harmonics]) - 1 * (i in harmonics)
            n_freq_i = n_freq - len(remove_single) - 1 - n_harm_i
            n_param = 2 * n_sectors + 1 * (n_harm_i > 0) + 2 * n_harm_i + 3 * n_freq_i
            bic = calc_bic(resid / signal_err, n_param)
            # if improvement, add to list of removed freqs
            if (np.round(bic_prev - bic, 2) > 0):
                remove_single = np.append(remove_single, i)
                cur_resid += model_sinusoid_r
                bic_prev = bic
    # lastly re-determine slope and const
    const, slope = linear_pars(times, cur_resid, i_sectors)
    # finally, remove all the designated sinusoids from the lists
    f_n = np.delete(f_n, remove_single)
    a_n = np.delete(a_n, remove_single)
    ph_n = np.delete(ph_n, remove_single)
    if verbose:
        str_bic = ut.float_to_str(bic_prev, dec=2)
        str_delta = ut.float_to_str(bic_init - bic_prev, dec=2)
        print(f'Single frequencies removed: {n_freq - len(f_n)}')
        print(f'N_f= {len(f_n)}, BIC= {str_bic} (delta= {str_delta})')
    return const, slope, f_n, a_n, ph_n


@nb.njit(cache=True)
def replace_frequency_groups(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Attempt the replacement of groups of frequencies by a single one

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days (may be 0)
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
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    const: numpy.ndarray[float]
        (Updated) y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        (Updated) slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        (Updated) frequencies of a (lower) number of sine waves
    a_n: numpy.ndarray[float]
        (Updated) amplitudes of a (lower) number of sine waves
    ph_n: numpy.ndarray[float]
        (Updated) phases of a (lower) number of sine waves

    Notes
    -----
    Checks whether the BIC can be improved by replacing a group of
    frequencies by only one. Harmonics are never removed.
    """
    freq_res = 1.5 / np.ptp(times)  # frequency resolution
    n_sectors = len(i_sectors)
    n_freq = len(f_n)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(n_freq), harmonics)
    n_harm = len(harmonics)
    # make an array of sets of frequencies (non-harmonic) to be investigated for replacement
    close_f_groups = af.chains_within_rayleigh(f_n[non_harm], freq_res)
    close_f_groups = [non_harm[group] for group in close_f_groups]  # convert to the right indices
    f_sets = [g[np.arange(p1, p2 + 1)]
              for g in close_f_groups
              for p1 in range(len(g) - 1)
              for p2 in range(p1 + 1, len(g))]
    # make an array of sets of frequencies (now with harmonics) to be investigated for replacement
    close_f_groups = af.chains_within_rayleigh(f_n, freq_res)
    f_sets_h = [g[np.arange(p1, p2 + 1)]
                for g in close_f_groups
                for p1 in range(len(g) - 1)
                for p2 in range(p1 + 1, len(g))
                if np.any(np.array([g_f in harmonics for g_f in g[np.arange(p1, p2 + 1)]]))]
    # join the two lists, and remember which is which
    harm_sets = np.arange(len(f_sets), len(f_sets) + len(f_sets_h))
    f_sets.extend(f_sets_h)
    remove_sets = np.zeros(0, dtype=np.int_)  # sets of frequencies to replace (by 1 freq)
    used_sets = np.zeros(0, dtype=np.int_)  # sets that are not to be examined anymore
    f_new, a_new, ph_new = np.zeros((3, 0))
    # determine initial bic
    model_linear = linear_curve(times, const, slope, i_sectors)
    model_sinusoid = sum_sines(times, f_n, a_n, ph_n)
    cur_resid = signal - model_sinusoid  # the residual after subtracting the model of sinusoids
    resid = cur_resid - model_linear
    n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_freq - n_harm)
    bic_prev = calc_bic(resid / signal_err, n_param)
    bic_init = bic_prev
    n_prev = -1
    # while frequencies are added to the remove list, continue loop
    while (len(remove_sets) > n_prev):
        n_prev = len(remove_sets)
        for i, set_i in enumerate(f_sets):
            if i in used_sets:
                continue
            
            # temporarily add this iteration to the remove indices
            remove = np.append([k for j in remove_sets for k in f_sets[j]], set_i).astype(np.int_)
            # make a model of the removed and new sinusoids and subtract/add it from/to the full sinusoid model
            model_sinusoid_r = sum_sines(times, f_n[set_i], a_n[set_i], ph_n[set_i])
            resid = cur_resid + model_sinusoid_r
            const, slope = linear_pars(times, resid, i_sectors)  # redetermine const and slope
            model_linear = linear_curve(times, const, slope, i_sectors)
            resid -= model_linear
            # extract a single freq to try replacing the set
            if i in harm_sets:
                harm_i = np.array([h for h in set_i if h in harmonics])
                f_i = f_n[harm_i]  # fixed f
                a_i = scargle_ampl(times, resid, f_n[harm_i])
                ph_i = scargle_phase(times, resid, f_n[harm_i])
            else:
                edges = [min(f_n[set_i]) - freq_res, max(f_n[set_i]) + freq_res]
                out = extract_single_narrow(times, resid, f0=edges[0], fn=edges[1], verbose=verbose)
                f_i, a_i, ph_i = np.array([out[0]]), np.array([out[1]]), np.array([out[2]])
            # make a model including the new freq
            model_sinusoid_n = sum_sines(times, f_i, a_i, ph_i)
            resid = cur_resid - model_sinusoid_n
            const, slope = linear_pars(times, resid, i_sectors)  # redetermine const and slope
            model_linear = linear_curve(times, const, slope, i_sectors)
            resid -= model_linear
            # number of parameters and bic
            n_freq_i = n_freq - sum([len(f_sets[j]) for j in remove_sets]) - len(set_i) + len(f_new) + len(f_i) - n_harm
            n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * n_freq_i
            bic = calc_bic(resid / signal_err, n_param)
            if (np.round(bic_prev - bic, 2) > 0):
                # do not look at sets with the same freqs as the just removed set anymore
                overlap = [j for j, subset in enumerate(f_sets) if np.any(np.array([k in set_i for k in subset]))]
                used_sets = np.unique(np.append(used_sets, overlap))
                # add to list of removed sets
                remove_sets = np.append(remove_sets, i)
                # remember the new frequency (or the current one if it is a harmonic)
                f_new, a_new, ph_new = np.append(f_new, f_i), np.append(a_new, a_i), np.append(ph_new, ph_i)
                cur_resid += model_sinusoid_r - model_sinusoid_n
                bic_prev = bic
    # lastly re-determine slope and const
    const, slope = linear_pars(times, cur_resid, i_sectors)
    # finally, remove all the designated sinusoids from the lists and add the new ones
    f_n = np.append(np.delete(f_n, [k for i in remove_sets for k in f_sets[i]]), f_new)
    a_n = np.append(np.delete(a_n, [k for i in remove_sets for k in f_sets[i]]), a_new)
    ph_n = np.append(np.delete(ph_n, [k for i in remove_sets for k in f_sets[i]]), ph_new)
    if verbose:
        str_bic = ut.float_to_str(bic_prev, dec=2)
        str_delta = ut.float_to_str(bic_init - bic_prev, dec=2)
        print(f'Frequency sets replaced by a single frequency: {len(remove_sets)} ({n_freq - len(f_n)} frequencies). ')
        print(f'N_f= {len(f_n)}, BIC= {str_bic} (delta= {str_delta})')
    return const, slope, f_n, a_n, ph_n


def reduce_frequencies(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Attempt to reduce the number of frequencies taking into account any harmonics if present.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days (may be 0)
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
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    const: numpy.ndarray[float]
        (Updated) y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        (Updated) slopes of a piece-wise linear curve
    f_n: numpy.ndarray[float]
        (Updated) frequencies of a (lower) number of sine waves
    a_n: numpy.ndarray[float]
        (Updated) amplitudes of a (lower) number of sine waves
    ph_n: numpy.ndarray[float]
        (Updated) phases of a (lower) number of sine waves
    
    Notes
    -----
    Checks whether the BIC can be improved by removing a frequency. Special attention
    is given to frequencies that are within the Rayleigh criterion of each other.
    It is attempted to replace these by a single frequency.
    """
    # first check if any frequency can be left out (after the fit, this may be possible)
    out_a = remove_frequencies_single(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors,
                                      verbose=verbose)
    const, slope, f_n, a_n, ph_n = out_a
    # Now go on to trying to replace sets of frequencies that are close together
    out_b = replace_frequency_groups(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors,
                                      verbose=verbose)
    const, slope, f_n, a_n, ph_n = out_b
    return const, slope, f_n, a_n, ph_n
