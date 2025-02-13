"""STAR SHADOW
Satellite Time series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This Python module contains functions for time series analysis;
specifically for the fitting of stellar oscillations and eclipses.

Notes
-----
Minimize methods:
Nelder-Mead is extensively tested and found robust, while slow.
TNC is tested, and seems reliable while being fast, though slightly worse BIC results.
L-BFGS-B is tested, and seems reliable while being fast, though slightly worse BIC results.
See publication appendix for more information.

Code written by: Luc IJspeert
"""

import numpy as np
import numba as nb
import scipy as sp
import scipy.optimize

try:
    import ellc  # optional functionality
except ImportError:
    ellc = None

from . import timeseries_functions as tsf
from . import analysis_functions as af
from . import utility as ut


@nb.njit(cache=True)
def dsin_dx(two_pi_t, f, a, ph, d='f', p_orb=0):
    """The derivative of a sine wave at times t,
    where x is on of the parameters.

    Parameters
    ----------
    two_pi_t: numpy.ndarray[float]
        Timestamps of the time series times two pi
    f: float
        The frequency of a sine wave
    a: float
        The amplitude of a sine wave
    ph: float
        The phase of a sine wave
    d: string
        Which derivative to take
        Choose f, a, ph, p_orb
    p_orb: float
        Orbital period of the eclipsing binary in days

    Returns
    -------
    model_deriv: numpy.ndarray[float]
        Model time series of the derivative of a sine wave to f.

    Notes
    -----
    Make sure the phases correspond to the given
    time zero point.
    If d='p_orb', it is assumed that f is a harmonic
    """
    if d == 'f':
        model_deriv = a * np.cos(two_pi_t * f + ph) * two_pi_t
    elif d == 'a':
        model_deriv = np.sin(two_pi_t * f + ph)
    elif d == 'ph':
        model_deriv = a * np.cos(two_pi_t * f + ph)
    elif d == 'p_orb':
        model_deriv = a * np.cos(two_pi_t * f + ph) * two_pi_t * f / p_orb
    else:
        model_deriv = np.zeros(len(two_pi_t))
    return model_deriv


@nb.njit(cache=True)
def objective_sinusoids(params, times, signal, i_sectors):
    """The objective function to give to scipy.optimize.minimize for a sum of sine waves.

    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of a set of sine waves and linear curve(s)
        Has to be a flat array and are ordered in the following way:
        [constant1, constant2, ..., slope1, slope2, ...,
         freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...]
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
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the residuals

    See Also
    --------
    linear_curve and sum_sines for the definition of the parameters.
    """
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_sect) // 3  # each sine has freq, ampl and phase
    # separate the parameters
    const = params[:n_sect]
    slope = params[n_sect:2 * n_sect]
    freqs = params[2 * n_sect:2 * n_sect + n_sin]
    ampls = params[2 * n_sect + n_sin:2 * n_sect + 2 * n_sin]
    phases = params[2 * n_sect + 2 * n_sin:2 * n_sect + 3 * n_sin]
    # make the linear and sinusoid model
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, freqs, ampls, phases)
    # calculate the likelihood (minus this for minimisation)
    resid = signal - model_linear - model_sinusoid
    ln_likelihood = tsf.calc_likelihood(resid, times=times, signal_err=None, iid=True)
    return -ln_likelihood


@nb.njit(cache=True)
def jacobian_sinusoids(params, times, signal, i_sectors):
    """The jacobian function to give to scipy.optimize.minimize for a sum of sine waves.
    
    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of a set of sine waves and linear curve(s)
        Has to be a flat array and are ordered in the following way:
        [constant1, constant2, ..., slope1, slope2, ...,
         freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...]
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
    jac: float
        The derivative of minus the (natural)log-likelihood of the residuals

    See Also
    --------
    objective_sinusoids
    """
    times_ms = times - np.mean(times)
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_sect) // 3  # each sine has freq, ampl and phase
    # separate the parameters
    const = params[:n_sect]
    slope = params[n_sect:2 * n_sect]
    freqs = params[2 * n_sect:2 * n_sect + n_sin]
    ampls = params[2 * n_sect + n_sin:2 * n_sect + 2 * n_sin]
    phases = params[2 * n_sect + 2 * n_sin:2 * n_sect + 3 * n_sin]
    # make the linear and sinusoid model
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, freqs, ampls, phases)
    # calculate the likelihood derivative (minus this for minimisation)
    resid = signal - model_linear - model_sinusoid
    two_pi_t = 2 * np.pi * times_ms
    # factor 1 of df/dx: -n / S
    df_1a = np.zeros(n_sect)  # calculated per sector
    df_1b = -len(times) / np.sum(resid**2)
    # calculate the rest of the jacobian for the linear parameters, factor 2 of df/dx:
    df_2a = np.zeros(2 * n_sect)
    for i, (co, sl, s) in enumerate(zip(const, slope, i_sectors)):
        i_s = i + n_sect
        df_1a[i] = -len(times[s[0]:s[1]]) / np.sum(resid[s[0]:s[1]]**2)
        df_2a[i] = np.sum(resid[s[0]:s[1]])
        df_2a[i_s] = np.sum(resid[s[0]:s[1]] * (times[s[0]:s[1]] - np.mean(times[s[0]:s[1]])))
    df_1a = np.append(df_1a, df_1a)  # copy to double length
    jac_lin = df_1a * df_2a
    # calculate the rest of the jacobian for the sinusoid parameters, factor 2 of df/dx:
    df_2b = np.zeros(3 * n_sin)
    for i, (f, a, ph) in enumerate(zip(freqs, ampls, phases)):
        i_a = i + n_sin
        i_ph = i + 2 * n_sin
        df_2b[i] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='f'))
        df_2b[i_a] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='a'))
        df_2b[i_ph] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='ph'))
    # jacobian = df/dx = df/dy * dy/dx (f is objective function, y is model)
    jac_sin = df_1b * df_2b
    jac = np.append(jac_lin, jac_sin)
    return jac


def fit_multi_sinusoid(times, signal, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Perform the multi-sinusoid, non-linear least-squares fit.

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
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
    res_const: numpy.ndarray[float]
        Updated y-intercepts of a piece-wise linear curve
    res_slope: numpy.ndarray[float]
        Updated slopes of a piece-wise linear curve
    res_freqs: numpy.ndarray[float]
        Updated frequencies of a number of sine waves
    res_ampls: numpy.ndarray[float]
        Updated amplitudes of a number of sine waves
    res_phases: numpy.ndarray[float]
        Updated phases of a number of sine waves

    Notes
    -----
    Strictly speaking it is doing a maximum log-likelihood fit, but that is
    in essence identical (and numerically more stable due to the logarithm).
    """
    res_const = np.copy(np.atleast_1d(const))
    res_slope = np.copy(np.atleast_1d(slope))
    res_freqs = np.copy(f_n)
    res_ampls = np.copy(a_n)
    res_phases = np.copy(ph_n)
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_sin = len(f_n)  # each sine has freq, ampl and phase
    # we don't want the frequencies to go lower than about 1/T/100
    t_tot = np.ptp(times)
    f_low = 0.01 / t_tot
    # do the fit
    par_init = np.concatenate((res_const, res_slope, res_freqs, res_ampls, res_phases))
    par_bounds = [(None, None) for _ in range(2 * n_sect)]
    par_bounds = par_bounds + [(f_low, None) for _ in range(n_sin)]
    par_bounds = par_bounds + [(0, None) for _ in range(n_sin)] + [(None, None) for _ in range(n_sin)]
    arguments = (times, signal, i_sectors)
    result = sp.optimize.minimize(objective_sinusoids, jac=jacobian_sinusoids, x0=par_init, args=arguments,
                                  method='L-BFGS-B', bounds=par_bounds, options={'maxiter': 10**4 * len(par_init)})
    # separate results
    res_const = result.x[0:n_sect]
    res_slope = result.x[n_sect:2 * n_sect]
    res_freqs = result.x[2 * n_sect:2 * n_sect + n_sin]
    res_ampls = result.x[2 * n_sect + n_sin:2 * n_sect + 2 * n_sin]
    res_phases = result.x[2 * n_sect + 2 * n_sin:2 * n_sect + 3 * n_sin]
    if verbose:
        model_linear = tsf.linear_curve(times, res_const, res_slope, i_sectors)
        model_sinusoid = tsf.sum_sines(times, res_freqs, res_ampls, res_phases)
        resid = signal - model_linear - model_sinusoid
        bic = tsf.calc_bic(resid, 2 * n_sect + 3 * n_sin)
        print(f'Fit convergence: {result.success} - BIC: {bic:1.2f}. '
              f'N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')
    return res_const, res_slope, res_freqs, res_ampls, res_phases


def fit_multi_sinusoid_per_group(times, signal, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Perform the multi-sinusoid, non-linear least-squares fit per frequency group

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
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
    res_const: numpy.ndarray[float]
        Updated y-intercepts of a piece-wise linear curve
    res_slope: numpy.ndarray[float]
        Updated slopes of a piece-wise linear curve
    res_freqs: numpy.ndarray[float]
        Updated frequencies of a number of sine waves
    res_ampls: numpy.ndarray[float]
        Updated amplitudes of a number of sine waves
    res_phases: numpy.ndarray[float]
        Updated phases of a number of sine waves

    Notes
    -----
    In reducing the overall runtime of the NL-LS fit, this will improve the
    fits per group of 15-20 frequencies, leaving the other frequencies as
    fixed parameters.
    """
    f_groups = ut.group_frequencies_for_fit(a_n, g_min=20, g_max=25)
    n_groups = len(f_groups)
    n_sect = len(i_sectors)
    n_sin = len(f_n)
    # make a copy of the initial parameters
    res_const = np.copy(const)
    res_slope = np.copy(slope)
    res_freqs = np.copy(f_n)
    res_ampls = np.copy(a_n)
    res_phases = np.copy(ph_n)
    # update the parameters for each group
    for k, group in enumerate(f_groups):
        if verbose:
            print(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)}', end='\r')
        # subtract all other sines from the data, they are fixed now
        resid = signal - tsf.sum_sines(times, np.delete(res_freqs, group), np.delete(res_ampls, group),
                                       np.delete(res_phases, group))
        # fit only the frequencies in this group (constant and slope are also fitted still)
        output = fit_multi_sinusoid(times, resid, res_const, res_slope, res_freqs[group],
                                    res_ampls[group], res_phases[group], i_sectors, verbose=False)
        res_const, res_slope, out_freqs, out_ampls, out_phases = output
        res_freqs[group] = out_freqs
        res_ampls[group] = out_ampls
        res_phases[group] = out_phases
        if verbose:
            model_linear = tsf.linear_curve(times, res_const, res_slope, i_sectors)
            model_sinusoid = tsf.sum_sines(times, res_freqs, res_ampls, res_phases)
            resid = signal - model_linear - model_sinusoid
            bic = tsf.calc_bic(resid, 2 * n_sect + 3 * n_sin)
            print(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)} - BIC: {bic:1.2f}')
    return res_const, res_slope, res_freqs, res_ampls, res_phases


@nb.njit(cache=True)
def objective_sinusoids_harmonics(params, times, signal, harmonic_n, i_sectors):
    """The objective function to give to scipy.optimize.minimize for a sum of sine waves
    plus a set of harmonic frequencies.

    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of a set of sine waves and linear curve(s).
        Has to be a flat array and are ordered in the following way:
        [p_orb, constant1, constant2, ..., slope1, slope2, ...,
         freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...,
         ampl_h1, ampl_h2, ..., phase_h1, phase_h2, ...]
        where _hi indicates harmonics.
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    harmonic_n: numpy.ndarray[int]
        Integer indicating which harmonic each index in 'harmonics'
        points to. n=1 for the base frequency (=orbital frequency)
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).

    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the residuals

    See Also
    --------
    linear_curve and sum_sines for the definition of the parameters.
    """
    n_harm = len(harmonic_n)
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_sect - 1 - 2 * n_harm) // 3  # each sine has freq, ampl and phase
    n_f_tot = n_sin + n_harm
    # separate the parameters
    p_orb = params[0]
    const = params[1:1 + n_sect]
    slope = params[1 + n_sect:1 + 2 * n_sect]
    freqs = np.zeros(n_f_tot)
    freqs[:n_sin] = params[1 + 2 * n_sect:1 + 2 * n_sect + n_sin]
    freqs[n_sin:] = harmonic_n / p_orb
    ampls = np.zeros(n_f_tot)
    ampls[:n_sin] = params[1 + 2 * n_sect + n_sin:1 + 2 * n_sect + 2 * n_sin]
    ampls[n_sin:] = params[1 + 2 * n_sect + 3 * n_sin:1 + 2 * n_sect + 3 * n_sin + n_harm]
    phases = np.zeros(n_f_tot)
    phases[:n_sin] = params[1 + 2 * n_sect + 2 * n_sin:1 + 2 * n_sect + 3 * n_sin]
    phases[n_sin:] = params[1 + 2 * n_sect + 3 * n_sin + n_harm:1 + 2 * n_sect + 3 * n_sin + 2 * n_harm]
    # make the linear and sinusoid model
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, freqs, ampls, phases)
    # calculate the likelihood (minus this for minimisation)
    resid = signal - model_linear - model_sinusoid
    ln_likelihood = tsf.calc_likelihood(resid, times=times, signal_err=None, iid=True)
    return -ln_likelihood


@nb.njit(cache=True)
def jacobian_sinusoids_harmonics(params, times, signal, harmonic_n, i_sectors):
    """The jacobian function to give to scipy.optimize.minimize for a sum of sine waves
    plus a set of harmonic frequencies.

    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of a set of sine waves and linear curve(s).
        Has to be a flat array and are ordered in the following way:
        [p_orb, constant1, constant2, ..., slope1, slope2, ...,
         freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...,
         ampl_h1, ampl_h2, ..., phase_h1, phase_h2, ...]
        where _hi indicates harmonics.
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    harmonic_n: numpy.ndarray[int]
        Integer indicating which harmonic each index in 'harmonics'
        points to. n=1 for the base frequency (=orbital frequency)
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).

    Returns
    -------
    jac: float
        The derivative of minus the (natural)log-likelihood of the residuals

    See Also
    --------
    linear_curve and sum_sines for the definition of the parameters.
    """
    times_ms = times - np.mean(times)
    n_harm = len(harmonic_n)
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_sect - 1 - 2 * n_harm) // 3  # each sine has freq, ampl and phase
    n_f_tot = n_sin + n_harm
    # separate the parameters
    p_orb = params[0]
    const = params[1:1 + n_sect]
    slope = params[1 + n_sect:1 + 2 * n_sect]
    freqs = np.zeros(n_f_tot)
    freqs[:n_sin] = params[1 + 2 * n_sect:1 + 2 * n_sect + n_sin]
    freqs[n_sin:] = harmonic_n / p_orb
    ampls = np.zeros(n_f_tot)
    ampls[:n_sin] = params[1 + 2 * n_sect + n_sin:1 + 2 * n_sect + 2 * n_sin]
    ampls[n_sin:] = params[1 + 2 * n_sect + 3 * n_sin:1 + 2 * n_sect + 3 * n_sin + n_harm]
    phases = np.zeros(n_f_tot)
    phases[:n_sin] = params[1 + 2 * n_sect + 2 * n_sin:1 + 2 * n_sect + 3 * n_sin]
    phases[n_sin:] = params[1 + 2 * n_sect + 3 * n_sin + n_harm:1 + 2 * n_sect + 3 * n_sin + 2 * n_harm]
    # make the linear and sinusoid model and subtract from the signal
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, freqs, ampls, phases)
    resid = signal - model_linear - model_sinusoid
    # common factor
    two_pi_t = 2 * np.pi * times_ms
    # factor 1 of df/dx: -n / S
    df_1a = np.zeros(n_sect)  # calculated per sector
    df_1b = -len(times) / np.sum(resid**2)
    # calculate the rest of the jacobian for the linear parameters, factor 2 of df/dx:
    df_2a = np.zeros(2 * n_sect)
    for i, (co, sl, s) in enumerate(zip(const, slope, i_sectors)):
        i_s = i + n_sect
        df_1a[i] = -len(times[s[0]:s[1]]) / np.sum(resid[s[0]:s[1]]**2)
        df_2a[i] = np.sum(resid[s[0]:s[1]])
        df_2a[i_s] = np.sum(resid[s[0]:s[1]] * (times[s[0]:s[1]] - np.mean(times[s[0]:s[1]])))
    df_1a = np.append(df_1a, df_1a)  # copy to double length
    jac_lin = df_1a * df_2a
    # calculate the rest of the jacobian, factor 2 of df/dx:
    df_2b = np.zeros(3 * n_sin + 2 * n_harm + 1)
    for i, (f, a, ph) in enumerate(zip(freqs[:n_sin], ampls[:n_sin], phases[:n_sin])):
        i_f = i + 1
        i_a = i + n_sin + 1
        i_ph = i + 2 * n_sin + 1
        df_2b[i_f] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='f'))
        df_2b[i_a] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='a'))
        df_2b[i_ph] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='ph'))
    for i, (f, a, ph) in enumerate(zip(freqs[n_sin:], ampls[n_sin:], phases[n_sin:])):
        i_a = i + 3 * n_sin + 1
        i_ph = i + 3 * n_sin + n_harm + 1
        df_2b[0] -= np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='p_orb', p_orb=p_orb))
        df_2b[i_a] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='a'))
        df_2b[i_ph] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='ph'))
    # jacobian = df/dx = df/dy * dy/dx (f is objective function, y is model)
    jac_sin = df_1b * df_2b
    jac = np.append(jac_lin, jac_sin)
    return jac


def fit_multi_sinusoid_harmonics(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Perform the multi-sinusoid, non-linear least-squares fit with harmonic frequencies.

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
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    res_p_orb: float
        Updated Orbital period in days
    res_const: numpy.ndarray[float]
        Updated y-intercepts of a piece-wise linear curve
    res_slope: numpy.ndarray[float]
        Updated slopes of a piece-wise linear curve
    res_freqs: numpy.ndarray[float]
        Updated frequencies of a number of sine waves
    res_ampls: numpy.ndarray[float]
        Updated amplitudes of a number of sine waves
    res_phases: numpy.ndarray[float]
        Updated phases of a number of sine waves

    Notes
    -----
    Strictly speaking it is doing a maximum log-likelihood fit, but that is
    in essence identical (and numerically more stable due to the logarithm).
    """
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_f_tot = len(f_n)
    n_harm = len(harmonics)
    n_sin = n_f_tot - n_harm  # each independent sine has freq, ampl and phase
    non_harm = np.delete(np.arange(n_f_tot), harmonics)
    # we don't want the frequencies to go lower than about 1/T/100
    t_tot = np.ptp(times)
    f_low = 0.01 / t_tot
    # do the fit
    par_init = np.concatenate(([p_orb], np.atleast_1d(const), np.atleast_1d(slope), np.delete(f_n, harmonics),
                               np.delete(a_n, harmonics), np.delete(ph_n, harmonics), a_n[harmonics], ph_n[harmonics]))
    par_bounds = [(0, None)] + [(None, None) for _ in range(2 * n_sect)]
    par_bounds = par_bounds + [(f_low, None) for _ in range(n_sin)]
    par_bounds = par_bounds + [(0, None) for _ in range(n_sin)] + [(None, None) for _ in range(n_sin)]
    par_bounds = par_bounds + [(0, None) for _ in range(n_harm)] + [(None, None) for _ in range(n_harm)]
    arguments = (times, signal, harmonic_n, i_sectors)
    result = sp.optimize.minimize(objective_sinusoids_harmonics, jac=jacobian_sinusoids_harmonics,
                                  x0=par_init, args=arguments, method='L-BFGS-B', bounds=par_bounds,
                                  options={'maxiter': 10**4 * len(par_init)})
    # separate results
    res_p_orb = result.x[0]
    res_const = result.x[1:1 + n_sect]
    res_slope = result.x[1 + n_sect:1 + 2 * n_sect]
    res_freqs = np.zeros(n_f_tot)
    res_freqs[non_harm] = result.x[1 + 2 * n_sect:1 + 2 * n_sect + n_sin]
    res_freqs[harmonics] = harmonic_n / res_p_orb
    res_ampls = np.zeros(n_f_tot)
    res_ampls[non_harm] = result.x[1 + 2 * n_sect + n_sin:1 + 2 * n_sect + 2 * n_sin]
    res_ampls[harmonics] = result.x[1 + 2 * n_sect + 3 * n_sin:1 + 2 * n_sect + 3 * n_sin + n_harm]
    res_phases = np.zeros(n_f_tot)
    res_phases[non_harm] = result.x[1 + 2 * n_sect + 2 * n_sin:1 + 2 * n_sect + 3 * n_sin]
    res_phases[harmonics] = result.x[1 + 2 * n_sect + 3 * n_sin + n_harm:1 + 2 * n_sect + 3 * n_sin + 2 * n_harm]
    if verbose:
        model_linear = tsf.linear_curve(times, res_const, res_slope, i_sectors)
        model_sinusoid = tsf.sum_sines(times, res_freqs, res_ampls, res_phases)
        resid = signal - model_linear - model_sinusoid
        bic = tsf.calc_bic(resid, 1 + 2 * n_sect + 3 * n_sin + 2 * n_harm)
        print(f'Fit convergence: {result.success} - BIC: {bic:1.2f}. '
              f'N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')
    return res_p_orb, res_const, res_slope, res_freqs, res_ampls, res_phases


def fit_multi_sinusoid_harmonics_per_group(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors,
                                           verbose=False):
    """Perform the multi-sinusoid, non-linear least-squares fit with harmonic frequencies
    per frequency group

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
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    res_p_orb: float
        Updated Orbital period in days
    res_const: numpy.ndarray[float]
        Updated y-intercepts of a piece-wise linear curve
    res_slope: numpy.ndarray[float]
        Updated slopes of a piece-wise linear curve
    res_freqs: numpy.ndarray[float]
        Updated frequencies of a number of sine waves
    res_ampls: numpy.ndarray[float]
        Updated amplitudes of a number of sine waves
    res_phases: numpy.ndarray[float]
        Updated phases of a number of sine waves

    Notes
    -----
    In reducing the overall runtime of the NL-LS fit, this will improve
    the fits per group of 15-20 frequencies, leaving the other frequencies
    as fixed parameters.
    Contrary to multi_sine_NL_LS_fit_per_group, the groups don't have to be provided
    as they are made with the default parameters of ut.group_frequencies_for_fit.
    The orbital harmonics are always the first group.
    """
    # get harmonics and group the remaining frequencies
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    indices = np.arange(len(f_n))
    i_non_harm = np.delete(indices, harmonics)
    f_groups = ut.group_frequencies_for_fit(a_n[i_non_harm], g_min=20, g_max=25)
    f_groups = [i_non_harm[g] for g in f_groups]  # convert back to indices for full f_n list
    n_groups = len(f_groups)
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_harm = len(harmonics)
    n_sin = len(f_n) - n_harm
    # make a copy of the initial parameters
    res_const = np.copy(np.atleast_1d(const))
    res_slope = np.copy(np.atleast_1d(slope))
    res_freqs, res_ampls, res_phases = np.copy(f_n), np.copy(a_n), np.copy(ph_n)
    # fit the harmonics (first group)
    if verbose:
        print(f'Fit of harmonics', end='\r')
    # remove harmonic frequencies
    resid = signal - tsf.sum_sines(times, np.delete(res_freqs, harmonics), np.delete(res_ampls, harmonics),
                                   np.delete(res_phases, harmonics))
    par_init = np.concatenate(([p_orb], res_const, res_slope, a_n[harmonics], ph_n[harmonics]))
    par_bounds = [(0, None)] + [(None, None) for _ in range(2 * n_sect)]
    par_bounds = par_bounds + [(0, None) for _ in range(n_harm)] + [(None, None) for _ in range(n_harm)]
    arguments = (times, resid, harmonic_n, i_sectors)
    result = sp.optimize.minimize(objective_sinusoids_harmonics, jac=jacobian_sinusoids_harmonics,
                                  x0=par_init,  args=arguments, method='L-BFGS-B', bounds=par_bounds,
                                  options={'maxiter': 10**4 * len(par_init)})
    # separate results
    res_p_orb = result.x[0]
    res_const = result.x[1:1 + n_sect]
    res_slope = result.x[1 + n_sect:1 + 2 * n_sect]
    res_freqs[harmonics] = harmonic_n / res_p_orb
    res_ampls[harmonics] = result.x[1 + 2 * n_sect:1 + 2 * n_sect + n_harm]
    res_phases[harmonics] = result.x[1 + 2 * n_sect + n_harm:1 + 2 * n_sect + 2 * n_harm]
    if verbose:
        model_linear = tsf.linear_curve(times, res_const, res_slope, i_sectors)
        model_sinusoid = tsf.sum_sines(times, res_freqs, res_ampls, res_phases)
        resid = signal - model_linear - model_sinusoid
        bic = tsf.calc_bic(resid, 1 + 2 * n_sect + 3 * n_sin + 2 * n_harm)
        print(f'Fit of harmonics - BIC: {bic:1.2f}. N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')
    # update the parameters for each group
    for k, group in enumerate(f_groups):
        if verbose:
            print(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)}', end='\r')
        # subtract all other sines from the data, they are fixed now
        resid = signal - tsf.sum_sines(times, np.delete(res_freqs, group), np.delete(res_ampls, group),
                                       np.delete(res_phases, group))
        # fit only the frequencies in this group (constant and slope are also fitted still)
        output = fit_multi_sinusoid(times, resid, res_const, res_slope, res_freqs[group],
                                    res_ampls[group], res_phases[group], i_sectors, verbose=False)
        res_const, res_slope, out_freqs, out_ampls, out_phases = output
        res_freqs[group] = out_freqs
        res_ampls[group] = out_ampls
        res_phases[group] = out_phases
        if verbose:
            model_linear = tsf.linear_curve(times, res_const, res_slope, i_sectors)
            model_sinusoid = tsf.sum_sines(times, res_freqs, res_ampls, res_phases)
            resid_new = signal - (model_linear + model_sinusoid)
            bic = tsf.calc_bic(resid_new, 1 + 2 * n_sect + 3 * n_sin + 2 * n_harm)
            print(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)} - BIC: {bic:1.2f}')
    return res_p_orb, res_const, res_slope, res_freqs, res_ampls, res_phases


@nb.njit(cache=True)
def objective_third_light(params, times, signal, p_orb, a_h, ph_h, harmonic_n, i_sectors):
    """The objective function to give to scipy.optimize.minimize for a sum of sine waves
    plus a set of harmonic frequencies and the effect of third light.

    Parameters
    ----------
    params: numpy.ndarray[float]
        Third light for a set of sectors and a stretch parameter.
        Has to be ordered in the following way:
        [third_light1, third_light2, ..., stretch]
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    a_h: numpy.ndarray[float]
        Corresponding amplitudes of the orbital harmonics
    ph_h: numpy.ndarray[float]
        Corresponding phases of the orbital harmonics
    harmonic_n: numpy.ndarray[int]
        Integer indicating which harmonic each index in 'harmonics'
        points to. n=1 for the base frequency (=orbital frequency)
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    
    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the residuals
    
    See Also
    --------
    linear_curve and sum_sines for the definition of the parameters.
    """
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    # separate the parameters
    light_3 = params[:n_sect]
    stretch = params[-1]
    freqs = harmonic_n / p_orb
    # finally, make the model and calculate the likelihood
    model = tsf.sum_sines(times, freqs, a_h * stretch, ph_h)
    # stretching the harmonic model is equivalent to multiplying the amplitudes (to get the right eclipse depth)
    const, slope = tsf.linear_pars(times, signal - model, i_sectors)
    model = model + tsf.linear_curve(times, const, slope, i_sectors)
    model = ut.model_crowdsap(model, 1 - light_3, i_sectors)  # incorporate third light
    # need minus the likelihood for minimisation
    resid = signal - model
    ln_likelihood = tsf.calc_likelihood(resid, times=times, signal_err=None, iid=True)
    return -ln_likelihood


def fit_minimum_third_light(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Fits for the minimum amount of third light needed in each sector.
    
    Since the contamination by third light can vary across (TESS) sectors,
    fully data driven fitting for third light will result in a lower bound for
    the flux fraction of contaminating light in the aperture per sector.
    (corresponding to an upper bound to CROWDSAP parameter, or 1-third_light)
    In the null hypothesis that our target is the one eclipsing, the deepest
    eclipses present in the data are in the sector with the least third light.
    If we have the CROWDSAP parameter, this can be compared to check the null
    hypothesis, or we can simply use the CROWDSAP as starting point.
    
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
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    res_light_3: float
        Minimum third light needed in each sector
    res_stretch: float
        Scaling factor for the harmonic amplitudes
    const: numpy.ndarray[float]
        Updated y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        Updated slopes of a piece-wise linear curve
    
    Notes
    -----
    For this to work we need at least an initial harmonic model of the eclipses
    (and at least a handful of sectors).
    The given third light values are fractional values of a median-normalised light curve.
    """
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_sin = len(non_harm)  # each independent sine has freq, ampl and phase
    n_harm = len(harmonics)
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    bic_init = tsf.calc_bic(resid, 2 * n_sect + 1 + 2 * n_harm + 3 * n_sin)
    # start off at third light of 0.01 and stretch parameter of 1.01
    par_init = np.concatenate((np.zeros(n_sect) + 0.01, [1.01]))
    par_bounds = [(0, 1) if (i < n_sect) else (1, None) for i in range(n_sect + 1)]
    arguments = (times, signal, p_orb, a_n[harmonics], ph_n[harmonics], harmonic_n, i_sectors)
    # do the fit
    result = sp.optimize.minimize(objective_third_light, x0=par_init, args=arguments, method='L-BFGS-B',
                                  bounds=par_bounds, options={'maxfev': 10**4 * len(par_init)})
    # separate results
    res_light_3 = result.x[:n_sect]
    res_stretch = result.x[-1]
    model_sinusoid_h = tsf.sum_sines(times, f_n[harmonics], a_n[harmonics] * res_stretch, ph_n[harmonics])
    model_sinusoid_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    resid = signal - model_sinusoid_nh - model_sinusoid_h
    const, slope = tsf.linear_pars(times, resid, i_sectors)
    if verbose:
        model_linear = tsf.linear_curve(times, const, slope, i_sectors)
        model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
        model_crowdsap = ut.model_crowdsap(model_linear + model_sinusoid, 1 - res_light_3, i_sectors)
        bic = tsf.calc_bic((signal - model_crowdsap), 2 * n_sect + 1 + 2 * n_harm + 3 * n_sin)
        print(f'Fit convergence: {result.success}. Old BIC: {bic_init:1.2f}, New BIC: {bic:1.2f}. '
              f'N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')
    return res_light_3, res_stretch, const, slope


@nb.njit(cache=True)
def eclipse_empirical_lc(times, p_orb, mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2):
    """Empirical model of two simple connected eclipses

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    mid_1: float
        Time of mid-eclipse 1
    mid_2: float
        Time of mid-eclipse 2
    dur_1: float
        Duration of eclipse 1
    dur_2: float
        Duration of eclipse 2
    dur_b_1: float
        Duration of a flat bottom of eclipse 1
    dur_b_2: float
        Duration of a flat bottom of eclipse 2
    d_1: float
        Depth of eclipse 1
    d_2: float
        Depth of eclipse 2
    h_1: float
        Height of the top of eclipse 1
    h_2: float
        Height of the top of eclipse 2

    Returns
    -------
    model_ecl: numpy.ndarray[float]
        symmetric eclipse model using cubic functions

    Notes
    -----
    The timings are deduced from the cubic curves in case the discriminant
    is positive and the slope at the inflection point has the right sign.

    Eclipses are connected with a cubic polynomial to allow varying their height.
    """
    # translate the timings by the time of primary minimum with respect to t_mean
    t_zero = mid_1
    mid_1, mid_2 = mid_1 - t_zero, mid_2 - t_zero
    # compute the times of eclipse contact and tangency
    t_c1_1, t_c2_1 = mid_1 - dur_1 / 2, mid_1 + dur_1 / 2
    t_c1_2, t_c2_2 = mid_1 - dur_b_1 / 2, mid_1 + dur_b_1 / 2
    t_c3_1, t_c4_1 = mid_2 - dur_2 / 2, mid_2 + dur_2 / 2
    t_c3_2, t_c4_2 = mid_2 - dur_b_2 / 2, mid_2 + dur_b_2 / 2
    # fold the time series
    t_folded, _, _ = tsf.fold_time_series(times, p_orb, t_zero, t_ext_1=0, t_ext_2=0)
    t_folded_adj = np.copy(t_folded)
    t_folded_adj[t_folded > p_orb + t_c1_1] -= p_orb  # stick eclipse 1 back together
    # make masks for the right time intervals
    mask_1 = (t_folded_adj >= t_c1_1) & (t_folded_adj < t_c1_2)
    mask_2 = (t_folded_adj > t_c2_2) & (t_folded_adj <= t_c2_1)
    mask_3 = (t_folded >= t_c3_1) & (t_folded < t_c3_2)
    mask_4 = (t_folded > t_c4_2) & (t_folded <= t_c4_1)
    # compute the cubic curves if there are points left
    if (np.any(mask_1) & np.any(mask_2)):
        # parameters for the cubic function
        c1_a, c1_b, c1_c, c1_d = tsf.cubic_pars_two_points(t_c1_1, h_1, t_c1_2, h_1 - d_1)
        mean_t_c1 = (t_c1_1 + t_c1_2) / 2
        # cubic function of the primary ingress
        mean_t_1 = np.mean(t_folded_adj[mask_1])
        cubic_1 = tsf.cubic_curve(t_folded_adj[mask_1], c1_a, c1_b, c1_c, c1_d, t_zero=mean_t_c1 - mean_t_1)
        # cubic function of the primary egress
        mean_t_2 = np.mean(2 * mid_1 - t_folded_adj[mask_2])
        cubic_2 = tsf.cubic_curve(2 * mid_1 - t_folded_adj[mask_2], c1_a, c1_b, c1_c, c1_d, t_zero=mean_t_c1 - mean_t_2)
    else:
        cubic_1 = np.ones(np.sum(mask_1)) * h_1
        cubic_2 = np.ones(np.sum(mask_2)) * h_1
    if (np.any(mask_3) & np.any(mask_4)):
        # parameters for the cubic function
        c3_a, c3_b, c3_c, c3_d = tsf.cubic_pars_two_points(t_c3_1, h_2, t_c3_2, h_2 - d_2)
        mean_t_c3 = (t_c3_1 + t_c3_2) / 2
        # cubic function of the secondary ingress
        mean_t_3 = np.mean(t_folded[mask_3])
        cubic_3 = tsf.cubic_curve(t_folded[mask_3], c3_a, c3_b, c3_c, c3_d, t_zero=mean_t_c3 - mean_t_3)
        # cubic function of the secondary egress
        mean_t_4 = np.mean(2 * mid_2 - t_folded[mask_4])
        cubic_4 = tsf.cubic_curve(2 * mid_2 - t_folded[mask_4], c3_a, c3_b, c3_c, c3_d, t_zero=mean_t_c3 - mean_t_4)
    else:
        cubic_3 = np.ones(np.sum(mask_3)) * h_2
        cubic_4 = np.ones(np.sum(mask_4)) * h_2
    # make lines for the bottom of the eclipse
    mask_b_1 = (t_folded_adj >= t_c1_2) & (t_folded_adj <= t_c2_2)
    mask_b_2 = (t_folded >= t_c3_2) & (t_folded <= t_c4_2)
    if not (np.any(mask_1) & np.any(mask_2)):
        mask_b_1 = np.zeros(len(t_folded), dtype=np.bool_)  # if no eclipse, also no bottom
    if not (np.any(mask_3) & np.any(mask_4)):
        mask_b_2 = np.zeros(len(t_folded), dtype=np.bool_)  # if no eclipse, also no bottom
    line_b_1 = np.ones(len(t_folded_adj[mask_b_1])) * (h_1 - d_1)
    line_b_2 = np.ones(len(t_folded_adj[mask_b_2])) * (h_2 - d_2)
    # make connecting lines
    mask_12 = (t_folded > t_c2_1) & (t_folded < t_c3_1)  # from 1 to 2
    mask_21 = (t_folded > t_c4_1) & (t_folded < t_c1_1 + p_orb)  # from 2 to 1
    if np.any(mask_12) & (h_1 != h_2):
        # parameters for the cubic function
        c12_a, c12_b, c12_c, c12_d = tsf.cubic_pars_two_points(t_c2_1, h_1, t_c3_1, h_2)
        mean_t_c12 = (t_c2_1 + t_c3_1) / 2
        # cubic function of the connection from 1 to 2
        mean_t_12 = np.mean(t_folded_adj[mask_12])
        cubic_12 = tsf.cubic_curve(t_folded_adj[mask_12], c12_a, c12_b, c12_c, c12_d, t_zero=mean_t_c12 - mean_t_12)
    else:
        cubic_12 = np.ones(np.sum(mask_12)) * h_1
    if np.any(mask_21) & (h_1 != h_2):
        # parameters for the cubic function
        c21_a, c21_b, c21_c, c21_d = tsf.cubic_pars_two_points(t_c4_1, h_2, t_c1_1 + p_orb, h_1)
        mean_t_c21 = (t_c4_1 + t_c1_1 + p_orb) / 2
        # cubic function of the connection from 2 to 1
        mean_t_21 = np.mean(t_folded[mask_21])
        cubic_21 = tsf.cubic_curve(t_folded[mask_21], c21_a, c21_b, c21_c, c21_d, t_zero=mean_t_c21 - mean_t_21)
    else:
        cubic_21 = np.ones(np.sum(mask_21)) * h_2
    # stick together the eclipse model (for t_folded_adj)
    model_ecl = np.ones(len(t_folded))
    model_ecl[mask_12] = cubic_12
    model_ecl[mask_21] = cubic_21
    model_ecl[mask_b_1] = line_b_1
    model_ecl[mask_b_2] = line_b_2
    model_ecl[mask_1] = cubic_1
    model_ecl[mask_2] = cubic_2
    model_ecl[mask_3] = cubic_3
    model_ecl[mask_4] = cubic_4
    return model_ecl


@nb.njit(cache=True)
def objective_empirical_lc(params, times, signal, p_orb):
    """Objective function for a set of eclipse timings

    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of a light curve model made of cubics.
        Has to be ordered in the following way:
        [mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2]
    times: numpy.ndarray[float]
        Timestamps of the time series, zero point at primary minimum
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days

    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the residuals

    See Also
    --------
    eclipse_empirical_lc
    """
    # midpoints, durations, depths, height adjustments
    mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2 = params
    # the model
    model = eclipse_empirical_lc(times, p_orb, mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2)
    # determine likelihood for the model (minus this for minimisation)
    ln_likelihood = tsf.calc_likelihood(signal - model, times=times, signal_err=None, iid=True)
    # make sure we don't allow impossible stuff
    dur_zero = (dur_1 < 0) | (dur_2 < 0) | (dur_b_1 < 0) | (dur_b_2 < 0)
    if dur_zero | (dur_b_1 > dur_1) | (dur_b_2 > dur_2) | (d_1 < 0) | (d_2 < 0):
        return -ln_likelihood + 10**9
    return -ln_likelihood


def fit_eclipse_empirical(times, signal, p_orb, timings, timings_err, i_sectors, verbose=False):
    """Perform least-squares fit for improved eclipse timings using an empirical model.

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    timings: numpy.ndarray[float]
        Eclipse timings: minima, first/last contact points, internal tangency and depths,
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2
    timings_err: numpy.ndarray[float]
        Error estimates for the eclipse timings and depths,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err,
        t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err, depth_1_err, depth_2_err
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    res_emp: numpy.ndarray[float]
        Updated empirical eclipse model parameters,
        mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2
    res_timings: numpy.ndarray[float]
        Eclipse timings: minima, first/last contact points, internal tangency and depths,
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2
    
    See Also
    --------
    eclipse_cubic_model, objective_cubic_lc

    Notes
    -----
    Times of primary and secondary eclispe are measured with respect to the mean time.
    Eclipses are symmetrical in this model.
    
    Strictly speaking it is doing a maximum log-likelihood fit, but that is
    in essence identical (and numerically more stable due to the logarithm).
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, d_1, d_2 = timings
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err = timings_err[:6]
    t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err, d_1_err, d_2_err = timings_err[6:]
    # make sure we remove trends
    const, slope = tsf.linear_pars(times, signal - np.mean(signal), i_sectors)
    ecl_signal = signal - tsf.linear_curve(times, const, slope, i_sectors)
    # initial parameters and bounds
    mid_1 = (t_1_1 + t_1_2) / 2
    mid_2 = (t_2_1 + t_2_2) / 2
    dur_1 = t_1_2 - t_1_1
    dur_2 = t_2_2 - t_2_1
    h_minmax = (np.min(ecl_signal), np.max(ecl_signal))
    dur_b_1 = t_b_1_2 - t_b_1_1
    dur_b_2 = t_b_2_2 - t_b_2_1
    par_init = np.array([mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, 1, 1])
    par_bounds = ((t_1_1 + dur_1 / 4, t_1_2 - dur_1 / 4), (t_2_1 + dur_2 / 4, t_2_2 - dur_2 / 4),
                  (0, 3 * (t_1_1_err + t_1_2_err)), (0, 3 * (t_2_1_err + t_2_2_err)),
                  (0, 3 * (t_b_1_1_err + t_b_1_2_err)), (0, 3 * (t_b_2_1_err + t_b_2_2_err)),
                  (d_1 - 3 * d_1_err, 2 * d_1), (d_2 - 3 * d_2_err, 2 * d_2), h_minmax, h_minmax)
    arguments = (times, ecl_signal, p_orb)
    result = sp.optimize.minimize(objective_empirical_lc, x0=par_init, args=arguments, method='L-BFGS-B',
                                  bounds=par_bounds, options={'maxiter': 10**4 * len(par_init)})
    mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2 = result.x
    # compute the times of eclipse contact and tangency
    t_1_1, t_1_2 = mid_1 - dur_1 / 2, mid_1 + dur_1 / 2
    t_b_1_1, t_b_1_2 = mid_1 - dur_b_1 / 2, mid_1 + dur_b_1 / 2
    t_2_1, t_2_2 = mid_2 - dur_2 / 2, mid_2 + dur_2 / 2
    t_b_2_1, t_b_2_2 = mid_2 - dur_b_2 / 2, mid_2 + dur_b_2 / 2
    # collect into arrays
    res_cubics = np.array([mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2])
    res_timings = np.array([mid_1, mid_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, d_1, d_2])
    if verbose:
        model_ecl = eclipse_empirical_lc(times, p_orb, *res_cubics)
        resid = ecl_signal - model_ecl
        bic = tsf.calc_bic(resid, 1 + len(result.x))
        print(f'Fit convergence: {result.success}. BIC: {bic:1.2f}. '
              f'N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}. ')
    return res_cubics, res_timings


@nb.njit(cache=True)
def objective_empirical_sinusoids_lc(params, times, signal, p_orb, i_sectors):
    """Objective function for a set of eclipse timings and harmonics

    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of a light curve model made of cubics,
        plus a set of harmonic sinusoids.
        Has to be ordered in the following way:
        [mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2,
         const, a_h_1, a_h_2, ..., ph_h_1, ph_h_2, ...]
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).

    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the residuals

    See Also
    --------
    eclipse_lc_simple
    """
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_sin = (len(params) - 10 - 2 * n_sect) // 3  # each sine has freq, ampl and phase
    # separate the parameters
    mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2 = params[0:10]
    const = params[10:10 + n_sect]
    slope = params[10 + n_sect:10 + 2 * n_sect]
    freqs = params[10 + 2 * n_sect:10 + 2 * n_sect + n_sin]
    ampls = params[10 + 2 * n_sect + n_sin:10 + 2 * n_sect + 2 * n_sin]
    phases = params[10 + 2 * n_sect + 2 * n_sin:10 + 2 * n_sect + 3 * n_sin]
    # make sure we don't allow impossible stuff
    dur_zero = (dur_1 < 0) | (dur_2 < 0) | (dur_b_1 < 0) | (dur_b_2 < 0)
    if dur_zero | (dur_b_1 > dur_1) | (dur_b_2 > dur_2) | (d_1 < 0) | (d_2 < 0):
        return 10**9
    # the model
    model_ecl = eclipse_empirical_lc(times, p_orb, mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2)
    # make the sinusoid model
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, freqs, ampls, phases)
    # calculate the likelihood (minus this for minimisation)
    model = model_linear + model_sinusoid + model_ecl
    resid = signal - model
    ln_likelihood = tsf.calc_likelihood(resid, times=times, signal_err=None, iid=True)
    return -ln_likelihood


def fit_eclipse_empirical_sinusoids(times, signal, p_orb, timings, cubic_pars, const, slope, f_n, a_n, ph_n, i_sectors,
                                    verbose=False):
    """Perform the multi-sinusoid, non-linear least-squares fit per frequency group
    and including the cubics eclipse model

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    timings: numpy.ndarray[float]
        Eclipse timings: minima, first/last contact points, internal tangency and depths,
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2
    cubic_pars: numpy.ndarray[float]
        Empirical eclipse model parameters,
        mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2
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
    res_cubics: numpy.ndarray[float]
        Updated empirical eclipse model parameters,
        mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2
    res_timings: numpy.ndarray[float]
        Eclipse timings: minima, first/last contact points, internal tangency and depths,
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2
    res_const: numpy.ndarray[float]
        Updated y-intercepts of a piece-wise linear curve
    res_slope: numpy.ndarray[float]
        Updated slopes of a piece-wise linear curve
    res_freqs: numpy.ndarray[float]
        Updated frequencies of a number of sine waves
    res_ampls: numpy.ndarray[float]
        Updated amplitudes of a number of sine waves
    res_phases: numpy.ndarray[float]
        Updated phases of a number of sine waves
    
    See Also
    --------
    eclipse_cubic_model, objective_cubic_lc
    
    Notes
    -----
    Times of primary and secondary eclispe are measured with respect to the mean time.
    Eclipses are symmetrical in this model.
    
    In reducing the overall runtime of the NL-LS fit, this will improve the
    fits on just the given groups of (closely spaced) frequencies, leaving the other
    frequencies as fixed parameters.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, d_1, d_2 = timings
    f_groups = ut.group_frequencies_for_fit(a_n, g_min=20, g_max=25)
    n_groups = len(f_groups)
    n_sect = len(i_sectors)
    n_sin = len(f_n)
    # make a copy of the initial parameters
    res_cubics = np.copy(cubic_pars)
    res_const = np.copy(np.atleast_1d(const))
    res_slope = np.copy(np.atleast_1d(slope))
    res_freqs = np.copy(f_n)
    res_ampls = np.copy(a_n)
    res_phases = np.copy(ph_n)
    # we don't want the frequencies to go lower than about 1/T/100
    t_tot = np.ptp(times)
    f_low = 0.01 / t_tot
    h_minmax = (np.min(signal), np.max(signal))
    # update the parameters for each group
    for k, group in enumerate(f_groups):
        if verbose:
            print(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)}', end='\r')
        # subtract all other sines from the data, they are fixed now
        resid = signal - tsf.sum_sines(times, np.delete(res_freqs, group), np.delete(res_ampls, group),
                                       np.delete(res_phases, group))
        # fit only the frequencies in this group (and eclipse model, constant and slope)
        par_init = np.concatenate((res_cubics, res_const, res_slope, res_freqs[group], res_ampls[group],
                                   res_phases[group]))
        par_bounds = [(t_1_1, t_1_2), (t_2_1, t_2_2), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None),
                      h_minmax, h_minmax]
        n_gr = len(res_freqs[group])
        par_bounds = par_bounds + [(None, None) for _ in range(2 * n_sect)]
        par_bounds = par_bounds + [(f_low, None) for _ in range(n_gr)]
        par_bounds = par_bounds + [(0, None) for _ in range(n_gr)] + [(None, None) for _ in range(n_gr)]
        arguments = (times, resid, p_orb, i_sectors)
        result = sp.optimize.minimize(objective_empirical_sinusoids_lc, x0=par_init, args=arguments, method='L-BFGS-B',
                                      bounds=par_bounds, options={'maxiter': 10**4 * len(par_init)})
        # separate results
        n_sin_g = len(res_freqs[group])
        res_cubics = result.x[0:10]
        res_const = result.x[10:10 + n_sect]
        res_slope = result.x[10 + n_sect:10 + 2 * n_sect]
        out_freqs = result.x[10 + 2 * n_sect:10 + 2 * n_sect + n_sin_g]
        out_ampls = result.x[10 + 2 * n_sect + n_sin_g:10 + 2 * n_sect + 2 * n_sin_g]
        out_phases = result.x[10 + 2 * n_sect + 2 * n_sin_g:10 + 2 * n_sect + 3 * n_sin_g]
        res_freqs[group] = out_freqs
        res_ampls[group] = out_ampls
        res_phases[group] = out_phases
        if verbose:
            model_linear = tsf.linear_curve(times, res_const, res_slope, i_sectors)
            model_sinusoid = tsf.sum_sines(times, res_freqs, res_ampls, res_phases)
            model_ecl = eclipse_empirical_lc(times, p_orb, *res_cubics)
            resid_new = signal - (model_linear + model_sinusoid + model_ecl)
            bic = tsf.calc_bic(resid_new, 2 * n_sect + 3 * n_sin + 11)
            print(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)} - BIC: {bic:1.2f}. '
                  f'N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')
    mid_1, mid_2, dur_1, dur_2, dur_b_1, dur_b_2, d_1, d_2, h_1, h_2 = res_cubics
    # compute the times of eclipse contact and tangency
    t_1_1, t_1_2 = mid_1 - dur_1 / 2, mid_1 + dur_1 / 2
    t_b_1_1, t_b_1_2 = mid_1 - dur_b_1 / 2, mid_1 + dur_b_1 / 2
    t_2_1, t_2_2 = mid_2 - dur_2 / 2, mid_2 + dur_2 / 2
    t_b_2_1, t_b_2_2 = mid_2 - dur_b_2 / 2, mid_2 + dur_b_2 / 2
    res_timings = np.array([mid_1, mid_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, d_1, d_2])
    return res_cubics, res_timings, res_const, res_slope, res_freqs, res_ampls, res_phases


@nb.njit(cache=True)
def eclipse_physical_lc(times, p_orb, t_zero, e, w, i, r_sum, r_rat, sb_rat):
    """Simple eclipse light curve model

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of the deepest minimum with respect to the mean time
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit (radians)
    r_sum: float
        Sum of radii in units of the semi-major axis
    r_rat: float
        Radius ratio r_2/r_1
    sb_rat: float
        Surface brightness ratio sb_2/sb_1

    Returns
    -------
    model: numpy.ndarray[float]
        Eclipse light curve model for the given time points
    """
    # guard against unphysical parameters
    if (e > 1):
        interp_model = np.ones(len(times))
        return interp_model
    # theta_1 is primary minimum, theta_2 is secondary minimum, the others are at the furthest projected distance
    theta_1, theta_2, theta_3, theta_4 = af.minima_phase_angles_2(e, w, i)
    # calculate time shift between primary minimum and superior conjunction
    t_shift = p_orb / (2 * np.pi) * af.integral_kepler_2(af.true_anomaly(theta_1, w)-2*np.pi, af.true_anomaly(0, w), e)
    # make the simple model
    thetas = np.arange(0, 2 * np.pi, 0.001)  # position angle along the orbit
    ecl_model = 1 - af.eclipse_depth(thetas, e, w, i, r_sum, r_rat, sb_rat, theta_3, theta_4)
    # determine the model times
    nu_1 = af.true_anomaly(0, w)
    nu_2 = af.true_anomaly(thetas, w)  # integral endpoints
    t_model = p_orb / (2 * np.pi) * af.integral_kepler_2(nu_1, nu_2, e)
    # interpolate the model (probably faster than trying to calculate the times)
    t_folded, _, _ = tsf.fold_time_series(times, p_orb, t_zero + t_shift, t_ext_1=0, t_ext_2=0)
    interp_model = np.interp(t_folded, t_model, ecl_model)
    return interp_model


@nb.njit(cache=True)
def objective_physical_lc(params, times, signal, p_orb, t_zero):
    """Objective function for a set of eclipse parameters

    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of a simple eclipse light curve model.
        Has to be ordered in the following way:
        [ecosw, esinw, cosi, phi_0, log_rr, log_sb, offset]
    times: numpy.ndarray[float]
        Timestamps of the time series, zero point at primary minimum
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of the deepest minimum with respect to the mean time

    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the residuals

    See Also
    --------
    eclipse_lc_simple
    """
    ecosw, esinw, cosi, phi_0, log_rr, log_sb, offset = params
    e, w, i, r_sum, r_rat, sb_rat = ut.convert_to_phys_space(ecosw, esinw, cosi, phi_0, log_rr, log_sb)
    # check for unphysical e
    model = eclipse_physical_lc(times, p_orb, t_zero, e, w, i, r_sum, r_rat, sb_rat) + offset
    # determine likelihood for the model (minus this for minimisation)
    ln_likelihood = tsf.calc_likelihood(signal - model, times=times, signal_err=None, iid=True)
    # check periastron distance
    d_peri = 1 - e
    if (r_sum < d_peri):
        return -ln_likelihood + 10**9
    return -ln_likelihood


def fit_eclipse_physical(times, signal, p_orb, t_zero, par_init, par_err, i_sectors, verbose=False):
    """Perform least-squares fit for the orbital parameters that can be obtained
    from the eclipses in the light curve.

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of the deepest minimum with respect to the mean time
    par_init: tuple[float], list[float], numpy.ndarray[float]
        Initial eclipse parameters to start the fit, consisting of:
        e, w, i, r_sum, r_rat, sb_rat
    par_err: numpy.ndarray[float]
        Errors in the initial eclipse parameters:
        e_err, w_err, i_err, r_sum_err, r_rat_err, sb_rat_err,
        ecosw_err, esinw_err, cosi_err, phi_0_err, log_rr_err, log_sb_err
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    par_out: numpy.ndarray[float]
        Fit results from the scipy optimizer,
        e, w, i, r_sum, r_rat, sb_rat, offset

    See Also
    --------
    eclipse_physical_lc, objective_physcal_lc

    Notes
    -----
    Strictly speaking it is doing a maximum log-likelihood fit, but that is
    in essence identical (and numerically more stable due to the logarithm).
    
    Fit is performed in the parameter space:
    ecosw, esinw, cosi, phi_0, log_rr, log_sb
    """
    e, w, i, r_sum, r_rat, sb_rat = par_init
    ecosw_err, esinw_err, cosi_err, phi_0_err, log_rr_err, log_sb_err = par_err[6:]
    # make sure we remove trends
    const, slope = tsf.linear_pars(times, signal - np.mean(signal), i_sectors)
    ecl_signal = signal - tsf.linear_curve(times, const, slope, i_sectors)
    # remove initial model to obtain initial offset level
    model_init = eclipse_physical_lc(times, p_orb, t_zero, *par_init)
    offset_init = np.mean(ecl_signal - model_init)
    # initial parameters and bounds
    ecosw, esinw, cosi, phi_0, log_rr, log_sb = ut.convert_from_phys_space(e, w, i, r_sum, r_rat, sb_rat)
    par_init = (ecosw, esinw, cosi, phi_0, log_rr, log_sb, offset_init)
    # limit bounds for fit if errors big (ubound for cosi is restricted - its measurement is sort of an upper limit)
    par_bounds = ((min(max(ecosw - min(3 * ecosw_err, 0.2), -1), 0.9),
                   min(max(ecosw + min(3 * ecosw_err, 0.2), -0.9), 1)),
                  (min(max(esinw - min(3 * esinw_err, 0.2), -1), 0.9),
                   min(max(esinw + min(3 * esinw_err, 0.2), -0.9), 1)),
                  (min(max(cosi - min(3 * cosi_err, 0.2), 0), 0.7),
                   min(max(cosi + min(3 * cosi_err, 0.001), 0.1), 0.9)),
                  (min(max(phi_0 - min(3 * phi_0_err, 0.1), 0), 0.7),
                   min(max(phi_0 + min(3 * phi_0_err, 0.1), 0.3), 1.57)),
                  (min(max(log_rr - min(3 * log_rr_err, 1), -3), 1),
                   min(max(log_rr + min(3 * log_rr_err, 1), -1), 3)),
                  (min(max(log_sb - min(3 * log_sb_err, 1), -3), 1),
                   min(max(log_sb + min(3 * log_sb_err, 1), -1), 3)), (-1, 1))
    arguments = (times, ecl_signal, p_orb, t_zero)
    # do a local fit and then a global fit within bounds to compare
    result_a = sp.optimize.minimize(objective_physical_lc, x0=par_init, args=arguments, method='Nelder-Mead',
                                    bounds=par_bounds, options={'maxiter': 10**4 * len(par_init)})
    result_b = sp.optimize.shgo(objective_physical_lc, args=arguments, bounds=par_bounds,
                                minimizer_kwargs={'method': 'SLSQP'}, options={'minimize_every_iter': True})
    # compare objective function values
    model_ecl = eclipse_physical_lc(times, p_orb, t_zero, *ut.convert_to_phys_space(*result_a.x[:6]))
    bic_a = tsf.calc_bic((ecl_signal - (model_ecl + result_a.x[6])), 2 + len(result_a.x))
    model_ecl = eclipse_physical_lc(times, p_orb, t_zero, *ut.convert_to_phys_space(*result_b.x[:6]))
    bic_b = tsf.calc_bic((ecl_signal - (model_ecl + result_b.x[6])), 2 + len(result_b.x))
    if (bic_a < bic_b - 2):
        opt = 'local'
        result = result_a
    elif (abs(result_b.x[0]) == 1) | (abs(result_b.x[1]) == 1) | (abs(result_b.x[6]) == 1):
        # global opt could fail drastically when bumping against outer bounds
        opt = 'local'
        result = result_a
    else:
        opt = 'global'
        result = result_b
    # convert back parameters
    ecosw, esinw, cosi, phi_0, log_rr, log_sb, offset = result.x
    e, w, i, r_sum, r_rat, sb_rat = ut.convert_to_phys_space(ecosw, esinw, cosi, phi_0, log_rr, log_sb)
    par_out = np.array([e, w, i, r_sum, r_rat, sb_rat, offset])
    if verbose:
        model_ecl = eclipse_physical_lc(times, p_orb, t_zero, e, w, i, r_sum, r_rat, sb_rat)
        resid = ecl_signal - (model_ecl + offset)
        bic = tsf.calc_bic(resid, 2 + len(par_out))
        print(f'Fit convergence: {result.success}. Used {opt} optimiser result - BIC: {bic:1.2f}. '
              f'N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')
    return par_out


def wrap_ellc_lc(times, p_orb, t_zero, f_c, f_s, i, r_sum, r_rat, sb_rat):
    """Wrapper for a simple ELLC model with some fixed inputs
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of the deepest minimum with respect to the mean time
    f_c: float
        Combination of e and w: sqrt(e)cos(w)
    f_s: float
        Combination of e and w: sqrt(e)sin(w)
    i: float
        Inclination of the orbit (radians)
    r_sum: float
        Sum of radii in units of the semi-major axis
    r_rat: float
        Radius ratio r_2/r_1
    sb_rat: float
        Surface brightness ratio sb_2/sb_1
    
    Returns
    -------
    model: numpy.ndarray[float]
        Eclipse light curve model for the given time points
    
    Notes
    -----
    See P. F. L. Maxted 2016:
    https://ui.adsabs.harvard.edu/abs/2016A%26A...591A.111M/abstract
    """
    incl = i / np.pi * 180  # ellc likes degrees
    r_1 = r_sum / (1 + r_rat)
    r_2 = r_sum * r_rat / (1 + r_rat)
    # mean center the time array
    mean_t = np.mean(times)
    times_ms = times - mean_t
    # try to prevent fatal crashes from RLOF cases (or from zero radius)
    if (r_sum > 0):
        d_roche_1 = 2.44 * r_2 * (r_1 / r_2)  # * (q)**(1 / 3)  # 2.44*R_M*(rho_M/rho_m)**(1/3)
        d_roche_2 = 2.44 * r_1 * (r_2 / r_1)  # * (1 / q)**(1 / 3)
    else:
        d_roche_1 = 1
        d_roche_2 = 1
    d_peri = (1 - f_c**2 - f_s**2)  # a*(1 - e), but a=1
    if (max(d_roche_1, d_roche_2) > 0.98 * d_peri):
        model = np.ones(len(times_ms))  # Roche radius close to periastron distance
    else:
        model = ellc.lc(times_ms, r_1, r_2, f_c=f_c, f_s=f_s, incl=incl, sbratio=sb_rat, period=p_orb, t_zero=t_zero,
                        light_3=0, q=1, shape_1='roche', shape_2='roche',
                        ld_1='lin', ld_2='lin', ldc_1=0.5, ldc_2=0.5, gdc_1=0., gdc_2=0., heat_1=0., heat_2=0.)
    return model


def objective_ellc_lc(params, times, signal, p_orb):
    """Objective function for a set of eclipse parameters
    
    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of a simple eclipse light curve model.
        Has to be ordered in the following way:
        [f_c, f_s, i, r_sum, r_rat, sb_rat]
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    
    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the residuals
    
    See Also
    --------
    ellc_lc_simple
    """
    f_c, f_s, i, r_sum, r_rat, sb_rat = params
    try:
        model = wrap_ellc_lc(times, p_orb, 0, f_c, f_s, i, r_sum, r_rat, sb_rat)
    except:  # try to catch every error (I know this is bad (won't work though))
        # (try to) catch ellc errors
        return 10**9
    # determine likelihood for the model (minus this for minimisation)
    ln_likelihood = tsf.calc_likelihood(signal - model, times=times, signal_err=None, iid=True)
    return -ln_likelihood


def fit_ellc_lc(times, signal, p_orb, t_zero, const, slope, par_init, i_sectors, verbose=False):
    """Perform least-squares fit for the orbital parameters that can be obtained
    from the eclipses in the light curve.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of the deepest minimum with respect to the mean time
    const: numpy.ndarray[float]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[float]
        The slopes of a piece-wise linear curve
    par_init: tuple[float], list[float], numpy.ndarray[float]
        Initial eclipse parameters to start the fit, consisting of:
        f_c, f_s, i, r_sum, r_rat, sb_rat
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    par_out: numpy.ndarray[float]
        Fit results from the scipy optimizer, like par_init
    
    See Also
    --------
    ellc_lc_simple, objective_ellc_lc
    
    Notes
    -----
    Strictly speaking it is doing a maximum log-likelihood fit, but that is
    in essence identical (and numerically more stable due to the logarithm).
    """
    f_c, f_s, i, r_sum, r_rat, sb_rat = par_init
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_linear + 1
    # initial parameters and bounds
    par_init = (f_c, f_s, i, r_sum, r_rat, sb_rat)  # no offset
    par_bounds = ((-1, 1), (-1, 1), (0, np.pi / 2), (0, 1), (0.001, 1000), (0.001, 1000))
    arguments = (times, ecl_signal, p_orb)
    result = sp.optimize.minimize(objective_ellc_lc, x0=par_init, args=arguments, method='Nelder-Mead',
                                  bounds=par_bounds, options={'maxiter': 10**4 * len(par_init)})
    par_out = result.x
    if verbose:
        opt_f_c, opt_f_s, opt_i, opt_r_sum, opt_r_rat, opt_sb_rat = par_out
        model_ecl = wrap_ellc_lc(times, p_orb, t_zero, opt_f_c, opt_f_s, opt_i, opt_r_sum, opt_r_rat, opt_sb_rat)
        resid = ecl_signal - model_ecl
        bic = tsf.calc_bic(resid, 2 + len(par_out))
        print(f'Fit convergence: {result.success}. BIC: {bic:1.2f}. '
              f'N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')
    return par_out


@nb.njit(cache=True)
def objective_eclipse_sinusoids(params, times, signal, p_orb, t_zero, i_sectors):
    """The objective function to give to scipy.optimize.minimize
    for an eclipse model plus a sum of sine waves.

    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of the eclipse model and
        a set of sine waves and linear curve(s).
        Has to be a flat array and are ordered in the following way:
        [ecosw, esinw, cosi, phi_0, log_rr, log_sb,
         constant1, constant2, ..., slope1, slope2, ...,
         freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...]
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of the deepest minimum with respect to the mean time
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).

    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the residuals

    See Also
    --------
    linear_curve and sum_sines for the definition of the parameters.
    """
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_sin = (len(params) - 6 - 2 * n_sect) // 3  # each sine has freq, ampl and phase
    # separate the parameters
    params_ecl = params[0:6]
    ecosw, esinw, cosi, phi_0, log_rr, log_sb = params_ecl
    const = params[6:6 + n_sect]
    slope = params[6 + n_sect:6 + 2 * n_sect]
    freqs = params[6 + 2 * n_sect:6 + 2 * n_sect + n_sin]
    ampls = params[6 + 2 * n_sect + n_sin:6 + 2 * n_sect + 2 * n_sin]
    phases = params[6 + 2 * n_sect + 2 * n_sin:6 + 2 * n_sect + 3 * n_sin]
    # make the linear and sinusoid model
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, freqs, ampls, phases)
    # eclipse model
    e, w, i, r_sum, r_rat, sb_rat = ut.convert_to_phys_space(ecosw, esinw, cosi, phi_0, log_rr, log_sb)
    model_ecl = eclipse_physical_lc(times, p_orb, t_zero, e, w, i, r_sum, r_rat, sb_rat)
    # make the linear model and calculate the likelihood (minus this for minimisation)
    resid = signal - model_linear - model_sinusoid - model_ecl
    ln_likelihood = tsf.calc_likelihood(resid, times=times, signal_err=None, iid=True)
    return -ln_likelihood


# @nb.njit(cache=True)  # not possible due to sp.optimize.approx_fprime
def jacobian_eclipse_sinusoids(params, times, signal, p_orb, t_zero, i_sectors):
    """The jacobian function to give to scipy.optimize.minimize for a sum of sine waves,
    accompanying an eclipse model (no derivatives for the latter).

    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of the eclipse model and
        a set of sine waves and linear curve(s).
        Has to be a flat array and are ordered in the following way:
        [ecosw, esinw, cosi, phi_0, log_rr, log_sb,
         constant1, constant2, ..., slope1, slope2, ...,
         freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...]
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of the deepest minimum with respect to the mean time
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).

    Returns
    -------
    jac: float
        The derivative of minus the (natural)log-likelihood of the residuals

    See Also
    --------
    objective_eclipse_sinusoids
    """
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_sin = (len(params) - 6 - 2 * n_sect) // 3  # each sine has freq, ampl and phase
    # separate the parameters
    params_ecl = params[0:6]
    ecosw, esinw, cosi, phi_0, log_rr, log_sb = params_ecl
    params_sin = params[6:]  # includes linear pars as well
    const = params[6:6 + n_sect]
    slope = params[6 + n_sect:6 + 2 * n_sect]
    freqs = params[6 + 2 * n_sect:6 + 2 * n_sect + n_sin]
    ampls = params[6 + 2 * n_sect + n_sin:6 + 2 * n_sect + 2 * n_sin]
    phases = params[6 + 2 * n_sect + 2 * n_sin:6 + 2 * n_sect + 3 * n_sin]
    # make the linear and sinusoid model
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, freqs, ampls, phases)
    # eclipse model
    e, w, i, r_sum, r_rat, sb_rat = ut.convert_to_phys_space(ecosw, esinw, cosi, phi_0, log_rr, log_sb)
    model_ecl = eclipse_physical_lc(times, p_orb, t_zero, e, w, i, r_sum, r_rat, sb_rat)
    resid_ecl = signal - model_ecl
    # sinusoid part of the Jacobian
    jac_sin = jacobian_sinusoids(params_sin, times, resid_ecl, i_sectors)
    # numerically determine Jacobian for the params_ecl part
    resid_sin = signal - model_linear - model_sinusoid
    epsilon = 1.4901161193847656e-08
    args = (times, resid_sin, p_orb, t_zero)
    params_ecl = np.append(params_ecl, [0])  # account for the offset parameter
    jac_ecl = sp.optimize.approx_fprime(params_ecl, objective_physical_lc, epsilon, *args)
    jac_ecl = jac_ecl[:-1]
    jac = np.append(jac_ecl, jac_sin)
    return jac


# @nb.njit(cache=True)  # will not work due to ellc
def objective_ellc_sinusoids(params, times, signal, p_orb, t_zero, i_sectors):
    """The objective function to give to scipy.optimize.minimize
    for an ellc model plus a sum of sine waves.

    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of the eclipse model and
        a set of sine waves and linear curve(s).
        Has to be a flat array and are ordered in the following way:
        [ecosw, esinw, i, r_sum, log_rr, log_sb,
        constant1, constant2, ..., slope1, slope2, ...,
         freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...]
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of the deepest minimum with respect to the mean time
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).

    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the residuals

    See Also
    --------
    linear_curve and sum_sines for the definition of the parameters.
    """
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_sin = (len(params) - 6 - 2 * n_sect) // 3  # each sine has freq, ampl and phase
    # separate the parameters
    ecl_par = params[0:6]
    ecosw, esinw, cosi, phi_0, log_rr, log_sb = ecl_par
    const = params[6:6 + n_sect]
    slope = params[6 + n_sect:6 + 2 * n_sect]
    freqs = params[6 + 2 * n_sect:6 + 2 * n_sect + n_sin]
    ampls = params[6 + 2 * n_sect + n_sin:6 + 2 * n_sect + 2 * n_sin]
    phases = params[6 + 2 * n_sect + 2 * n_sin:6 + 2 * n_sect + 3 * n_sin]
    # make the sinusoid model
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, freqs, ampls, phases)
    # eclipse model
    e, w, i, r_sum, r_rat, sb_rat = ut.convert_to_phys_space(ecosw, esinw, cosi, phi_0, log_rr, log_sb)
    f_c, f_s = e**0.5 * np.cos(w), e**0.5 * np.sin(w)
    model_ecl = wrap_ellc_lc(times, p_orb, t_zero, f_c, f_s, i, r_sum, r_rat, sb_rat)
    # calculate the likelihood (minus this for minimisation)
    model = model_linear + model_sinusoid + model_ecl
    resid = signal - model
    ln_likelihood = tsf.calc_likelihood(resid, times=times, signal_err=None, iid=True)
    return -ln_likelihood


def fit_eclipse_physical_sinusoid(times, signal, p_orb, t_zero, ecl_par, const, slope, f_n, a_n, ph_n, i_sectors,
                                  model='simple', verbose=False):
    """Perform the multi-sinusoid, non-linear least-squares fit per frequency group
    and including an eclipse light curve model

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
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
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    model: str
        Which eclipse light curve model to use. Choose 'simple' or 'ellc'
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    res_const: numpy.ndarray[float]
        Updated y-intercepts of a piece-wise linear curve
    res_slope: numpy.ndarray[float]
        Updated slopes of a piece-wise linear curve
    res_freqs: numpy.ndarray[float]
        Updated frequencies of a number of sine waves
    res_ampls: numpy.ndarray[float]
        Updated amplitudes of a number of sine waves
    res_phases: numpy.ndarray[float]
        Updated phases of a number of sine waves
    res_ecl_par: numpy.ndarray[float]
        Updated eclipse parameters, consisting of:
        e, w, i, r_sum, r_rat, sb_rat

    Notes
    -----
    In reducing the overall runtime of the NL-LS fit, this will improve the
    fits on just the given groups of (closely spaced) frequencies, leaving the other
    frequencies as fixed parameters.
    """
    f_groups = ut.group_frequencies_for_fit(a_n, g_min=20, g_max=25)
    n_groups = len(f_groups)
    n_sect = len(i_sectors)
    n_sin = len(f_n)
    # make a copy of the initial parameters
    e, w, i, r_sum, r_rat, sb_rat = ecl_par
    ecosw, esinw, cosi, phi_0, log_rr, log_sb = ut.convert_from_phys_space(e, w, i, r_sum, r_rat, sb_rat)
    res_ecl_par = np.array([ecosw, esinw, cosi, phi_0, log_rr, log_sb])
    res_const = np.copy(np.atleast_1d(const))
    res_slope = np.copy(np.atleast_1d(slope))
    res_freqs = np.copy(f_n)
    res_ampls = np.copy(a_n)
    res_phases = np.copy(ph_n)
    # we don't want the frequencies to go lower than about 1/T/100
    t_tot = np.ptp(times)
    f_low = 0.01 / t_tot
    # update the parameters for each group
    for k, group in enumerate(f_groups):
        if verbose:
            print(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)}', end='\r')
        n_sin_g = len(res_freqs[group])
        # subtract all other sines from the data, they are fixed now
        resid = signal - tsf.sum_sines(times, np.delete(res_freqs, group), np.delete(res_ampls, group),
                                       np.delete(res_phases, group))
        # fit only the frequencies in this group (and eclipse model, constant and slope)
        par_init = np.concatenate((res_ecl_par, res_const, res_slope, res_freqs[group], res_ampls[group],
                                   res_phases[group]))
        par_bounds = [(-1, 1), (-1, 1), (0, 0.9), (0, 1.57), (-3, 3), (-3, 3)]
        par_bounds = par_bounds + [(None, None) for _ in range(2 * n_sect)]
        par_bounds = par_bounds + [(f_low, None) for _ in range(n_sin_g)]
        par_bounds = par_bounds + [(0, None) for _ in range(n_sin_g)] + [(None, None) for _ in range(n_sin_g)]
        arguments = (times, resid, p_orb, t_zero, i_sectors)
        if (model is 'ellc'):
            obj_fun = objective_ellc_sinusoids
        else:
            obj_fun = objective_eclipse_sinusoids
        result = sp.optimize.minimize(obj_fun, jac=jacobian_eclipse_sinusoids, x0=par_init, args=arguments,
                                      method='L-BFGS-B', bounds=par_bounds, options={'maxiter': 10**4 * len(par_init)})
        # separate results
        n_sin_g = len(res_freqs[group])
        res_ecl_par = result.x[0:6]
        res_const = result.x[6:6 + n_sect]
        res_slope = result.x[6 + n_sect:6 + 2 * n_sect]
        out_freqs = result.x[6 + 2 * n_sect:6 + 2 * n_sect + n_sin_g]
        out_ampls = result.x[6 + 2 * n_sect + n_sin_g:6 + 2 * n_sect + 2 * n_sin_g]
        out_phases = result.x[6 + 2 * n_sect + 2 * n_sin_g:6 + 2 * n_sect + 3 * n_sin_g]
        res_freqs[group] = out_freqs
        res_ampls[group] = out_ampls
        res_phases[group] = out_phases
        if verbose:
            model_linear = tsf.linear_curve(times, res_const, res_slope, i_sectors)
            model_sinusoid = tsf.sum_sines(times, res_freqs, res_ampls, res_phases)
            ecosw, esinw, cosi, phi_0, log_rr, log_sb = res_ecl_par
            e, w, i, r_sum, r_rat, sb_rat = ut.convert_to_phys_space(ecosw, esinw, cosi, phi_0, log_rr, log_sb)
            if (model is 'ellc'):
                f_c = res_ecl_par[0] / np.sqrt(e)
                f_s = res_ecl_par[1] / np.sqrt(e)
                model_ecl = wrap_ellc_lc(times, p_orb, t_zero, f_c, f_s, i, r_sum, r_rat, sb_rat)
            else:
                model_ecl = eclipse_physical_lc(times, p_orb, t_zero, e, w, i, r_sum, r_rat, sb_rat)
            resid_new = signal - (model_linear + model_sinusoid + model_ecl)
            bic = tsf.calc_bic(resid_new, 2 * n_sect + 3 * n_sin + 1 + len(res_ecl_par))
            print(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)} - BIC: {bic:1.2f}. '
                  f'N_iter: {result.nit}, N_fev: {int(result.nfev)}.')
    # convert back parameters
    ecosw, esinw, cosi, phi_0, log_rr, log_sb = res_ecl_par
    e, w, i, r_sum, r_rat, sb_rat = ut.convert_to_phys_space(ecosw, esinw, cosi, phi_0, log_rr, log_sb)
    res_ecl_par = np.array([e, w, i, r_sum, r_rat, sb_rat])
    return res_const, res_slope, res_freqs, res_ampls, res_phases, res_ecl_par


def delta_obj_physical_lc(par, i_par, params, obj_min, delta, times, signal, p_orb, t_zero):
    """Displaces the objective function minimum to a certain delta
    
    Parameters
    ----------
    par: float
        The parameter to change
    i_par: int
        The index in params of the parameter to change
    params: numpy.ndarray[float]
        The parameters of a simple eclipse light curve model.
        Has to be ordered in the following way:
        [ecosw, esinw, cosi, phi_0, log_rr, log_sb, offset]
    obj_min: float
        Minimum objective function value
    delta: float
        Target delta objective function to reach
    times: numpy.ndarray[float]
        Timestamps of the time series, zero point at primary minimum
    signal: numpy.ndarray[float]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of the deepest minimum with respect to the mean time

    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the residuals

    See Also
    --------
    objective_physcal_lc
    """
    params[i_par] = par
    obj = objective_physical_lc(params, times, signal, p_orb, t_zero)
    delta_obj = abs(obj - obj_min - delta)
    return delta_obj
