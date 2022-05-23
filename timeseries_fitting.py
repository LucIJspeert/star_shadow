"""STAR SHADOW
Satellite Time-series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This Python module contains functions for time-series analysis;
specifically for the fitting of stellar oscillations and eclipses.

Code written by: Luc IJspeert
"""

import numpy as np
import numba as nb
import scipy as sp
import scipy.optimize
import ellc

from . import timeseries_functions as tsf
from . import analysis_functions as af
from . import utility as ut


@nb.njit
def objective_sines(params, times, signal, i_sectors, verbose=False):
    """This is the objective function to give to scipy.optimize.minimize for a sum of sine waves.
    
    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of a set of sine waves and linear curve(s).
        Has to be a flat array and are ordered in the following way:
        [constant1, constant2, ..., slope1, slope2, ...,
         freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...]
    times: numpy.ndarray[float]
        Timestamps of the time-series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    
    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the resuduals

    See Also
    --------
    linear_curve and sum_sines for the definition of the parameters.
    """
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_sect) // 3  # each sine has freq, ampl and phase
    # separate the parameters
    const = params[0:n_sect]
    slope = params[n_sect:2 * n_sect]
    freqs = params[2 * n_sect:2 * n_sect + n_sin]
    ampls = params[2 * n_sect + n_sin:2 * n_sect + 2 * n_sin]
    phases = params[2 * n_sect + 2 * n_sin:2 * n_sect + 3 * n_sin]
    # make the model and calculate the likelihood
    model = tsf.linear_curve(times, const, slope, i_sectors)  # the linear part of the model
    model = model + tsf.sum_sines(times, freqs, ampls, phases)  # the sinusoid part of the model
    ln_likelihood = tsf.calc_likelihood(signal - model)  # need minus the likelihood for minimisation
    # to keep track, sometimes print the value
    if verbose:
        if np.random.randint(10000) == 0:
            print('log-likelihood:', ln_likelihood)
            # print(f'log-likelihood: {ln_likelihood:1.3f}')  # pre-jit
    return -ln_likelihood


def multi_sine_NL_LS_fit(times, signal, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Perform the multi-sinusoid, non-linear least-squares fit.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time-series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
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
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    res_const: numpy.ndarray[float]
        Updated y-intercept(s) of a piece-wise linear curve
    res_slope: numpy.ndarray[float]
        Updated slope(s) of a piece-wise linear curve
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
    # do the fit
    params_init = np.concatenate((res_const, res_slope, res_freqs, res_ampls, res_phases))
    result = sp.optimize.minimize(objective_sines, x0=params_init, args=(times, signal, i_sectors, verbose),
                                  method='Nelder-Mead', options={'maxfev': 10**5 * len(params_init)})
    # separate results
    res_const = result.x[0:n_sect]
    res_slope = result.x[n_sect:2 * n_sect]
    res_freqs = result.x[2 * n_sect:2 * n_sect + n_sin]
    res_ampls = result.x[2 * n_sect + n_sin:2 * n_sect + 2 * n_sin]
    res_phases = result.x[2 * n_sect + 2 * n_sin:2 * n_sect + 3 * n_sin]
    if verbose:
        model = tsf.linear_curve(times, res_const, res_slope, i_sectors)
        model += tsf.sum_sines(times, res_freqs, res_ampls, res_phases)
        bic = tsf.calc_bic(signal - model, 2 * n_sect + 3 * n_sin)
        print(f'Fit convergence: {result.success}. N_iter: {result.nit}. BIC: {bic:1.2f}')
    return res_const, res_slope, res_freqs, res_ampls, res_phases


def multi_sine_NL_LS_fit_per_group(times, signal, const, slope, f_n, a_n, ph_n, i_sectors, f_groups, verbose=False):
    """Perform the multi-sinusoid, non-linear least-squares fit per frequency group

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time-series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
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
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    f_groups: list[numpy.ndarray[int]]
        List of sets of frequencies to be fit separately in order
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    res_const: numpy.ndarray[float]
        Updated y-intercept(s) of a piece-wise linear curve
    res_slope: numpy.ndarray[float]
        Updated slope(s) of a piece-wise linear curve
    res_freqs: numpy.ndarray[float]
        Updated frequencies of a number of sine waves
    res_ampls: numpy.ndarray[float]
        Updated amplitudes of a number of sine waves
    res_phases: numpy.ndarray[float]
        Updated phases of a number of sine waves
    
    Notes
    -----
    In reducing the overall runtime of the NL-LS fit, this will improve the
    fits on just the given groups of (closely spaced) frequencies, leaving the other
    frequencies as fixed parameters.
    """
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
    for i, group in enumerate(f_groups):
        if verbose:
            print(f'Starting fit of group {i + 1} of {n_groups}')
        # subtract all other sines from the data, they are fixed now
        resid = signal - tsf.sum_sines(times, np.delete(res_freqs, group), np.delete(res_ampls, group),
                                       np.delete(res_phases, group))  # the sinusoid part of the model
        # fit only the frequencies in this group (constant and slope are also fitted still)
        output = multi_sine_NL_LS_fit(times, resid, res_const, res_slope, res_freqs[group], res_ampls[group],
                                      res_phases[group], i_sectors, verbose=verbose)
        res_const, res_slope, out_freqs, out_ampls, out_phases = output
        res_freqs[group] = out_freqs
        res_ampls[group] = out_ampls
        res_phases[group] = out_phases
        if verbose:
            model = tsf.linear_curve(times, res_const, res_slope, i_sectors)
            model += tsf.sum_sines(times, res_freqs, res_ampls, res_phases)
            bic = tsf.calc_bic(signal - model, 2 * n_sect + 3 * n_sin)
            print(f'BIC in fit convergence message above invalid. Actual BIC: {bic:1.2f}')
    return res_const, res_slope, res_freqs, res_ampls, res_phases


@nb.njit
def objective_sines_harmonics(params, times, signal, harmonic_n, i_sectors, verbose=False):
    """This is the objective function to give to scipy.optimize.minimize for a sum of sine waves
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
        Timestamps of the time-series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    harmonic_n: numpy.ndarray[int]
        Integer indicating which harmonic each index in 'harmonics'
        points to. n=1 for the base frequency (=orbital frequency)
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the resuduals

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
    # finally, make the model and calculate the likelihood
    model = tsf.linear_curve(times, const, slope, i_sectors)  # the linear part of the model
    model = model + tsf.sum_sines(times, freqs, ampls, phases)  # the sinusoid part of the model
    ln_likelihood = tsf.calc_likelihood(signal - model)  # need minus the likelihood for minimisation
    # to keep track, sometimes print the value
    if verbose:
        if np.random.randint(10000) == 0:
            print('log-likelihood:', ln_likelihood)
            # print(f'log-likelihood: {ln_likelihood:1.3f}')  # pre-jit
    return -ln_likelihood


def multi_sine_NL_LS_harmonics_fit(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Perform the multi-sinusoid, non-linear least-squares fit with harmonic frequencies.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time-series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    p_orb: float
        Orbital period of the eclipsing binary in days
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
    i_sectors: list[int], numpy.ndarray[int]
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
        Updated y-intercept(s) of a piece-wise linear curve
    res_slope: numpy.ndarray[float]
        Updated slope(s) of a piece-wise linear curve
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
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_f_tot = len(f_n)
    n_harm = len(harmonics)
    n_sin = n_f_tot - n_harm  # each independent sine has freq, ampl and phase
    not_harmonics = np.delete(np.arange(n_f_tot), harmonics)
    # do the fit
    params_init = np.concatenate(([p_orb], np.atleast_1d(const), np.atleast_1d(slope), np.delete(f_n, harmonics),
                                  np.delete(a_n, harmonics), np.delete(ph_n, harmonics),
                                  a_n[harmonics], ph_n[harmonics]))
    result = sp.optimize.minimize(objective_sines_harmonics, x0=params_init,
                                  args=(times, signal, harmonic_n, i_sectors, verbose), method='Nelder-Mead',
                                  options={'maxfev': 10**5 * len(params_init)})
    # separate results
    res_p_orb = result.x[0]
    res_const = result.x[1:1 + n_sect]
    res_slope = result.x[1 + n_sect:1 + 2 * n_sect]
    res_freqs = np.zeros(n_f_tot)
    res_freqs[not_harmonics] = result.x[1 + 2 * n_sect:1 + 2 * n_sect + n_sin]
    res_freqs[harmonics] = harmonic_n / res_p_orb
    res_ampls = np.zeros(n_f_tot)
    res_ampls[not_harmonics] = result.x[1 + 2 * n_sect + n_sin:1 + 2 * n_sect + 2 * n_sin]
    res_ampls[harmonics] = result.x[1 + 2 * n_sect + 3 * n_sin:1 + 2 * n_sect + 3 * n_sin + n_harm]
    res_phases = np.zeros(n_f_tot)
    res_phases[not_harmonics] = result.x[1 + 2 * n_sect + 2 * n_sin:1 + 2 * n_sect + 3 * n_sin]
    res_phases[harmonics] = result.x[1 + 2 * n_sect + 3 * n_sin + n_harm:1 + 2 * n_sect + 3 * n_sin + 2 * n_harm]
    if verbose:
        model = tsf.linear_curve(times, res_const, res_slope, i_sectors)
        model += tsf.sum_sines(times, res_freqs, res_ampls, res_phases)
        bic = tsf.calc_bic(signal - model, 1 + 2 * n_sect + 3 * n_sin + 2 * n_harm)
        print(f'Fit convergence: {result.success}. N_iter: {result.nit}. BIC: {bic:1.2f}')
    return res_p_orb, res_const, res_slope, res_freqs, res_ampls, res_phases


def multi_sine_NL_LS_harmonics_fit_per_group(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors,
                                             verbose=False):
    """Perform the multi-sinusoid, non-linear least-squares fit with harmonic frequencies
    per frequency group
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time-series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    p_orb: float
        Orbital period of the eclipsing binary in days
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
    i_sectors: list[int], numpy.ndarray[int]
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
        Updated y-intercept(s) of a piece-wise linear curve
    res_slope: numpy.ndarray[float]
        Updated slope(s) of a piece-wise linear curve
    res_freqs: numpy.ndarray[float]
        Updated frequencies of a number of sine waves
    res_ampls: numpy.ndarray[float]
        Updated amplitudes of a number of sine waves
    res_phases: numpy.ndarray[float]
        Updated phases of a number of sine waves
    
    Notes
    -----
    In reducing the overall runtime of the NL-LS fit, this will improve
    the fits per group of 20-25 frequencies, leaving the otherfrequencies
    as fixed parameters.
    Contrary to multi_sine_NL_LS_fit_per_group, the groups don't have to be provided
    as they are made with the default parameters of ut.group_fequencies_for_fit.
    The orbital harmonics are always the first group.
    """
    # get harmonics and group the remaining frequencies
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    indices = np.arange(len(f_n))
    i_non_harm = np.delete(indices, harmonics)
    f_groups = ut.group_fequencies_for_fit(a_n[i_non_harm], g_min=20, g_max=25)
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
        print(f'Starting fit of orbital harmonics')
    resid = signal - tsf.sum_sines(times, np.delete(res_freqs, harmonics), np.delete(res_ampls, harmonics),
                                   np.delete(res_phases, harmonics))  # the sinusoid part of the model without harmonics
    params_init = np.concatenate(([p_orb], res_const, res_slope, a_n[harmonics], ph_n[harmonics]))
    output = sp.optimize.minimize(objective_sines_harmonics, x0=params_init,
                                  args=(times, resid, harmonic_n, i_sectors, verbose), method='Nelder-Mead',
                                  options={'maxfev': 10**5 * len(params_init)})
    # separate results
    res_p_orb = output.x[0]
    res_const = output.x[1:1 + n_sect]
    res_slope = output.x[1 + n_sect:1 + 2 * n_sect]
    res_freqs[harmonics] = harmonic_n / res_p_orb
    res_ampls[harmonics] = output.x[1 + 2 * n_sect:1 + 2 * n_sect + n_harm]
    res_phases[harmonics] = output.x[1 + 2 * n_sect + n_harm:1 + 2 * n_sect + 2 * n_harm]
    if verbose:
        model = tsf.linear_curve(times, res_const, res_slope, i_sectors)
        model += tsf.sum_sines(times, res_freqs, res_ampls, res_phases)
        bic = tsf.calc_bic(signal - model, 1 + 2 * n_sect + 3 * n_sin + 2 * n_harm)
        print(f'Fit convergence: {output.success}. N_iter: {output.nit}. BIC: {bic:1.2f}')
    # update the parameters for each group
    for i, group in enumerate(f_groups):
        if verbose:
            print(f'Starting fit of group {i + 1} of {n_groups}')
        # subtract all other sines from the data, they are fixed now
        resid = signal - tsf.sum_sines(times, np.delete(res_freqs, group), np.delete(res_ampls, group),
                                       np.delete(res_phases, group))  # the sinusoid part of the model without group
        # fit only the frequencies in this group (constant and slope are also fitted still)
        output = multi_sine_NL_LS_fit(times, resid, res_const, res_slope, res_freqs[group], res_ampls[group],
                                      res_phases[group], i_sectors, verbose=verbose)
        res_const, res_slope, out_freqs, out_ampls, out_phases = output
        res_freqs[group] = out_freqs
        res_ampls[group] = out_ampls
        res_phases[group] = out_phases
        if verbose:
            model = tsf.linear_curve(times, res_const, res_slope, i_sectors)
            model += tsf.sum_sines(times, res_freqs, res_ampls, res_phases)
            bic = tsf.calc_bic(signal - model, 1 + 2 * n_sect + 3 * n_sin + 2 * n_harm)
            print(f'BIC in fit convergence message above invalid. Actual BIC: {bic:1.2f}')
    return res_p_orb, res_const, res_slope, res_freqs, res_ampls, res_phases


@nb.njit
def objective_third_light(params, times, signal, p_orb, a_h, ph_h, harmonic_n, i_sectors, verbose=False):
    """This is the objective function to give to scipy.optimize.minimize for a sum of sine waves
    plus a set of harmonic frequencies and the effect of third light.

    Parameters
    ----------
    params: numpy.ndarray[float]
        Third light for a set of sectors and a stretch parameter.
        Has to be ordered in the following way:
        [third_light1, third_light2, ..., stretch]
    times: numpy.ndarray[float]
        Timestamps of the time-series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    p_orb: float
        Orbital period of the eclipsing binary in days
    a_h: numpy.ndarray[float]
        Corresponding amplitudes of the orbital harmonics
    ph_h: numpy.ndarray[float]
        Corresponding phases of the orbital harmonics
    harmonic_n: numpy.ndarray[int]
        Integer indicating which harmonic each index in 'harmonics'
        points to. n=1 for the base frequency (=orbital frequency)
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the resuduals
    
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
    model = tsf.sum_sines(times, freqs, a_h * stretch, ph_h)  # the sinusoid part of the model
    # stretching the harmonic model is equivalent to multiplying the amplitudes (to get the right eclipse depth)
    const, slope = tsf.linear_slope(times, signal - model, i_sectors)
    model = model + tsf.linear_curve(times, const, slope, i_sectors)  # the linear part of the model
    model = ut.model_crowdsap(model, 1 - light_3, i_sectors)  # incorporate third light
    ln_likelihood = tsf.calc_likelihood(signal - model)  # need minus the likelihood for minimisation
    # to keep track, sometimes print the value
    if verbose:
        if np.random.randint(10000) == 0:
            print('log-likelihood:', ln_likelihood)
            # print(f'log-likelihood: {ln_likelihood:1.3f}')  # pre-jit
    return -ln_likelihood


def fit_minimum_third_light(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, verbose=False):
    """Fits for the minimum amount of third light needed in each sector.
    
    Since the contamination by third light can vary across (TESS) sectors,
    fully data driven fitting for third light will result in a lower bound for
    the flux fraction of contaminating light in the aperture per sector.
    (corresponding to an upper bound to CROWDSAP parameter, or 1-third_light)
    In the null hypothesis that our target is the one eclipsing, the deepest
    eclipses present in the data are in the sector with least third light.
    If we have the CROWDSAP parameter, this can be compared to check the null hypothesis
    or we can simply use the CROWDSAP as starting point.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time-series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    p_orb: float
        Orbital period of the eclipsing binary in days
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
    i_sectors: list[int], numpy.ndarray[int]
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
        Updated y-intercept(s) of a piece-wise linear curve
    slope: numpy.ndarray[float]
        Updated slope(s) of a piece-wise linear curve
    
    Notes
    -----
    For this to work we need at least an initial harmonic model of the eclipses
    (and at least a hand full of sectors).
    The given third light values are fractional values of a median-normalised light curve.
    """
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_f_tot = len(f_n)
    n_sin = len(non_harm)  # each independent sine has freq, ampl and phase
    n_harm = len(harmonics)
    model_init = tsf.sum_sines(times, f_n, a_n, ph_n)
    model_init += tsf.linear_curve(times, const, slope, i_sectors)
    bic_init = tsf.calc_bic(signal - model_init, 2 * n_sect + 1 + 2 * n_harm + 3 * n_sin)
    # start off at third light of 0.01 and stretch parameter of 1.01
    params_init = np.concatenate((np.zeros(n_sect) + 0.01, [1.01]))
    param_bounds = [(0, 1) if (i < n_sect) else (1, None) for i in range(n_sect + 1)]
    arguments = (times, signal, p_orb, a_n[harmonics], ph_n[harmonics], harmonic_n, i_sectors, verbose)
    # do the fit
    result = sp.optimize.minimize(objective_third_light, x0=params_init, args=arguments, method='Nelder-Mead',
                                  bounds=param_bounds, options={'maxfev': 10**5 * len(params_init)})
    # separate results
    res_light_3 = result.x[:n_sect]
    res_stretch = result.x[-1]
    model = tsf.sum_sines(times, f_n[harmonics], a_n[harmonics] * res_stretch, ph_n[harmonics])
    model += tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    const, slope = tsf.linear_slope(times, signal - model, i_sectors)
    if verbose:
        model += tsf.linear_curve(times, const, slope, i_sectors)
        model = ut.model_crowdsap(model, 1 - res_light_3, i_sectors)
        bic = tsf.calc_bic(signal - model, 2 * n_sect + 1 + 2 * n_harm + 3 * n_sin)
        print(f'Fit convergence: {result.success}. N_iter: {result.nit}. Old BIC: {bic_init:1.2f}. New BIC: {bic:1.2f}')
    return res_light_3, res_stretch, const, slope


def ellc_lc_simple(times, p_orb, t_zero, f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset):
    """Wrapper for a simple ELLC model with some fixed inputs
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time-series
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of deepest minimum modulo p_orb
    f_c: float
        Combination of e and w: sqrt(e)cos(w)
    f_s: float
        Combination of e and w: sqrt(e)sin(w)
    i: float
        Inclination of the orbit
    r_sum_sma: float
        Sum of radii in units of the semi-major axis
    r_ratio: float
        Radius ratio r_2/r_1
    sb_ratio: float
        Surface brightness ratio sb_2/sb_1
    offset: float
        Constant offset for the light level
    
    Returns
    -------
    model: numpy.ndarray[float]
        Eclipse light curve model for the given time points
    """
    incl = i / np.pi * 180  # ellc likes degrees
    r_1 = r_sum_sma / (1 + r_ratio)
    r_2 = r_sum_sma * r_ratio / (1 + r_ratio)
    # try to prevent fatal crashes from RLOF cases
    d_roche_1 = 2.44 * r_2 * (r_1 / r_2) # * (q)**(1 / 3)  # 2.44*R_M*(rho_M/rho_m)**(1/3)
    d_roche_2 = 2.44 * r_1 * (r_2 / r_1) # * (1 / q)**(1 / 3)
    d_peri = (1 - f_c**2 - f_s**2)  # a*(1 - e), but a=1
    if (max(d_roche_1, d_roche_2) > d_peri):
        model = np.ones(len(times))
    else:
        model = ellc.lc(times, r_1, r_2, f_c=f_c, f_s=f_s, incl=incl, sbratio=sb_ratio, period=p_orb, t_zero=t_zero,
                        light_3=0, q=1, shape_1='roche', shape_2='roche',
                        ld_1='lin', ld_2='lin', ldc_1=0.5, ldc_2=0.5, gdc_1=0., gdc_2=0., heat_1=0., heat_2=0.)
    # add constant offset for light level
    model = model + offset
    return model


def objective_ellc_lc(params, times, signal, signal_err, p_orb):
    """Objective function for a set of eclipse parameters
    
    Parameters
    ----------
    params: numpy.ndarray[float]
        The parameters of a simple eclipse light curve model.
        Has to be ordered in the following way:
        [f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset]
    times: numpy.ndarray[float]
        Timestamps of the time-series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days
    
    Returns
    -------
    -ln_likelihood: float
        Minus the (natural)log-likelihood of the resuduals
    
    See Also
    --------
    ellc_lc_simple
    """
    f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset = params
    r_1 = r_sum_sma / (1 + r_ratio)
    r_2 = r_sum_sma * r_ratio / (1 + r_ratio)
    # catch radii that come too close to the Roche lobe
    if (r_1 > 0.33) | (r_2 > 0.33):
        return 10**9
    try:
        model = ellc_lc_simple(times, p_orb, 0, f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset)
    except:
        # (try to) catch ellc errors
        return 10**9
    # determine likelihood for the model
    ln_likelihood = tsf.calc_likelihood((signal - model) / signal_err)  # need minus the likelihood for minimisation
    return -ln_likelihood


def fit_eclipse_ellc(times, signal, signal_err, p_orb, t_zero, timings, const, slope, f_n, a_n, ph_n, par_init,
                     par_bounds, i_sectors, verbose=False):
    """Perform least-squares fit for the orbital parameters that can be obtained
    from the eclipses in the light curve.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time-series
    signal: numpy.ndarray[float]
        Measurement values of the time-series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of deepest minimum modulo p_orb
    timings: numpy.ndarray[float]
        Eclipse timings of minima and first and last contact points
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2
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
    par_init: tuple[float], list[float], numpy.ndarray[float]
        Initial eclipse parameters to start the fit, consisting of:
        f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio
    par_bounds: tuple[tuple[float]], list[tuple[float]], numpy.ndarray[tuple[float]]
        Bounds on the eclipse parameters (par_init)
        (may also be different combinations of tuple/list/array)
    i_sectors: list[int], numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    result: Object
        Fit results object from the scipy optimizer
    
    See Also
    --------
    ellc_lc_simple, objective_ellc_lc
    
    Notes
    -----
    Strictly speaking it is doing a maximum log-likelihood fit, but that is
    in essence identical (and numerically more stable due to the logarithm).
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2 = timings
    f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio = par_init
    f_c_bd, f_s_bd, i_bd, r_sum_sma_bd, r_ratio_bd, sb_ratio_bd = par_bounds
    # make a time-series spanning a full orbital eclipse from primary first contact to primary last contact
    t_model = (times - t_zero) % p_orb
    ext_left = (t_model > p_orb + t_1_1)
    ext_right = (t_model < t_1_2)
    t_model = np.concatenate((t_model[ext_left] - p_orb, t_model, t_model[ext_right] + p_orb))
    # make a mask for the eclipses, as only the eclipses will be fitted
    mask = ((t_model > t_1_1) & (t_model < t_1_2)) | ((t_model > t_2_1) & (t_model < t_2_2))
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line + 1
    ecl_signal = np.concatenate((ecl_signal[ext_left], ecl_signal, ecl_signal[ext_right]))
    signal_err = np.concatenate((signal_err[ext_left], signal_err, signal_err[ext_right]))
    # f_c, f_s, i, r_sum, r_rat, sb_rat
    params_init = (f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, 0)
    # bounds = (f_c_bd, f_s_bd, i_bd, r_sum_sma_bd, r_ratio_bd, sb_ratio_bd, (-0.5, 0.5))
    # bounds = ((max(f_c - 2*(f_c - f_c_bd[0]), -1), min(f_c + 2*(f_c_bd[1] - f_c), 1)),
    #             (max(f_s - 2*(f_s - f_s_bd[0]), -1), min(f_s + 2*(f_s_bd[1] - f_s), 1)),
    #             (max(i - 2*(i - i_bd[0]), 0), min(i + 2*(i_bd[1] - i), np.pi/2)),
    #             (max(r_sum_sma - 2*(r_sum_sma - r_sum_sma_bd[0]), 0),
    #              min(r_sum_sma + 2*(r_sum_sma_bd[1] - r_sum_sma), 1)),
    #             (max(r_ratio - 2*(r_ratio - r_ratio_bd[0]), 0), min(r_ratio + 2*(r_ratio_bd[1] - r_ratio), 1)),
    #             (max(sb_ratio - 2*(sb_ratio - sb_ratio_bd[0]), 0), min(sb_ratio + 2*(sb_ratio_bd[1] - sb_ratio), 1)),
    #             (-0.5, 0.5))
    bounds = ((-1, 1), (-1, 1), (0, np.pi/2), (0, 1), (0.01, 100), (0.01, 100), (-0.5, 0.5))
    args = (t_model[mask], ecl_signal[mask], signal_err[mask], p_orb)
    result = sp.optimize.minimize(objective_ellc_lc, params_init, args=args, method='nelder-mead', bounds=bounds,
                               options={'maxiter': 10000})
    if verbose:
        print('Fit 1 complete')
        print(f'fun: {result.fun}')
        print(f'message: {result.message}')
        print(f'nfev: {result.nfev}, nit: {result.nit}, status: {result.status}, success: {result.success}')
    return result
