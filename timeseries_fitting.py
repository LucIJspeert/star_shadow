"""STAR SHADOW

This Python module contains functions for time series analysis;
specifically for the fitting of stellar oscillations and eclipses.

Code written by: Luc IJspeert
"""

import numpy as np
import numba as nb
import scipy as sp
import scipy.spatial
import ellc

import utility as ut
import timeseries_functions as tsf
import analysis_functions as af


@nb.njit
def objective_sines(params, times, signal, i_sectors, verbose=False):
    """This is the objective function to give to scipy.optimize.minimize for a sum of sine waves.

    The parameters (params) have to be a flat array and are ordered in the following way:
    params = array(constant1, constant2, ..., slope1, slope2, ..., freq1, freg2, ..., ampl1, ampl2, ...,
                   phase1, phase2, ...)
    See linear_curve and sum_sines for the definition of the parameters.
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
    freqs = np.copy(f_n)
    ampls = np.copy(a_n)
    phases = np.copy(ph_n)
    # update the parameters for each group
    for i, group in enumerate(f_groups):
        if verbose:
            print(f'Starting fit of group {i + 1} of {n_groups}')
        # subtract all other sines from the data, they are fixed now
        resid = signal - tsf.sum_sines(times, np.delete(freqs, group), np.delete(ampls, group),
                                       np.delete(phases, group))  # the sinusoid part of the model
        # fit only the frequencies in this group (constant and slope are also fitted still)
        result = multi_sine_NL_LS_fit(times, resid, res_const, res_slope, freqs[group], ampls[group],
                                      phases[group], i_sectors, verbose=verbose)
        res_const, res_slope, res_freqs, res_ampls, res_phases = result
        freqs[group] = res_freqs
        ampls[group] = res_ampls
        phases[group] = res_phases
        if verbose:
            model = tsf.linear_curve(times, res_const, res_slope, i_sectors)
            model += tsf.sum_sines(times, freqs, ampls, phases)
            bic = tsf.calc_bic(signal - model, 2 * n_sect + 3 * n_sin)
            print(f'BIC in fit convergence message above invalid. Actual BIC: {bic:1.2f}')
    return res_const, res_slope, freqs, ampls, phases


@nb.njit
def objective_sines_harmonics(params, times, signal, harmonic_n, i_sectors, verbose=False):
    """This is the objective function to give to scipy.optimize.minimize for a sum of sine waves
    plus a set of harmonic frequencies.

    The parameters (params) have to be a flat array and are ordered in the following way:
    params = array(p_orb, constant1, constant2, ..., slope1, slope2, ..., freq1, freg2, ..., ampl1, ampl2, ...,
                   phase1, phase2, ..., ampl_h1, ampl_h2, ..., phase_h1, phase_h2, ...)
    where _hi indicates harmonics. The harmonic number must also be provided in the order that
    the harmonics appear in the parameters.
    See linear_curve and sum_sines for the definition of the parameters.
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

    In reducing the overall runtime of the NL-LS fit, this will improve the
    fits on just the given groups of (closely spaced) frequencies, leaving the other
    frequencies as fixed parameters.
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
    freqs = np.copy(f_n)
    ampls = np.copy(a_n)
    phases = np.copy(ph_n)
    # fit the harmonics (first group)
    if verbose:
        print(f'Starting fit of orbital harmonics')
    resid = signal - tsf.sum_sines(times, np.delete(freqs, harmonics), np.delete(ampls, harmonics),
                                   np.delete(phases, harmonics))  # the sinusoid part of the model without harmonics
    params_init = np.concatenate(([p_orb], res_const, res_slope, a_n[harmonics], ph_n[harmonics]))
    result = sp.optimize.minimize(objective_sines_harmonics, x0=params_init,
                                  args=(times, resid, harmonic_n, i_sectors, verbose), method='Nelder-Mead',
                                  options={'maxfev': 10**5 * len(params_init)})
    # separate results
    res_p_orb = result.x[0]
    res_const = result.x[1:1 + n_sect]
    res_slope = result.x[1 + n_sect:1 + 2 * n_sect]
    freqs[harmonics] = harmonic_n / res_p_orb
    ampls[harmonics] = result.x[1 + 2 * n_sect:1 + 2 * n_sect + n_harm]
    phases[harmonics] = result.x[1 + 2 * n_sect + n_harm:1 + 2 * n_sect + 2 * n_harm]
    if verbose:
        model = tsf.linear_curve(times, res_const, res_slope, i_sectors)
        model += tsf.sum_sines(times, freqs, ampls, phases)
        bic = tsf.calc_bic(signal - model, 1 + 2 * n_sect + 3 * n_sin + 2 * n_harm)
        print(f'Fit convergence: {result.success}. N_iter: {result.nit}. BIC: {bic:1.2f}')
    # update the parameters for each group
    for i, group in enumerate(f_groups):
        if verbose:
            print(f'Starting fit of group {i + 1} of {n_groups}')
        # subtract all other sines from the data, they are fixed now
        resid = signal - tsf.sum_sines(times, np.delete(freqs, group), np.delete(ampls, group),
                                       np.delete(phases, group))  # the sinusoid part of the model without group
        # fit only the frequencies in this group (constant and slope are also fitted still)
        result = multi_sine_NL_LS_fit(times, resid, res_const, res_slope, freqs[group], ampls[group],
                                      phases[group], i_sectors, verbose=verbose)
        res_const, res_slope, res_freqs, res_ampls, res_phases = result
        freqs[group] = res_freqs
        ampls[group] = res_ampls
        phases[group] = res_phases
        if verbose:
            model = tsf.linear_curve(times, res_const, res_slope, i_sectors)
            model += tsf.sum_sines(times, freqs, ampls, phases)
            bic = tsf.calc_bic(signal - model, 1 + 2 * n_sect + 3 * n_sin + 2 * n_harm)
            print(f'BIC in fit convergence message above invalid. Actual BIC: {bic:1.2f}')
    return res_p_orb, res_const, res_slope, freqs, ampls, phases


@nb.njit
def objective_third_light(params, times, signal, p_orb, const, slope, a_h, ph_h, harmonic_n, i_sectors, verbose=False):
    """This is the objective function to give to scipy.optimize.minimize for a sum of sine waves
    plus a set of harmonic frequencies and the effect of third light.

    The parameters (params) have to be a flat array and are ordered in the following way:
    params = array(third_light1, third_light2, ..., ampl_h1, ampl_h2, ...)
    where _hi indicates harmonics. The harmonic number (harmonic_n) must also be provided
    in the order that the harmonics appear in the parameters.
    See linear_curve and sum_sines for the definition of some of the the parameters.
    """
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_harm = len(harmonic_n)
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
    In the null hypothesis that our target is the one eclipsing, the lowest depths
    present in the data are in the sector with least third light.
    If we have the CROWDSAP parameter, this can be compared to check the null hypothesis
    or we can simply use the CROWDSAP as starting point.
    
    For this to work we need at least an initial harmonic model of the eclipses.
    The given third light values are fractional values of a median-normalised light curve.
    """
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    n_sect = len(i_sectors)  # each sector has its own slope (or two)
    n_f_tot = len(f_n)
    n_sin = len(non_harm)  # each independent sine has freq, ampl and phase
    n_harm = len(harmonics)
    if verbose:
        model = tsf.sum_sines(times, f_n, a_n, ph_n)
        model += tsf.linear_curve(times, const, slope, i_sectors)
        bic_init = tsf.calc_bic(signal - model, 2 * n_sect + 1 + 2 * n_harm + 3 * n_sin)
    # start off at third light of 0.01 and stretch parameter of 1.01
    params_init = np.concatenate((np.zeros(n_sect) + 0.01, [1.01]))
    param_bounds = [(0, 1) if (i < n_sect) else (1, None) for i in range(n_sect + 1)]
    arguments = (times, signal, p_orb, const, slope, a_n[harmonics], ph_n[harmonics], harmonic_n, i_sectors, verbose)
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
    """Wrapper for a simple ELLC model with some fixed inputs"""
    incl = i / np.pi * 180  # ellc likes degrees
    r_1 = r_sum_sma / (1 + r_ratio)
    r_2 = r_sum_sma * r_ratio / (1 + r_ratio)
    model = ellc.lc(times, r_1, r_2, f_c=f_c, f_s=f_s, incl=incl, sbratio=sb_ratio, period=p_orb, t_zero=t_zero,
                    light_3=0, q=1, shape_1='roche', shape_2='roche',
                    ld_1='lin', ld_2='lin', ldc_1=0.5, ldc_2=0.5, gdc_1=0., gdc_2=0., heat_1=0., heat_2=0.)
    # add constant offset for light level
    model = model + offset
    return model


def objective_ellc_lc(params, times, signal, signal_err, p_orb, timings):
    """Objective function for a set of eclipse parameters"""
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


def fit_eclipse_ellc(times, signal, signal_err, p_orb, t_zero, timings, const, slope, f_n, a_n, ph_n, i_sectors,
                     par_init, par_bounds, verbose=False):
    """Perform least-squares fit for the orbital parameters that can be obtained
    from the eclipses in the light curve.
    
    Strictly speaking it is doing a maximum log-likelihood fit, but that is
    in essence identical (and numerically more stable due to the logarithm).
    The fit has free parameters:
    sqrt(e)*cos(w), sqrt(e)*sin(w), i, (r1+r2)/a, r2/r1, sb_ratio
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
    #             (max(r_sum_sma - 2*(r_sum_sma - r_sum_sma_bd[0]), 0), min(r_sum_sma + 2*(r_sum_sma_bd[1] - r_sum_sma), 1)),
    #             (max(r_ratio - 2*(r_ratio - r_ratio_bd[0]), 0), min(r_ratio + 2*(r_ratio_bd[1] - r_ratio), 1)),
    #             (max(sb_ratio - 2*(sb_ratio - sb_ratio_bd[0]), 0), min(sb_ratio + 2*(sb_ratio_bd[1] - sb_ratio), 1)),
    #             (-0.5, 0.5))
    bounds = ((-1, 1), (-1, 1), (0, np.pi/2), (0, 1), (0.01, 100), (0.01, 100), (-0.5, 0.5))
    args = (t_model[mask], ecl_signal[mask], signal_err[mask], p_orb, timings)
    res = sp.optimize.minimize(objective_ellc_lc, params_init, args=args, method='nelder-mead', bounds=bounds,
                               options={'maxiter':10000})
    if verbose:
        print('Fit 1 complete')
        print(f'fun: {res.fun}')
        print(f'message: {res.message}')
        print(f'nfev: {res.nfev}, nit: {res.nit}, status: {res.status}, success: {res.success}')
    return res


