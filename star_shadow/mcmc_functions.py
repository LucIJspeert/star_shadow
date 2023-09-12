"""STAR SHADOW
Satellite Time series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This module contains functions for the use in and to perform
Markov Chain Monte Carlo (MCMC) with PyMC3

Code written by: Luc IJspeert
"""

import logging
import numpy as np
import scipy as sp
import scipy.stats

import arviz as az
try:
    # optional functionality
    import pymc3 as pm
    import theano.tensor as tt
    from fastprogress import fastprogress
except ImportError:
    pm = None
    tt = None
    fastprogress = None

from . import analysis_functions as af
from . import utility as ut


def fold_time_series(times, p_orb, t_zero, t_ext_1=0, t_ext_2=0):
    """Fold the given time series over the orbital period

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    p_orb: float
        The orbital period with which the time series is folded
    t_zero: float, None
        Reference zero point in time (with respect to the
        time series mean time) when the phase equals zero
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
    t_mean = tt.mean(times)
    t_folded = (times - t_mean - t_zero) % p_orb
    # extend to both sides
    ext_left = (t_folded > p_orb + t_ext_1)
    ext_right = (t_folded < t_ext_2)
    t_extended = tt.concatenate((t_folded[ext_left] - p_orb, t_folded, t_folded[ext_right] + p_orb))
    return t_extended, ext_left, ext_right


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
    Taken from timeseries_functions, but without JIT-ting
    
    Determines the slope and y-intercept with respect to the center
    between the two x-values.
    """
    slope = (y2 - y1) / (x2 - x1)
    y_inter = (y1 + y2) / 2  # halfway point is y-intercept for mean-centered x
    return y_inter, slope


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
    
    Notes
    -----
    Taken from utility, but without JIT-ting
    """
    y_inter, slope = linear_pars_two_points(xp1, yp1, xp2, yp2)
    y = y_inter + slope * (x - (xp1 + xp2) / 2)  # assumes output of y_inter is for mean-centered x
    return y


def interp(x, xp, yp):
    """Linear interpolation for a 1d array

    Parameters
    ----------
    x: numpy.ndarray[float]
        The x-coordinates at which to interpolate.
    xp: float, numpy.ndarray[float]
        The x-coordinates of the interpolation grid
    yp: float, numpy.ndarray[float]
        The y-coordinate of the interpolation grid

    Returns
    -------
    y: numpy.ndarray[float]
        The interpolated values, same shape as x.
    """
    # obtain indices of the points to either side of x
    insert_points = tt.extra_ops.searchsorted(xp, x, side='left')
    # insert_points_1 = tt.clip(insert_points, 1, 6284)
    xp1 = xp[insert_points - 1]
    yp1 = yp[insert_points - 1]
    # insert_points_2 = tt.clip(insert_points, 0, 6283)
    xp2 = xp[insert_points]
    yp2 = yp[insert_points]
    # calculate the y
    y = interp_two_points(x, xp1, yp1, xp2, yp2)
    return y


def true_anomaly(theta, w):
    """True anomaly in terms of the phase angle and argument of periastron

    Parameters
    ----------
    theta: float, numpy.ndarray[float]
        Phase angle (0 or pi degrees at conjunction)
    w: float, numpy.ndarray[float]
        Argument of periastron

    Returns
    -------
    nu: float, numpy.ndarray[float]
        True anomaly

    Notes
    -----
    Taken from analysis_functions, but without JIT-ting
    
    ν = π / 2 - ω + θ
    """
    nu = np.pi / 2 - w + theta
    return nu


def integral_kepler_2(nu_1, nu_2, e):
    """Integrated version of Keplers second law of areas

    Parameters
    ----------
    nu_1: float, numpy.ndarray[float]
        True anomaly value of the lower integral boundary
    nu_2: float, numpy.ndarray[float]
        True anomaly value of the upper integral boundary
    e: float, numpy.ndarray[float]
        Eccentricity

    Returns
    -------
    integral: float, numpy.ndarray[float]
        Outcome of the integral

    Notes
    -----
    Taken from analysis_functions and adapted, and without JIT-ting
    
    Returns the quantity 2π(t2 - t1)/P given an eccentricity (e) and
    corresponding true anomaly values ν1 and ν2.
    The indefinite integral formula is:
    2 arctan(sqrt(1 - e)sin(nu/2) / (sqrt(1 + e)cos(nu/2))) - e sqrt(1 - e**2)sin(nu) / (1 + e cos(nu))
    """
    
    def indefinite_integral(nu, ecc):
        term_1 = 2 * tt.arctan2(np.sqrt(1 - ecc) * tt.sin(nu / 2), tt.sqrt(1 + ecc) * tt.cos(nu / 2))
        term_2 = - ecc * tt.sqrt(1 - ecc**2) * tt.sin(nu) / (1 + ecc * tt.cos(nu))
        mod_term = 4 * np.pi * ((nu // (2 * np.pi) + 1) // 2)  # correction term for going over 2pi
        return term_1 + term_2 + mod_term
    
    end_boundary = indefinite_integral(nu_2, e)
    start_boundary = indefinite_integral(nu_1, e)
    integral = end_boundary - start_boundary
    return integral


def delta_deriv(theta, e, w, i):
    """Derivative of the projected normalised distance between the centres of the stars

    Parameters
    ----------
    theta: float, numpy.ndarray[float]
        Phase angle of the eclipse minimum
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit

    Returns
    -------
    minimize: float, numpy.ndarray[float]
        Numeric result of the function that should equal 0

    Notes
    -----
    Taken from analysis_functions and adapted, and without JIT-ting
    
    For circular orbits, delta has minima at 0 and 180 degrees, but this will deviate for
    eccentric *and* inclined orbits due to conjunction no longer lining up with the minimum
    projected separation between the stars.

    Minimize this function w.r.t. theta near zero to get the phase angle of minimum separation
    at primary eclipse (eclipse maximum), or near pi to get it for the secondary eclipse.
    """
    sin_i_2 = tt.sin(i)**2
    # previous (identical except for a factor 1/2 which doesn't matter because it equals zero) formula, from Kopal 1959
    # term_1 = (1 - e * np.sin(theta - w)) * sin_i_2 * np.sin(2*theta)
    # term_2 = 2 * e * np.cos(theta - w) * (1 - np.cos(theta)**2 * sin_i_2)
    minimize = e * tt.cos(theta - w) + sin_i_2 * tt.cos(theta) * (tt.sin(theta) - e * tt.cos(w))
    return minimize


def delta_deriv_2(theta, e, w, i):
    """Second derivative of the projected normalised distance between the centres of the stars

    Parameters
    ----------
    theta: float, numpy.ndarray[float]
        Phase angle of the eclipse minimum
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit

    Returns
    -------
    deriv: float, numpy.ndarray[float]
        Derivative value of the delta_deriv function
        
    Notes
    -----
    Taken from analysis_functions and adapted, and without JIT-ting
    """
    sin_i_2 = tt.sin(i)**2
    deriv = -e * tt.cos(w) * (1 - sin_i_2) * tt.sin(theta) + e * tt.sin(w) * tt.cos(theta) + sin_i_2 * tt.cos(2 * theta)
    return deriv


def minima_phase_angles_2(e, w, i):
    """Determine the phase angles of minima for given e, w, i

    Parameters
    ----------
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit

    Returns
    -------
    theta_1: float
        Phase angle of primary minimum
    theta_2: float
        Phase angle of secondary minimum
    theta_3: float
        Phase angle of maximum separation between 1 and 2
    theta_4: float
        Phase angle of maximum separation between 2 and 1

    Notes
    -----
    Taken from analysis_functions and adapted, and without JIT-ting
    
    Other implementation for minima_phase_angles that can be JIT-ted.
    On its own it is 10x slower, but as part of other functions it can be faster
    if it means that other function can then also be JIT-ted
    """
    x0 = tt.as_tensor_variable(np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2]))  # initial theta values
    if (e == 0):
        # this would break, so return the defaults for circular orbits
        return x0[0], x0[1], x0[2], x0[3]
    # use the derivative of the projected distance to get theta angles
    deriv_1 = delta_deriv(x0, e, w, i)  # value of the projected distance derivative
    deriv_2 = delta_deriv_2(x0, e, w, i)  # value of the second derivative
    walk_sign = -tt.cast(tt.sgn(deriv_1), 'int64') * tt.cast(tt.sgn(deriv_2), 'int64')
    # walk along the curve to find zero points
    two_pi = 2 * np.pi
    step = 0.001  # step in rad (does not determine final precision)
    # start at x0
    cur_x = x0
    cur_y = delta_deriv(cur_x, e, w, i)
    f_sign_x0 = tt.cast(tt.sgn(cur_y), 'int64')  # sign of delta_deriv at initial position
    # step in the desired direction
    try_x = cur_x + step * walk_sign
    try_y = delta_deriv(try_x, e, w, i)
    # check whether the sign stays the same
    check = (tt.sgn(cur_y) == tt.sgn(try_y))
    # if we take this many steps, we've gone full circle
    for _ in range(int(two_pi // step + 1)):
        if not np.any(check):
            break
        # make the approved steps and continue if any were approved
        cur_x = tt.switch(check, try_x, cur_x)  # cur_x[check] = try_x[check]
        cur_y = tt.switch(check, try_y, cur_y)  # cur_y[check] = try_y[check]
        # try the next steps
        try_x = tt.switch(check, cur_x + step * walk_sign, try_x)  # try_x[check]=cur_x[check]+step*walk_sign[check]
        try_y = tt.switch(check, delta_deriv(try_x, e, w, i), try_y)  # try_y[check]=delta_deriv(try_x[check], e, w, i)
        # check whether the sign stays the same
        # check[check] = (tt.sgn(cur_y[check]) == tt.sgn(try_y[check]))
        check = tt.switch(check, (tt.sgn(cur_y) == tt.sgn(try_y)), check)
    # interpolate for better precision than the angle step
    condition = tt.eq(f_sign_x0, 1)
    xp1 = tt.switch(condition, try_y, cur_y)
    yp1 = tt.switch(condition, try_x, cur_x)
    xp2 = tt.switch(condition, cur_y, try_y)
    yp2 = tt.switch(condition, cur_x, try_x)
    thetas_interp = interp_two_points(tt.zeros((4,)), xp1, yp1, xp2, yp2)
    thetas_interp = thetas_interp % two_pi
    # theta_1 is primary minimum, theta_2 is secondary minimum, the others are at the furthest projected distance
    theta_1, theta_3, theta_2, theta_4 = thetas_interp[0], thetas_interp[1], thetas_interp[2], thetas_interp[3]
    return theta_1, theta_2, theta_3, theta_4


def projected_separation(e, w, i, theta):
    """Projected separation between the centres of the two components
    at a given phase theta

    Parameters
    ----------
    e: float, numpy.ndarray[float]
        Eccentricity
    w: float, numpy.ndarray[float]
        Argument of periastron
    i: float, numpy.ndarray[float]
        Inclination of the orbit
    theta: float, numpy.ndarray[float]
        Phase angle (0 or pi at conjunction)

    Returns
    -------
    sep: float, numpy.ndarray[float]
        The projected separation in units of the
        semi-major axis.

    Notes
    -----
    Taken from analysis_functions and adapted, and without JIT-ting
    
    delta^2 = a^2 (1-e^2)^2(1 - sin^2(i)cos^2(theta))/(1 - e sin(theta - w))^2
    sep = delta/a
    """
    num = (1 - e**2)**2 * (1 - tt.sin(i)**2 * tt.cos(theta)**2)
    denom = (1 - e * tt.sin(theta - w))**2
    sep = tt.sqrt(num / denom)
    return sep


def covered_area(d, r_1, r_2):
    """Area covered for two overlapping circles separated by a certain distance

    Parameters
    ----------
    d: float, numpy.ndarray[float]
        Separation between the centres of the two circles
    r_1: float
        Radius of circle 1
    r_2: float
        Radius of circle 2

    Returns
    -------
    area: float
        Area covered by one circle overlapping the other

    Notes
    -----
    Taken from analysis_functions and adapted, and without JIT-ting
    
    For d between |r_1 - r_2| and r_1 + r_2:
    area = r_1^2 * arccos((d^2 + r_1^2 - r2^2)/(2 d r_1))
           + r_2^2 * arccos((d^2 + r_2^2 - r_1^2)/(2 d r_2))
           - r_1 r_2 sqrt(1 - ((r_1^2 + r_2^2 - d^2)/(2 r_1 r_2))^2)
    """
    # define conditions for separating parameter space
    cond_1 = (d > 1.00001 * abs(r_1 - r_2)) & (d < (r_1 + r_2))
    cond_2 = (d <= 1.00001 * abs(r_1 - r_2)) & tt.invert(cond_1)
    # formula for condition 1
    term_1 = r_1**2 * tt.arccos((d**2 + r_1**2 - r_2**2) / (2 * d * r_1))
    term_2 = r_2**2 * tt.arccos((d**2 + r_2**2 - r_1**2) / (2 * d * r_2))
    term_3 = - r_1 * r_2 * tt.sqrt(1 - ((r_1**2 + r_2**2 - d**2) / (2 * r_1 * r_2))**2)
    formula = term_1 + term_2 + term_3
    # value for condition 2
    value = np.pi * tt.minimum(r_1**2, r_2**2)
    # decision between the values
    area = tt.switch(cond_1, formula, 0)
    area = tt.switch(cond_2, value, area)
    return area


def eclipse_depth(e, w, i, theta, r_sum_sma, r_ratio, sb_ratio, theta_3, theta_4):
    """Theoretical eclipse depth in the assumption of uniform brightness

    Parameters
    ----------
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit
    theta: float, numpy.ndarray[float]
        Phase angle (0 or pi degrees at conjunction)
        Around 0, the light of the primary is blocked,
        around pi, the light of the secondary is blocked.
    r_sum_sma: float
        Sum of radii in units of the semi-major axis
    r_ratio: float
        Radius ratio r_2/r_1
    sb_ratio: float
        Surface brightness ratio sb_2/sb_1
    theta_3: float
        Phase angle of maximum separation between 1 and 2
    theta_4: float
        Phase angle of maximum separation between 2 and 1

    Returns
    -------
    light_lost: float
        Fractional loss of light at the given phase angle

    Notes
    -----
    Taken from analysis_functions and adapted, and without JIT-ting
    
    light_lost(1) = covered_area / (pi r_1^2 + pi r_2^2 sb_ratio)
    light_lost(2) = covered_area sb_ratio / (pi r_1^2 + pi r_2^2 sb_ratio)
    """
    # calculate radii and projected separation
    r_1 = r_sum_sma / (1 + r_ratio)
    r_2 = r_sum_sma * r_ratio / (1 + r_ratio)
    sep = projected_separation(e, w, i, theta)
    # with those, calculate the covered area and light lost due to that
    area = covered_area(sep, r_1, r_2)
    light_lost = area / (np.pi * r_1**2 + np.pi * r_2**2 * sb_ratio)
    light_lost = tt.switch(tt.eq(r_sum_sma, 0), tt.zeros_like(theta), light_lost)
    # factor sb_ratio depends on primary or secondary, theta ~ 180 is secondary
    cond_1 = (theta > theta_3) & (theta < theta_4)
    light_lost = tt.set_subtensor(light_lost[cond_1], light_lost[cond_1] * sb_ratio)
    return light_lost


def simple_eclipse_lc(times, p_orb, t_zero, e, w, i, r_sum_sma, r_ratio, sb_ratio):
    """Simple eclipse light curve model

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of the deepest minimum modulo p_orb
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit (radians)
    r_sum_sma: float
        Sum of radii in units of the semi-major axis
    r_ratio: float
        Radius ratio r_2/r_1
    sb_ratio: float
        Surface brightness ratio sb_2/sb_1

    Returns
    -------
    model: numpy.ndarray[float]
        Eclipse light curve model for the given time points
    
    Notes
    -----
    Taken from timeseries_fitting and adapted, and without JIT-ting
    """
    # position angle along the orbit (step just over 0.001)
    thetas = tt.as_tensor_variable(np.linspace(0, 2 * np.pi, 6284))
    # theta_1 is primary minimum, theta_2 is secondary minimum, the others are at the furthest projected distance
    theta_1, theta_2, theta_3, theta_4 = minima_phase_angles_2(e, w, i)
    # make the simple model
    ecl_model = 1 - eclipse_depth(e, w, i, thetas, r_sum_sma, r_ratio, sb_ratio, theta_3, theta_4)
    # determine the model times
    nu_1 = true_anomaly(theta_1, w)  # zero to good approximation
    nu_2 = true_anomaly(theta_1 + thetas, w)  # integral endpoints
    t_model = p_orb / (2 * np.pi) * integral_kepler_2(nu_1, nu_2, e)
    # interpolate the model (probably faster than trying to calculate the times)
    t_folded, _, _ = fold_time_series(times, p_orb, t_zero, t_ext_1=0, t_ext_2=0)
    # interp_model = np.interp(t_folded, t_model, ecl_model)
    interp_model = interp(t_folded, t_model, ecl_model)
    return interp_model


def sample_sinusoid(times, signal, const, slope, f_n, a_n, ph_n, c_err, sl_err, f_n_err, a_n_err, ph_n_err, noise_level,
                    i_sectors, verbose=False):
    """NUTS sampling of a linear + sinusoid + eclipse model
    
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
    c_err: numpy.ndarray[float]
        Uncertainty in the y-intercepts of a number of sine waves
    sl_err: numpy.ndarray[float]
        Uncertainty in the slopes of a number of sine waves
    f_n_err: numpy.ndarray[float]
        Uncertainty in the frequencies of a number of sine waves
    a_n_err: numpy.ndarray[float]
        Uncertainty in the amplitudes of a number of sine waves
    ph_n_err: numpy.ndarray[float]
        Uncertainty in the phases of a number of sine waves
    noise_level: float
        The noise level (standard deviation of the residuals)
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    inf_data: object
        Arviz inference data object
    par_means: list[float]
        Parameter mean values in the following order:
        const, slope, f_n, a_n, ph_n
    par_hdi: list[float]
        Parameter HDI error values, same order as par_means
    """
    # setup
    times_t = times.reshape(-1, 1)  # transposed times
    t_mean = tt.as_tensor_variable(np.mean(times))
    t_mean_s = tt.as_tensor_variable(np.array([np.mean(times[s[0]:s[1]]) for s in i_sectors]))
    lin_shape = (len(const),)
    sin_shape = (len(f_n),)
    # progress bar
    if verbose:
        fastprogress.printing = lambda: True
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.INFO)
    else:
        fastprogress.printing = lambda: False
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.ERROR)
    # make pymc3 model
    with pm.Model() as lc_model:
        # piece-wise linear curve parameter models
        const_pm = pm.Normal('const', mu=const, sigma=c_err, shape=lin_shape, testval=const)
        slope_pm = pm.Normal('slope', mu=slope, sigma=sl_err, shape=lin_shape, testval=slope)
        # piece-wise linear curve
        linear_curves = [const_pm[k] + slope_pm[k] * (times[s[0]:s[1]] - t_mean_s[k]) for k, s in enumerate(i_sectors)]
        model_linear = tt.concatenate(linear_curves)
        # sinusoid parameter models
        f_n_pm = pm.TruncatedNormal('f_n', mu=f_n, sigma=f_n_err, lower=0, shape=sin_shape, testval=f_n)
        a_n_pm = pm.TruncatedNormal('a_n', mu=a_n, sigma=a_n_err, lower=0, shape=sin_shape, testval=a_n)
        ph_n_pm = pm.VonMises('ph_n', mu=ph_n, kappa=1 / ph_n_err**2, shape=sin_shape, testval=ph_n)
        # sum of sinusoids
        model_sinusoid = pm.math.sum(a_n_pm * pm.math.sin((2 * np.pi * f_n_pm * (times_t - t_mean)) + ph_n_pm), axis=1)
        # full light curve model
        model = model_linear + model_sinusoid
        # observed distribution
        pm.Normal('obs', mu=model, sigma=noise_level, observed=signal)
    
    # do the sampling
    with lc_model:
        inf_data = pm.sample(draws=1000, tune=1000, init='adapt_diag', cores=1, progressbar=verbose,
                             return_inferencedata=True)
    
    if verbose:
        az.summary(inf_data, round_to=2, circ_var_names=['ph_n'])
    # stacked parameter chains
    const_ch = inf_data.posterior.const.stack(dim=['chain', 'draw']).to_numpy()
    slope_ch = inf_data.posterior.slope.stack(dim=['chain', 'draw']).to_numpy()
    f_n_ch = inf_data.posterior.f_n.stack(dim=['chain', 'draw']).to_numpy()
    a_n_ch = inf_data.posterior.a_n.stack(dim=['chain', 'draw']).to_numpy()
    ph_n_ch = inf_data.posterior.ph_n.stack(dim=['chain', 'draw']).to_numpy()
    # parameter means
    const_m = np.mean(const_ch, axis=1).flatten()
    slope_m = np.mean(slope_ch, axis=1).flatten()
    f_n_m = np.mean(f_n_ch, axis=1).flatten()
    a_n_m = np.mean(a_n_ch, axis=1).flatten()
    ph_n_m = sp.stats.circmean(ph_n_ch, axis=1).flatten()
    par_means = [const_m, slope_m, f_n_m, a_n_m, ph_n_m]
    # parameter errors (from hdi) [hdi expects (chain, draw) as first two axes... annoying warnings...]
    const_e = az.hdi(np.moveaxis(const_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    slope_e = az.hdi(np.moveaxis(slope_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    f_n_e = az.hdi(np.moveaxis(f_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    a_n_e = az.hdi(np.moveaxis(a_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    ph_n_e = az.hdi(np.moveaxis(ph_n_ch[np.newaxis], 1, 2), hdi_prob=0.683, circular=True)
    # convert interval to error bars
    const_e = np.column_stack([const_m - const_e[:, 0], const_e[:, 1] - const_m])
    slope_e = np.column_stack([slope_m - slope_e[:, 0], slope_e[:, 1] - slope_m])
    f_n_e = np.column_stack([f_n_m - f_n_e[:, 0], f_n_e[:, 1] - f_n_m])
    a_n_e = np.column_stack([a_n_m - a_n_e[:, 0], a_n_e[:, 1] - a_n_m])
    ph_n_e = np.column_stack([ph_n_m - ph_n_e[:, 0], ph_n_e[:, 1] - ph_n_m])
    par_hdi = [const_e, slope_e, f_n_e, a_n_e, ph_n_e]
    return inf_data, par_means, par_hdi


def sample_sinusoid_h(times, signal, p_orb, const, slope, f_n, a_n, ph_n, p_err, c_err, sl_err, f_n_err, a_n_err,
                      ph_n_err, noise_level, i_sectors, verbose=False):
    """NUTS sampling of a linear + sinusoid + eclipse model
    
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
    p_err: float
        Uncertainty in the orbital period
    c_err: numpy.ndarray[float]
        Uncertainty in the y-intercepts of a number of sine waves
    sl_err: numpy.ndarray[float]
        Uncertainty in the slopes of a number of sine waves
    f_n_err: numpy.ndarray[float]
        Uncertainty in the frequencies of a number of sine waves
    a_n_err: numpy.ndarray[float]
        Uncertainty in the amplitudes of a number of sine waves
    ph_n_err: numpy.ndarray[float]
        Uncertainty in the phases of a number of sine waves
    noise_level: float
        The noise level (standard deviation of the residuals)
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    inf_data: object
        Arviz inference data object
    par_means: list[float]
        Parameter mean values in the following order:
        p_orb, const, slope, f_n, a_n, ph_n
    par_hdi: list[float]
        Parameter HDI error values, same order as par_means
    """
    # setup
    times_t = times.reshape(-1, 1)  # transposed times
    t_mean = tt.as_tensor_variable(np.mean(times))
    t_mean_s = tt.as_tensor_variable(np.array([np.mean(times[s[0]:s[1]]) for s in i_sectors]))
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    lin_shape = (len(const),)
    sin_shape = (len(f_n[non_harm]),)
    harm_shape = (len(f_n[harmonics]),)
    # progress bar
    if verbose:
        fastprogress.printing = lambda: True
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.INFO)
    else:
        fastprogress.printing = lambda: False
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.ERROR)
    # make pymc3 model
    with pm.Model() as lc_model:
        # piece-wise linear curve parameter models
        const_pm = pm.Normal('const', mu=const, sigma=c_err, shape=lin_shape, testval=const)
        slope_pm = pm.Normal('slope', mu=slope, sigma=sl_err, shape=lin_shape, testval=slope)
        # piece-wise linear curve
        linear_curves = [const_pm[k] + slope_pm[k] * (times[s[0]:s[1]] - t_mean_s[k]) for k, s in enumerate(i_sectors)]
        model_linear = tt.concatenate(linear_curves)
        # sinusoid parameter models
        f_n_pm = pm.TruncatedNormal('f_n', mu=f_n[non_harm], sigma=f_n_err[non_harm], lower=0, shape=sin_shape,
                                    testval=f_n[non_harm])
        a_n_pm = pm.TruncatedNormal('a_n', mu=a_n[non_harm], sigma=a_n_err[non_harm], lower=0, shape=sin_shape,
                                    testval=a_n[non_harm])
        ph_n_pm = pm.VonMises('ph_n', mu=ph_n[non_harm], kappa=1 / ph_n_err[non_harm]**2, shape=sin_shape,
                              testval=ph_n[non_harm])
        # sum of sinusoids
        model_sinusoid = pm.math.sum(a_n_pm * pm.math.sin((2 * np.pi * f_n_pm * (times_t - t_mean)) + ph_n_pm), axis=1)
        # harmonic parameter models
        p_orb_pm = pm.TruncatedNormal('p_orb', mu=p_orb, sigma=p_err, lower=0, testval=p_orb)
        f_h_pm = pm.Deterministic('f_h', harmonic_n / p_orb_pm)
        a_h_pm = pm.TruncatedNormal('a_h', mu=a_n[harmonics], sigma=a_n_err[harmonics], lower=0, shape=harm_shape,
                                    testval=a_n[harmonics])
        ph_h_pm = pm.VonMises('ph_h', mu=ph_n[harmonics], kappa=1 / ph_n_err[harmonics]**2, shape=harm_shape,
                              testval=ph_n[harmonics])
        # sum of harmonic sinusoids
        model_harmonic = pm.math.sum(a_h_pm * pm.math.sin((2 * np.pi * f_h_pm * (times_t - t_mean)) + ph_h_pm), axis=1)
        # full light curve model
        model = model_linear + model_sinusoid + model_harmonic
        # observed distribution
        pm.Normal('obs', mu=model, sigma=noise_level, observed=signal)
    
    # do the sampling
    with lc_model:
        inf_data = pm.sample(draws=1000, tune=1000, init='adapt_diag', cores=1, progressbar=verbose,
                             return_inferencedata=True)
    
    if verbose:
        az.summary(inf_data, round_to=2, circ_var_names=['ph_n'])
    # stacked parameter chains
    p_orb_ch = inf_data.posterior.p_orb.stack(dim=['chain', 'draw']).to_numpy()
    const_ch = inf_data.posterior.const.stack(dim=['chain', 'draw']).to_numpy()
    slope_ch = inf_data.posterior.slope.stack(dim=['chain', 'draw']).to_numpy()
    f_n_ch = inf_data.posterior.f_n.stack(dim=['chain', 'draw']).to_numpy()
    a_n_ch = inf_data.posterior.a_n.stack(dim=['chain', 'draw']).to_numpy()
    ph_n_ch = inf_data.posterior.ph_n.stack(dim=['chain', 'draw']).to_numpy()
    f_h_ch = inf_data.posterior.f_h.stack(dim=['chain', 'draw']).to_numpy()
    a_h_ch = inf_data.posterior.a_h.stack(dim=['chain', 'draw']).to_numpy()
    ph_h_ch = inf_data.posterior.ph_h.stack(dim=['chain', 'draw']).to_numpy()
    # parameter means
    p_orb_m = np.mean(p_orb_ch)
    const_m = np.mean(const_ch, axis=1).flatten()
    slope_m = np.mean(slope_ch, axis=1).flatten()
    f_n_m = np.mean(f_n_ch, axis=1).flatten()
    a_n_m = np.mean(a_n_ch, axis=1).flatten()
    ph_n_m = sp.stats.circmean(ph_n_ch, axis=1).flatten()
    f_h_m = harmonic_n / p_orb_m  # taking the mean from the chain leads to slightly different results and is wrong
    a_h_m = np.mean(a_h_ch, axis=1).flatten()
    ph_h_m = sp.stats.circmean(ph_h_ch, axis=1).flatten()
    f_n_m = np.append(f_n_m, f_h_m)
    a_n_m = np.append(a_n_m, a_h_m)
    ph_n_m = np.append(ph_n_m, ph_h_m)
    par_means = [p_orb_m, const_m, slope_m, f_n_m, a_n_m, ph_n_m]
    # parameter errors (from hdi) [hdi expects (chain, draw) as first two axes... annoying warnings...]
    p_orb_e = az.hdi(p_orb_ch, hdi_prob=0.683)
    const_e = az.hdi(np.moveaxis(const_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    slope_e = az.hdi(np.moveaxis(slope_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    f_n_e = az.hdi(np.moveaxis(f_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    a_n_e = az.hdi(np.moveaxis(a_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    ph_n_e = az.hdi(np.moveaxis(ph_n_ch[np.newaxis], 1, 2), hdi_prob=0.683, circular=True)
    f_h_e = az.hdi(np.moveaxis(f_h_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    a_h_e = az.hdi(np.moveaxis(a_h_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    ph_h_e = az.hdi(np.moveaxis(ph_h_ch[np.newaxis], 1, 2), hdi_prob=0.683, circular=True)
    f_n_e = np.append(f_n_e, f_h_e, axis=0)
    a_n_e = np.append(a_n_e, a_h_e, axis=0)
    ph_n_e = np.append(ph_n_e, ph_h_e, axis=0)
    # convert interval to error bars
    p_orb_e = np.array([p_orb_m - p_orb_e[0], p_orb_e[1] - p_orb_m])
    const_e = np.column_stack([const_m - const_e[:, 0], const_e[:, 1] - const_m])
    slope_e = np.column_stack([slope_m - slope_e[:, 0], slope_e[:, 1] - slope_m])
    f_n_e = np.column_stack([f_n_m - f_n_e[:, 0], f_n_e[:, 1] - f_n_m])
    a_n_e = np.column_stack([a_n_m - a_n_e[:, 0], a_n_e[:, 1] - a_n_m])
    ph_n_e = np.column_stack([ph_n_m - ph_n_e[:, 0], ph_n_e[:, 1] - ph_n_m])
    par_hdi = [p_orb_e, const_e, slope_e, f_n_e, a_n_e, ph_n_e]
    return inf_data, par_means, par_hdi


def sample_sinusoid_eclipse(times, signal, p_orb, t_zero, ecl_par, const, slope, f_n, a_n, ph_n, ecl_par_err,
                            c_err, sl_err, f_n_err, a_n_err, ph_n_err, noise_level, i_sectors, verbose=False):
    """NUTS sampling of a linear + sinusoid + eclipse model

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
        e, w, i, r_sum_sma, r_ratio, sb_ratio
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
    ecl_par_err: numpy.ndarray[float]
        Uncertainty in the initial eclipse parameters to start the fit, consisting of:
        e_err, w_err, i_err, r_sum_err, r_rat_err, sb_rat_err,
        ecosw_err, esinw_err, cosi_err, phi_0_err, log_rr_err, log_sb_err
    c_err: numpy.ndarray[float]
        Uncertainty in the y-intercepts of a number of sine waves
    sl_err: numpy.ndarray[float]
        Uncertainty in the slopes of a number of sine waves
    f_n_err: numpy.ndarray[float]
        Uncertainty in the frequencies of a number of sine waves
    a_n_err: numpy.ndarray[float]
        Uncertainty in the amplitudes of a number of sine waves
    ph_n_err: numpy.ndarray[float]
        Uncertainty in the phases of a number of sine waves
    noise_level: float
        The noise level (standard deviation of the residuals)
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. If only a single curve is wanted,
        set i_sectors = np.array([[0, len(times)]]).
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    inf_data: object
        Arviz inference data object
    par_means: list[float]
        Parameter mean values in the following order:
        const, slope, f_n, a_n, ph_n, ecosw, esinw, cosi, phi_0, log_rr,
        log_sb, e, w, i, r_sum, r_rat, sb_rat
    par_hdi: list[float]
        Parameter HDI error values, same order as par_means
    """
    # unpack parameters
    e, w, i, r_sum, r_rat, sb_rat = ecl_par
    ecosw, esinw, cosi, phi_0, log_rr, log_sb = ut.convert_from_phys_space(e, w, i, r_sum, r_rat, sb_rat)
    ecosw_err, esinw_err, cosi_err, phi_0_err, log_rr_err, log_sb_err = ecl_par_err[6:]
    # setup
    times_t = times.reshape(-1, 1)  # transposed times
    t_mean = tt.as_tensor_variable(np.mean(times))
    t_mean_s = tt.as_tensor_variable(np.array([np.mean(times[s[0]:s[1]]) for s in i_sectors]))
    lin_shape = (len(const),)
    sin_shape = (len(f_n),)
    # progress bar
    if verbose:
        fastprogress.printing = lambda: True
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.INFO)
    else:
        fastprogress.printing = lambda: False
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.ERROR)
    # make pymc3 model
    with pm.Model() as lc_model:
        # piece-wise linear curve parameter models
        const_pm = pm.Normal('const', mu=const, sigma=c_err, shape=lin_shape, testval=const)
        slope_pm = pm.Normal('slope', mu=slope, sigma=sl_err, shape=lin_shape, testval=slope)
        # piece-wise linear curve
        linear_curves = [const_pm[k] + slope_pm[k] * (times[s[0]:s[1]] - t_mean_s[k]) for k, s in enumerate(i_sectors)]
        model_linear = tt.concatenate(linear_curves)
        # sinusoid parameter models
        f_n_pm = pm.TruncatedNormal('f_n', mu=f_n, sigma=f_n_err, lower=0, shape=sin_shape, testval=f_n)
        a_n_pm = pm.TruncatedNormal('a_n', mu=a_n, sigma=a_n_err, lower=0, shape=sin_shape, testval=a_n)
        ph_n_pm = pm.VonMises('ph_n', mu=ph_n, kappa=1 / ph_n_err**2, shape=sin_shape, testval=ph_n)
        # sum of sinusoids
        model_sinusoid = pm.math.sum(a_n_pm * pm.math.sin((2 * np.pi * f_n_pm * (times_t - t_mean)) + ph_n_pm), axis=1)
        # eclipse parameters
        ecosw_pm = pm.TruncatedNormal('ecosw', mu=ecosw, sigma=ecosw_err, lower=-1, upper=1, testval=ecosw)
        esinw_pm = pm.TruncatedNormal('esinw', mu=esinw, sigma=esinw_err, lower=-1, upper=1, testval=esinw)
        cosi_pm = pm.TruncatedNormal('cosi', mu=cosi, sigma=cosi_err, lower=0, upper=1, testval=cosi)
        phi_0_pm = pm.TruncatedNormal('phi_0', mu=phi_0, sigma=phi_0_err, lower=0, testval=phi_0)
        log_rr_pm = pm.TruncatedNormal('log_rr', mu=log_rr, sigma=log_rr_err, testval=log_rr)
        log_sb_pm = pm.TruncatedNormal('log_sb', mu=log_sb, sigma=log_sb_err, testval=log_sb)
        # some transformations (done to sample a less correlated parameter space)
        e_pm = pm.math.sqrt(ecosw_pm**2 + esinw_pm**2)
        w_pm = tt.arctan2(esinw_pm, ecosw_pm) % (2 * np.pi)
        incl_pm = tt.arccos(cosi_pm)
        r_sum_pm = pm.math.sqrt((1 - pm.math.sin(incl_pm)**2 * pm.math.cos(phi_0_pm)**2)) * (1 - e_pm**2)
        r_rat_pm = 10**log_rr_pm
        sb_rat_pm = 10**log_sb_pm
        # physical eclipse model
        model_eclipse = simple_eclipse_lc(times, p_orb, t_zero, e_pm, w_pm, incl_pm, r_sum_pm, r_rat_pm, sb_rat_pm)
        # full light curve model
        model = model_linear + model_sinusoid + model_eclipse
        # observed distribution
        pm.Normal('obs', mu=model, sigma=noise_level, observed=signal)

    # do the sampling
    with lc_model:
        inf_data = pm.sample(draws=1000, tune=1000, init='adapt_diag', chains=2, cores=1, progressbar=verbose,
                             return_inferencedata=True)

    if verbose:
        az.summary(inf_data, round_to=2, circ_var_names=['ph_n'])
    # stacked parameter chains
    const_ch = inf_data.posterior.const.stack(dim=['chain', 'draw']).to_numpy()
    slope_ch = inf_data.posterior.slope.stack(dim=['chain', 'draw']).to_numpy()
    f_n_ch = inf_data.posterior.f_n.stack(dim=['chain', 'draw']).to_numpy()
    a_n_ch = inf_data.posterior.a_n.stack(dim=['chain', 'draw']).to_numpy()
    ph_n_ch = inf_data.posterior.ph_n.stack(dim=['chain', 'draw']).to_numpy()
    ecosw_ch = inf_data.posterior.ecosw.stack(dim=['chain', 'draw']).to_numpy()
    esinw_ch = inf_data.posterior.esinw.stack(dim=['chain', 'draw']).to_numpy()
    cosi_ch = inf_data.posterior.cosi.stack(dim=['chain', 'draw']).to_numpy()
    phi_0_ch = inf_data.posterior.phi_0.stack(dim=['chain', 'draw']).to_numpy()
    log_rr_ch = inf_data.posterior.log_rr.stack(dim=['chain', 'draw']).to_numpy()
    log_sb_ch = inf_data.posterior.log_sb.stack(dim=['chain', 'draw']).to_numpy()
    # parameter transforms
    e_ch = np.sqrt(ecosw_ch**2 + esinw_ch**2)
    w_ch = np.arctan2(esinw_ch, ecosw_ch) % (2 * np.pi)
    i_ch = np.arccos(cosi_ch)
    r_sum_ch = af.r_sum_sma_from_phi_0(e_ch, i_ch, phi_0_ch)
    r_rat_ch = 10**log_rr_ch
    sb_rat_ch = 10**log_sb_ch
    # parameter means
    const_m = np.mean(const_ch, axis=1).flatten()
    slope_m = np.mean(slope_ch, axis=1).flatten()
    f_n_m = np.mean(f_n_ch, axis=1).flatten()
    a_n_m = np.mean(a_n_ch, axis=1).flatten()
    ph_n_m = sp.stats.circmean(ph_n_ch, axis=1).flatten()
    ecosw_m = np.mean(ecosw_ch)
    esinw_m = np.mean(esinw_ch)
    cosi_m = np.mean(cosi_ch)
    phi_0_m = np.mean(phi_0_ch)
    log_rr_m = np.mean(log_rr_ch)
    log_sb_m = np.mean(log_sb_ch)
    # transformed parameter means
    e_m = np.mean(e_ch)
    w_m = sp.stats.circmean(w_ch)
    i_m = np.mean(i_ch)
    r_sum_m = np.mean(r_sum_ch)
    r_rat_m = np.mean(r_rat_ch)
    sb_rat_m = np.mean(sb_rat_ch)
    par_means = [const_m, slope_m, f_n_m, a_n_m, ph_n_m, ecosw_m, esinw_m, cosi_m, phi_0_m, log_rr_m, log_sb_m,
                 e_m, w_m, i_m, r_sum_m, r_rat_m, sb_rat_m]
    # parameter errors (from hdi) [hdi expects (chain, draw) as first two axes... annoying warnings...]
    const_e = az.hdi(np.moveaxis(const_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    slope_e = az.hdi(np.moveaxis(slope_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    f_n_e = az.hdi(np.moveaxis(f_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    a_n_e = az.hdi(np.moveaxis(a_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    ph_n_e = az.hdi(np.moveaxis(ph_n_ch[np.newaxis], 1, 2), hdi_prob=0.683, circular=True)
    ecosw_e = az.hdi(ecosw_ch, hdi_prob=0.683)
    esinw_e = az.hdi(esinw_ch, hdi_prob=0.683)
    cosi_e = az.hdi(cosi_ch, hdi_prob=0.683)
    phi_0_e = az.hdi(phi_0_ch, hdi_prob=0.683)
    log_rr_e = az.hdi(log_rr_ch, hdi_prob=0.683)
    log_sb_e = az.hdi(log_sb_ch, hdi_prob=0.683)
    # transformed parameter interval (from hdi)
    e_e = az.hdi(e_ch.T, hdi_prob=0.683)
    w_e = az.hdi(w_ch.T, hdi_prob=0.683)
    i_e = az.hdi(i_ch, hdi_prob=0.683)
    r_sum_e = az.hdi(r_sum_ch.T, hdi_prob=0.683)
    r_rat_e = az.hdi(r_rat_ch, hdi_prob=0.683)
    sb_rat_e = az.hdi(sb_rat_ch, hdi_prob=0.683)
    # convert interval to error bars
    const_e = np.column_stack([const_m - const_e[:, 0], const_e[:, 1] - const_m])
    slope_e = np.column_stack([slope_m - slope_e[:, 0], slope_e[:, 1] - slope_m])
    f_n_e = np.column_stack([f_n_m - f_n_e[:, 0], f_n_e[:, 1] - f_n_m])
    a_n_e = np.column_stack([a_n_m - a_n_e[:, 0], a_n_e[:, 1] - a_n_m])
    ph_n_e = np.column_stack([ph_n_m - ph_n_e[:, 0], ph_n_e[:, 1] - ph_n_m])
    ecosw_e = np.array([ecosw_m - ecosw_e[0], ecosw_e[1] - ecosw_m])
    esinw_e = np.array([esinw_m - esinw_e[0], esinw_e[1] - esinw_m])
    cosi_e = np.array([cosi_m - cosi_e[0], cosi_e[1] - cosi_m])
    phi_0_e = np.array([phi_0_m - phi_0_e[0], phi_0_e[1] - phi_0_m])
    log_rr_e = np.array([log_rr_m - log_rr_e[0], log_rr_e[1] - log_rr_m])
    log_sb_e = np.array([log_sb_m - log_sb_e[0], log_sb_e[1] - log_sb_m])
    # transformed parameter error bars
    e_e = np.array([e_m - e_e[0], e_e[1] - e_m])
    w_e = np.array([w_m - w_e[0], w_e[1] - w_m])
    i_e = np.array([i_m - i_e[0], i_e[1] - i_m])
    r_sum_e = np.array([r_sum_m - r_sum_e[0], r_sum_e[1] - r_sum_m])
    r_rat_e = np.array([r_rat_m - r_rat_e[0], r_rat_e[1] - r_rat_m])
    sb_rat_e = np.array([sb_rat_m - sb_rat_e[0], sb_rat_e[1] - sb_rat_m])
    par_hdi = [const_e, slope_e, f_n_e, a_n_e, ph_n_e, ecosw_e, esinw_e, cosi_e, phi_0_e, log_rr_e, log_sb_e,
               e_e, w_e, i_e, r_sum_e, r_rat_e, sb_rat_e]
    return inf_data, par_means, par_hdi
