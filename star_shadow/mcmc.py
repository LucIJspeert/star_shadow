"""STAR SHADOW
Satellite Time-series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This module contains functions for the use in and to perform
Markov Chain Monte Carlo (MCMC) with PyMC3

Code written by: Luc IJspeert
"""

import numpy as np

import pymc3 as pm
import theano.tensor as tt
import arviz as az
# from fastprogress import fastprogress
# fastprogress.printing = lambda: True


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
    Taken directly from timeseries_functions, but without JIT-ting
    """
    slope = (y2 - y1) / (x2 - x1)
    y_inter = y1 - slope * x1  # take point 1 to calculate y intercept
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
    Taken directly from utility, but without JIT-ting
    """
    print(xp1.dtype, yp1.dtype, xp2.dtype, yp2.dtype, x.dtype)
    y_inter, slope = linear_pars_two_points(xp1, yp1, xp2, yp2)
    y = y_inter + slope * x
    return y


def true_anomaly(theta, w):
    """True anomaly in terms of the phase angle and argument of periastron

    Parameters
    ----------
    theta: float, np.ndarray[float]
        Phase angle (0 or pi degrees at conjunction)
    w: float, np.ndarray[float]
        Argument of periastron

    Returns
    -------
    nu: float, np.ndarray[float]
        True anomaly

    Notes
    -----
    Taken directly from analysis_functions, but without JIT-ting
    
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
    Taken directly from analysis_functions, but without JIT-ting
    
    Returns the quantity 2π(t2 - t1)/P given an eccentricity (e) and
    corresponding true anomaly values ν1 and ν2.
    The indefinite integral formula is:
    2 arctan(sqrt(1 - e)sin(nu/2) / (sqrt(1 + e)cos(nu/2))) - e sqrt(1 - e**2)sin(nu) / (1 + e cos(nu))
    """
    
    def indefinite_integral(nu, ecc):
        term_1 = 2 * np.arctan2(np.sqrt(1 - ecc) * np.sin(nu / 2), np.sqrt(1 + ecc) * np.cos(nu / 2))
        term_2 = - ecc * np.sqrt(1 - ecc**2) * np.sin(nu) / (1 + ecc * np.cos(nu))
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
    Taken directly from analysis_functions, but without JIT-ting
    
    For circular orbits, delta has minima at 0 and 180 degrees, but this will deviate for
    eccentric *and* inclined orbits due to conjunction no longer lining up with the minimum
    projected separation between the stars.

    Minimize this function w.r.t. theta near zero to get the phase angle of minimum separation
    at primary eclipse (eclipse maximum), or near pi to get it for the secondary eclipse.
    """
    sin_i_2 = np.sin(i)**2
    # previous (identical except for a factor 1/2 which doesn't matter because it equals zero) formula, from Kopal 1959
    # term_1 = (1 - e * np.sin(theta - w)) * sin_i_2 * np.sin(2*theta)
    # term_2 = 2 * e * np.cos(theta - w) * (1 - np.cos(theta)**2 * sin_i_2)
    minimize = e * np.cos(theta - w) + sin_i_2 * np.cos(theta) * (np.sin(theta) - e * np.cos(w))
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
    Taken directly from analysis_functions, but without JIT-ting
    """
    sin_i_2 = np.sin(i)**2
    deriv = -e * np.cos(w) * (1 - sin_i_2) * np.sin(theta) + e * np.sin(w) * np.cos(theta) + sin_i_2 * np.cos(2 * theta)
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
    Taken from analysis_functions and adapted, but without JIT-ting
    
    Other implementation for minima_phase_angles that can be JIT-ted.
    On its own it is 10x slower, but as part of other functions it can be faster
    if it means that other function can then also be JIT-ted
    """
    if (e == 0):
        # this would break, so return the defaults for circular orbits
        return 0, np.pi, np.pi / 2, 3 * np.pi / 2
    # use the derivative of the projected distance to get theta angles
    x0 = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
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
        cur_x[check] = try_x[check]
        cur_y[check] = try_y[check]
        # try the next steps
        try_x[check] = cur_x[check] + step * walk_sign[check]
        try_y[check] = delta_deriv(try_x[check], e, w, i)
        # check whether the sign stays the same
        check[check] = (tt.sgn(cur_y[check]) == tt.sgn(try_y[check]))
    # interpolate for better precision than the angle step
    condition = tt.as_tensor_variable(f_sign_x0 == 1)  # todo: https://theano-pymc.readthedocs.io/en/latest/library/tensor/basic.html
    xp1 = tt.switch(condition, try_y, cur_y)
    yp1 = tt.switch(condition, try_x, cur_x)
    xp2 = tt.switch(condition, cur_y, try_y)
    yp2 = tt.switch(condition, cur_x, try_x)
    print(xp1)
    print(try_y, cur_y)
    print(xp1.dtype, yp1.dtype, xp2.dtype, yp2.dtype, np.zeros(len(x0)).dtype)
    thetas_interp = interp_two_points(np.zeros(len(x0)), xp1, yp1, xp2, yp2)
    thetas_interp = thetas_interp % two_pi
    # theta_1 is primary minimum, theta_2 is secondary minimum, the others are at the furthest projected distance
    theta_1, theta_3, theta_2, theta_4 = thetas_interp
    return theta_1, theta_2, theta_3, theta_4


def projected_separation(e, w, i, theta):
    """Projected separation between the centres of the two components
    at a given phase theta

    Parameters
    ----------
    e: float, np.ndarray[float]
        Eccentricity
    w: float, np.ndarray[float]
        Argument of periastron
    i: float, np.ndarray[float]
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
    Taken directly from analysis_functions, but without JIT-ting
    
    delta^2 = a^2 (1-e^2)^2(1 - sin^2(i)cos^2(theta))/(1 - e sin(theta - w))^2
    sep = delta/a
    """
    num = (1 - e**2)**2 * (1 - np.sin(i)**2 * np.cos(theta)**2)
    denom = (1 - e * np.sin(theta - w))**2
    sep = np.sqrt(num / denom)
    return sep


def covered_area(d, r_1, r_2):
    """Area covered for two overlapping circles separated by a certain distance

    Parameters
    ----------
    d: float
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
    Taken directly from analysis_functions, but without JIT-ting
    
    For d between |r_1 - r_2| and r_1 + r_2:
    area = r_1^2 * arccos((d^2 + r_1^2 - r2^2)/(2 d r_1))
           + r_2^2 * arccos((d^2 + r_2^2 - r_1^2)/(2 d r_2))
           - r_1 r_2 sqrt(1 - ((r_1^2 + r_2^2 - d^2)/(2 r_1 r_2))^2)
    """
    if (d > 1.00001 * abs(r_1 - r_2)) & (d < (r_1 + r_2)):
        term_1 = r_1**2 * np.arccos((d**2 + r_1**2 - r_2**2) / (2 * d * r_1))
        term_2 = r_2**2 * np.arccos((d**2 + r_2**2 - r_1**2) / (2 * d * r_2))
        term_3 = - r_1 * r_2 * np.sqrt(1 - ((r_1**2 + r_2**2 - d**2) / (2 * r_1 * r_2))**2)
        area = term_1 + term_2 + term_3
    elif (d <= 1.00001 * abs(r_1 - r_2)):
        area = np.pi * min(r_1**2, r_2**2)
    else:
        area = 0
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
    theta: float
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
    Taken directly from analysis_functions, but without JIT-ting
    
    light_lost(1) = covered_area / (pi r_1^2 + pi r_2^2 sb_ratio)
    light_lost(2) = covered_area sb_ratio / (pi r_1^2 + pi r_2^2 sb_ratio)
    """
    # calculate radii and projected separation
    r_1 = r_sum_sma / (1 + r_ratio)
    r_2 = r_sum_sma * r_ratio / (1 + r_ratio)
    sep = projected_separation(e, w, i, theta)
    # with those, calculate the covered area and light lost due to that
    if (r_sum_sma == 0):
        light_lost = 0
    else:
        area = covered_area(sep, r_1, r_2)
        light_lost = area / (np.pi * r_1**2 + np.pi * r_2**2 * sb_ratio)
    # factor sb_ratio depends on primary or secondary, theta ~ 180 is secondary
    if (theta > theta_3) & (theta < theta_4):
        light_lost = light_lost * sb_ratio
    return light_lost


def simple_eclipse_lc(times, p_orb, t_zero, e, w, i, r_sum_sma, r_ratio, sb_ratio):
    """Simple eclipse light curve model

    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time-series
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
    Taken directly from timeseries_fitting, but without JIT-ting
    """
    thetas = np.arange(0, 2 * np.pi, 0.001)  # position angle along the orbit
    # theta_1 is primary minimum, theta_2 is secondary minimum, the others are at the furthest projected distance
    theta_1, theta_2, theta_3, theta_4 = minima_phase_angles_2(e, w, i)
    # make the simple model
    ecl_model = np.zeros(len(thetas))
    for k in range(len(thetas)):
        ecl_model[k] = 1 - eclipse_depth(e, w, i, thetas[k], r_sum_sma, r_ratio, sb_ratio, theta_3, theta_4)
    # determine the model times
    nu_1 = true_anomaly(theta_1, w)  # zero to good approximation
    nu_2 = true_anomaly(theta_1 + thetas, w)  # integral endpoints
    t_model = p_orb / (2 * np.pi) * integral_kepler_2(nu_1, nu_2, e)
    # interpolate the model (probably faster than trying to calculate the times)
    t_folded = (times - t_zero) % p_orb
    interp_model = np.interp(t_folded, t_model, ecl_model)
    return interp_model
