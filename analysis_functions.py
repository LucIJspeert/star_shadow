"""STAR SHADOW
Satellite Time-series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This Python module contains functions for data analysis;
specifically for the analysis of stellar variability and eclipses.

Code written by: Luc IJspeert
"""

import numpy as np
import scipy as sp
import scipy.signal
import scipy.stats
import scipy.optimize
import numba as nb
import itertools as itt

import arviz

from . import timeseries_functions as tsf
from . import utility as ut


@nb.njit()
def f_within_rayleigh(i, f_n, rayleigh):
    """Selects a chain of frequencies within the Rayleigh criterion from each other
    around the chosen frequency.
    
    Parameters
    ----------
    i: int
        Index of the frequency around which to search
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    rayleigh: float
        The appropriate frequency resolution (usually 1.5/T)
    
    Returns
    -------
    i_close_unsorted: numpy.ndarray[int]
        Indices of close frequencies in the chain
    """
    indices = np.arange((len(f_n)))
    sorter = np.argsort(f_n)  # first sort by frequency
    f_diff = np.diff(f_n[sorter])  # spaces between frequencies
    sorted_pos = indices[sorter == i][0]  # position of i in the sorted array
    if np.all(f_diff > rayleigh):
        # none of the frequencies are close
        i_close = np.zeros(0, dtype=np.int_)
    elif np.all(f_diff < rayleigh):
        # all of the frequencies are close
        i_close = indices
    else:
        # the frequency to investigate is somewhere inbetween other frequencies
        right_not_close = indices[sorted_pos + 1:][f_diff[sorted_pos:] > rayleigh]
        left_not_close = indices[:sorted_pos][f_diff[:sorted_pos] > rayleigh]
        # if any freqs to left or right are not close, take the first index where this happens
        if (len(right_not_close) > 0):
            i_right_nc = right_not_close[0]
        else:
            i_right_nc = len(f_n)  # else take the right edge of the array
        if (len(left_not_close) > 0):
            i_left_nc = left_not_close[-1]
        else:
            i_left_nc = -1  # else take the left edge of the array (minus one)
        # now select the frequencies close to f_n[i] by using the found boundaries
        i_close = indices[i_left_nc + 1:i_right_nc]
    # convert back to unsorted indices
    i_close_unsorted = sorter[i_close]
    return i_close_unsorted


@nb.njit()
def chains_within_rayleigh(f_n, rayleigh):
    """Find all chains of frequencies within each others Rayleigh criterion.
    
    Parameters
    ----------
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    rayleigh: float
        The appropriate frequency resolution (usually 1.5/T)
    
    Returns
    -------
    groups: list[numpy.ndarray[int]]
        Indices of close frequencies in all found chains
    
    See Also
    --------
    f_within_rayleigh
    """
    indices = np.arange(len(f_n))
    used = []
    groups = []
    for i in indices:
        if i not in used:
            i_close = f_within_rayleigh(i, f_n, rayleigh)
            if (len(i_close) > 1):
                used.extend(i_close)
                groups.append(i_close)
    return groups


@nb.njit()
def remove_insignificant_sigma(f_n, f_n_err, a_n, a_n_err, sigma_a=3., sigma_f=1.):
    """Removes insufficiently significant frequencies in terms of error margins.
    
    Parameters
    ----------
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    f_n_err: numpy.ndarray[float]
        Formal errors in the frequencies
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    a_n_err: numpy.ndarray[float]
        Formal errors in the amplitudes
    sigma_a: float
        Number of times the error to use for check of significant amplitude
    sigma_f: float
        Number of times the error to use for check of significant
        frequency separation
    
    Returns
    -------
    remove: numpy.ndarray[int]
        Indices of frequencies deemed insignificant
    
    Notes
    -----
    Frequencies with an amplitude less than sigma times the error are removed,
    as well as those that have an overlapping frequency error and are lower amplitude
    than any of the overlapped frequencies.
    """
    # amplitude not significant enough
    a_insig = (a_n / a_n_err < sigma_a)
    # frequency error overlaps with neighbour
    f_insig = np.zeros(len(f_n), dtype=np.bool_)
    for i in range(len(f_n)):
        overlap = (f_n[i] + sigma_f * f_n_err[i] > f_n) & (f_n[i] - sigma_f * f_n_err[i] < f_n)
        # if any of the overlap is higher in amplitude, throw this one out
        if np.any((a_n[overlap] > a_n[i]) & (f_n[overlap] != f_n[i])):
            f_insig[i] = True
    remove = np.arange(len(f_n))[a_insig | f_insig]
    return remove


@nb.njit()
def remove_insignificant_snr(a_n, noise_level, n_points):
    """Removes insufficiently significant frequencies in terms of S/N.
    
    Parameters
    ----------
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    noise_level: float
        The noise level (standard deviation of the residuals)
    n_points: int
        Number of data points
    
    Returns
    -------
    remove: numpy.ndarray[int]
        Indices of frequencies deemed insignificant
    
    Notes
    -----
    Frequencies with an amplitude less than the S/N threshold are removed,
    using a threshold appropriate for TESS as function of the number of
    data points.
    """
    snr_threshold = ut.signal_to_noise_threshold(n_points)
    # amplitude not significant enough
    a_insig_1 = (a_n / noise_level < snr_threshold)
    # red noise S/N limit
    # a_insig_2 = np.zeros(len(a_n), dtype=np.bool_)
    # remove = np.arange(len(a_n))[a_insig_1 | a_insig_2]
    remove = np.arange(len(a_n))[a_insig_1]
    return remove


@nb.njit()
def subtract_sines(a_n_1, ph_n_1, a_n_2, ph_n_2):
    """Analytically subtract a set of sine waves from another set
     with equal freguencies
     
    Parameters
    ----------
    a_n_1: numpy.ndarray[float]
        Amplitudes of the sinusoids of set 1
    ph_n_1: numpy.ndarray[float]
        Phases of the sinusoids of set 1
    a_n_2: numpy.ndarray[float]
        Amplitudes of the sinusoids of set 2
    ph_n_2: numpy.ndarray[float]
        Phases of the sinusoids of set 2
    
    Returns
    -------
    a_n_3: numpy.ndarray[float]
        Amplitudes of the resulting sinusoids
    ph_n_3: numpy.ndarray[float]
        Phases of the resulting sinusoids

    Notes
    -----
    Subtracts the sine waves in set 2 from the ones in set 1 by:
    Asin(wt+a) - Bsin(wt+b) = sqrt((Acos(a) - Bcos(b))^2 + (Asin(a) - Bsin(b))^2)
        * sin(wt + arctan((Asin(a) - Bsin(b))/(Acos(a) - Bcos(b)))
    """
    # duplicate terms
    cos_term = (a_n_1 * np.cos(ph_n_1) - a_n_2 * np.cos(ph_n_2))
    sin_term = (a_n_1 * np.sin(ph_n_1) - a_n_2 * np.sin(ph_n_2))
    # amplitude of new sine wave
    a_n_3 = np.sqrt(cos_term**2 + sin_term**2)
    # phase of new sine wave
    ph_n_3 = np.arctan2(sin_term, cos_term)
    return a_n_3, ph_n_3


@nb.njit()
def subtract_harmonic_sines(p_orb, f_n_1, a_n_1, ph_n_1, f_n_2, a_n_2, ph_n_2):
    """Analytically subtract a set of sine waves from another set
     both containing harmonics of p_orb

    Parameters
    ----------
    p_orb: float
        Orbital period of the eclipsing binary in days
    f_n_1: numpy.ndarray[float]
        Frequencies of the orbital harmonics, in per day
    a_n_1: numpy.ndarray[float]
        Corresponding amplitudes of the sinusoids
    ph_n_1: numpy.ndarray[float]
        Corresponding phases of the sinusoids
    f_n_2: numpy.ndarray[float]
        Frequencies of the orbital harmonics, in per day
    a_n_2: numpy.ndarray[float]
        Corresponding amplitudes of the sinusoids
    ph_n_2: numpy.ndarray[float]
        Corresponding phases of the sinusoids

    Returns
    -------
    f_n_3: numpy.ndarray[float]
        Frequencies of the orbital harmonics, in per day
    a_n_3: numpy.ndarray[float]
        Corresponding amplitudes of the sinusoids
    ph_n_3: numpy.ndarray[float]
        Corresponding phases of the sinusoids
    
    See Also
    --------
    subtract_sines
    """
    # find the harmonics in each set
    h_1 = (f_n_1 * p_orb - 0.5).astype(np.int_) + 1  # h_1 = np.round(f_n_1 * p_orb)
    h_2 = (f_n_2 * p_orb - 0.5).astype(np.int_) + 1  # h_2 = np.round(f_n_2 * p_orb)
    h_3 = np.unique(np.append(h_1, h_2))
    # organise in neat arrays following h_3
    a_n_1_full = np.array([a_n_1[h_1 == n][0] if n in h_1 else 0 for n in h_3])
    a_n_2_full = np.array([a_n_2[h_2 == n][0] if n in h_2 else 0 for n in h_3])
    ph_n_1_full = np.array([ph_n_1[h_1 == n][0] if n in h_1 else 0 for n in h_3])
    ph_n_2_full = np.array([ph_n_2[h_2 == n][0] if n in h_2 else 0 for n in h_3])
    # subtract
    a_n_3, ph_n_3 = subtract_sines(a_n_1_full, ph_n_1_full, a_n_2_full, ph_n_2_full)
    # frequency of the new sine waves
    f_n_3 = h_3 / p_orb
    # select only non-zero amplitudes
    non_zero = (a_n_3 > 0)
    f_n_3, a_n_3, ph_n_3 = f_n_3[non_zero], a_n_3[non_zero], ph_n_3[non_zero]
    return f_n_3, a_n_3, ph_n_3


@nb.njit()
def find_harmonics(f_n, f_n_err, p_orb, sigma=1.):
    """Find the orbital harmonics from a set of frequencies, given the orbital period.
    
    Parameters
    ----------
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    f_n_err: numpy.ndarray[float]
        Formal errors in the frequencies
    p_orb: float
        The orbital period
    sigma: float
        Number of times the error to use for check of significance
    
    Returns
    -------
    i_harmonic: numpy.ndarray[bool]
        Indices of frequencies that are harmonics of p_orb
    
    Notes
    -----
    Only includes those frequencies that are within sigma * error of an orbital harmonic.
    If multiple frequencies correspond to one harmonic, only the closest is kept.
    """
    # the frequencies divided by the orbital frequency gives integers for hamronics
    test_int = f_n * p_orb
    is_harmonic = ((test_int % 1) > 1 - sigma * f_n_err * p_orb) | ((test_int % 1) < sigma * f_n_err * p_orb)
    # look for harmonics that have multiple close frequencies
    harmonic_f = f_n[is_harmonic]
    sorter = np.argsort(harmonic_f)
    harmonic_n = np.round(test_int[is_harmonic], 0, np.zeros(np.sum(is_harmonic)))  # third arg needed for numba
    n_diff = np.diff(harmonic_n[sorter])
    # only keep the closest frequencies
    if np.any(n_diff == 0):
        n_dup = np.unique(harmonic_n[sorter][:-1][n_diff == 0])
        for n in n_dup:
            is_harmonic[np.round(test_int, 0, np.zeros(len(test_int))) == n] = False
            is_harmonic[np.argmin(np.abs(test_int - n))] = True
    i_harmonic = np.arange(len(f_n))[is_harmonic]
    return i_harmonic


@nb.njit()
def construct_harmonic_range(f_0, domain):
    """create a range of harmonic frequencies given the base frequency.
    
    Parameters
    ----------
    f_0: float
        Base frequency in the range, from where the rest of the pattern is built.
    domain: list[float], numpy.ndarray[float]
        Two values that give the borders of the range.
        Sensible values could be the Rayleigh criterion and the Nyquist frequency
    
    Returns
    -------
    harmonics: numpy.ndarray[float]
        Frequencies of the harmonic series in the domain
    n_range: numpy.ndarray[int]
        Corresponding harmonic numbers (base frequency is 1)
    """
    # determine where the range of harmonics starts and ends
    n_start = np.ceil(domain[0] / f_0)
    n_end = np.floor(domain[1] / f_0)
    n_range = np.arange(max(1, n_start), n_end + 1).astype(np.int_)
    harmonics = f_0 * n_range
    return harmonics, n_range


@nb.njit()
def find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9):
    """Get the indices of the frequencies matching closest to the harmonics.
    
    Parameters
    ----------
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    p_orb: float
        The orbital period
    f_tol: float
        Tolerance in the frequency for accepting harmonics
    
    Returns
    -------
     : numpy.ndarray[int]
        Indices of the frequencies in f_n that are deemed harmonics
     : numpy.ndarray[int]
        Corresponding harmonic numbers (base frequency is 1)
    
    Notes
    -----
    A frequency is only axcepted as harmonic if it is within 1e-9 of the pattern.
    This can now be user defined for more flexibility.
    """
    # make the pattern of harmonics
    domain = (0, np.max(f_n) + 0.5 / p_orb)
    harmonic_pattern, harmonic_n = construct_harmonic_range(1 / p_orb, domain)
    # sort the frequencies
    sorter = np.argsort(f_n)
    f_n = f_n[sorter]
    # get nearest neighbour in harmonics for each f_n by looking to the left and right of the sorted position
    i_nn = np.searchsorted(f_n, harmonic_pattern)
    i_nn[i_nn == len(f_n)] = len(f_n) - 1
    closest = np.abs(f_n[i_nn] - harmonic_pattern) < np.abs(harmonic_pattern - f_n[i_nn - 1])
    i_nn = i_nn * closest + (i_nn - 1) * np.invert(closest)
    # get the distances to nearest neighbours
    d_nn = np.abs(f_n[i_nn] - harmonic_pattern)
    # check that the closest neighbours are reasonably close to the harmonic
    m_cn = (d_nn < min(f_tol, 1 / (2 * p_orb)))  # distance must be smaller than tollerance
    return sorter[i_nn[m_cn]], harmonic_n[m_cn]


@nb.njit()
def find_harmonics_tolerance(f_n, p_orb, f_tol):
    """Get the indices of the frequencies matching within a tolerance to the harmonics.
    
    Parameters
    ----------
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    p_orb: float
        The orbital period
    f_tol: float
        Tolerance in the frequency for accepting harmonics
    
    Returns
    -------
     harmonics: numpy.ndarray[int]
        Indices of the frequencies in f_n that are deemed harmonics
     harmonic_n: numpy.ndarray[int]
        Corresponding harmonic numbers (base frequency is 1)
    
    Notes
    -----
    A frequency is only axcepted as harmonic if it is within a 5% relative error.
    This can now be user defined for more flexibility.
    """
    harmonic_n = np.zeros(len(f_n))
    harmonic_n = np.round(f_n * p_orb, 0, harmonic_n)  # closest harmonic (out argument needed in numba atm)
    harmonic_n[harmonic_n == 0] = 1  # avoid zeros resulting from large f_tol
    # get the distances to nearest pattern frequency
    d_nn = np.abs(f_n - harmonic_n / p_orb)
    # check that the closest neighbours are reasonably close to the harmonic
    m_cn = (d_nn < min(f_tol, 1 / (2 * p_orb)))  # distance smaller than tolerance (or half the harmonic spacing)
    # produce indices and make the right selection
    harmonics = np.arange(len(f_n))[m_cn]
    harmonic_n = harmonic_n[m_cn].astype(np.int_)
    return harmonics, harmonic_n


# @nb.njit()  # won't work due to itertools
def find_combinations(f_n, f_n_err, sigma=1.):
    """Find linear combinations from a set of frequencies.
    
    Parameters
    ----------
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    f_n_err: numpy.ndarray[float]
        Formal errors in the frequencies
    sigma: float
        Number of times the error to use for check of significance
    
    Returns
    -------
    final_o2: dict[int]
        Dictionary containing the indices of combinations of order 2
    final_o3: dict[int]
        Dictionary containing the indices of combinations of order 3

    Notes
    -----
    Only includes those frequencies that are within sigma * error of a linear combination.
    Does 2nd and 3rd order combinations. The number of sigma tolerance can be specified.
    """
    indices = np.arange(len(f_n))
    # make all combinations
    comb_order_2 = np.array(list(itt.combinations_with_replacement(indices, 2)))  # combinations of order 2
    comb_order_3 = np.array(list(itt.combinations_with_replacement(indices, 3)))  # combinations of order 2
    comb_freqs_o2 = np.sum(f_n[comb_order_2], axis=1)
    comb_freqs_o3 = np.sum(f_n[comb_order_3], axis=1)
    # check if any of the frequencies is a combination within error
    final_o2 = {}
    final_o3 = {}
    for i in indices:
        match_o2 = (f_n[i] > comb_freqs_o2 - sigma * f_n_err[i]) & (f_n[i] < comb_freqs_o2 + sigma * f_n_err[i])
        match_o3 = (f_n[i] > comb_freqs_o3 - sigma * f_n_err[i]) & (f_n[i] < comb_freqs_o3 + sigma * f_n_err[i])
        if np.any(match_o2):
            final_o2[i] = comb_order_2[match_o2]
        if np.any(match_o3):
            final_o3[i] = comb_order_3[match_o3]
    return final_o2, final_o3


def find_unknown_harmonics(f_n, f_n_err, sigma=1., n_max=5, f_tol=None):
    """Try to find harmonic series of unknown base frequency
    
    Parameters
    ----------
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    f_n_err: numpy.ndarray[float]
        Formal errors in the frequencies
    sigma: float
        Number of times the error to use for check of significance
    n_max: int
        Maximum divisor for each frequency in search of a base harmonic
    f_tol: None, float
        Tolerance in the frequency for accepting harmonics
        If None, use sigma matching instead of pattern matching
    
    Returns
    -------
    candidate_h: dict[int]
        Dictionary containing dictionaries with the indices of harmonic series
    
    Notes
    -----
    The first layer of the dictionary has the indices of frequencies as keys,
    the second layer uses n as keys and values are the indices of the harmonics
    stored in an array.
    n denotes the integer by which the frequency in question (index of the
    first layer) is divided to get the base frequency of the series.
    """
    indices = np.arange(len(f_n))
    n_harm = np.arange(1, n_max + 1)  # range of harmonic number that is tried for each frequency
    # test all frequencies for being the n-th harmonic in a series of harmonics
    candidate_h = {}
    for i in indices:
        for n in n_harm:
            p_base = n / f_n[i]  # 1/(f_n[i]/n)
            if f_tol is not None:
                i_harmonic, _ = find_harmonics_from_pattern(f_n, p_base, f_tol=f_tol)
            else:
                i_harmonic = find_harmonics(f_n, f_n_err, p_base, sigma=sigma)  # harmonic indices
            if (len(i_harmonic) > 1):
                # don't allow any gaps of more than 20 + the number of preceding harmonics
                set_i = np.arange(len(i_harmonic))
                set_sorter = np.argsort(f_n[i_harmonic])
                harm_n = np.rint(np.sort(f_n[i_harmonic][set_sorter]) / (f_n[i] / n))  # harmonic integer n
                large_gaps = (np.diff(harm_n) > 20 + set_i[:-1])
                if np.any(large_gaps):
                    cut_off = set_i[1:][large_gaps][0]
                    i_harmonic = i_harmonic[set_sorter][:cut_off]
                # only take sets that don't have frequencies lower than f_n[i]
                cond_1 = np.all(f_n[i_harmonic] >= f_n[i])
                cond_2 = (len(i_harmonic) > 1)  # check this again after cutting off gap
                if cond_1 & cond_2 & (i in candidate_h.keys()):
                    candidate_h[i][n] = i_harmonic
                elif cond_1 & cond_2:
                    candidate_h[i] = {n: i_harmonic}
    # check for conditions that only look at the set itself
    i_n_remove = []
    for i in candidate_h.keys():
        for n in candidate_h[i].keys():
            set_len = len(candidate_h[i][n])
            # determine the harmonic integer n of each frequency in the set
            harm_n = np.rint(np.sort(f_n[candidate_h[i][n]]) / (f_n[i] / n))
            # remove sets of two where n is larger than two
            cond_1 = (set_len == 2) & (n > 2)
            # remove sets of three where n is larger than three
            cond_2 = (set_len == 3) & (n > 3)
            # remove sets where all gaps are larger than three
            cond_3 = np.all(np.diff(harm_n) > 3)
            # remove sets with large gap between the first and second frequency
            cond_4 = (np.diff(harm_n)[0] > 7)
            # also remove any sets with n>1 that are not longer than the one with n=1
            if cond_1 | cond_2 | cond_3 | cond_4:
                if ([i, n] not in i_n_remove):
                    i_n_remove.append([i, n])
    # remove entries
    for i, n in i_n_remove:
        candidate_h[i].pop(n, None)
        if (len(candidate_h[i]) == 0):
            candidate_h.pop(i, None)
    # check whether a series is fully contained in another (and other criteria involving other sets)
    i_n_redundant = []
    for i in candidate_h.keys():
        for n in candidate_h[i].keys():
            # sets of keys to compare current (i, n) to
            compare = np.array([[j, k] for j in candidate_h.keys() for k in candidate_h[j].keys()
                                if (j != i) | (k != n)])
            # check whether this set is fully contained in another
            this_contained = np.array([np.all(np.in1d(candidate_h[i][n], candidate_h[j][k])) for j, k in compare])
            # check whether another set is fully contained in this one
            other_contained = np.array([np.all(np.in1d(candidate_h[j][k], candidate_h[i][n])) for j, k in compare])
            # check for equal length ones
            equal_length = np.array([len(candidate_h[i][n]) == len(candidate_h[j][k]) for j, k in compare])
            # those that are fully contained and same length are equal
            equal = (equal_length & this_contained)
            # remove equals from contained list
            this_contained = (this_contained & np.invert(equal))
            other_contained = (other_contained & np.invert(equal))
            # check for sets with the same starting f (or multiple)
            f_i, e_i = f_n[i] / n, f_n_err[i] / n  # the base frequency and corresponding error
            same_start = np.array([np.any((f_i + e_i * sigma > f_n[candidate_h[j][k]])
                                          & (f_i - e_i * sigma < f_n[candidate_h[j][k]])) for j, k in compare])
            # if this set is contained in another (larger) set and n is larger or equal, it is redundant
            for j, k in compare[this_contained]:
                if (k <= n) & ([i, n] not in i_n_redundant):
                    i_n_redundant.append([i, n])
            # if another set is contained in this (larger) one and n is larger, it is redundant
            for j, k in compare[other_contained]:
                if (k < n) & ([i, n] not in i_n_redundant):
                    i_n_redundant.append([i, n])
            # if this set is equal to another set but has higher n, it is redundant (we keep lowest n)
            for j, k in compare[equal]:
                if (k < n) & ([i, n] not in i_n_redundant):
                    i_n_redundant.append([i, n])
            # if this set starts with the same base frequency as another, but has higher n, it is redundant
            for j, k in compare[same_start]:
                if (k < n) & ([i, n] not in i_n_redundant):
                    i_n_redundant.append([i, n])
    # remove redundant entries
    for i, n in i_n_redundant:
        candidate_h[i].pop(n, None)
        if (len(candidate_h[i]) == 0):
            candidate_h.pop(i, None)
    return candidate_h


def base_harmonic_search(f_n, f_n_err, a_n, f_tol=None):
    """Test to find the base harmonic of a harmonic series within a set of frequencies
    
    Parameters
    ----------
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    f_n_err: numpy.ndarray[float]
        Formal errors in the frequencies
    a_n: list[float], numpy.ndarray[float]
        The amplitudes of a number of sine waves
    f_tol: None, float
        Tolerance in the frequency for accepting harmonics
        If None, use sigma matching instead of pattern matching
    
    Returns
    -------
    p_best: float
        Best fitting base period
    gof: dict[dict[float]]
        Goodness of fit metrics for all harmonic series investigated
        (sum of squared distances, sum of amplitudes, completeness of pattern)
    minimize: numpy.ndarray[float]
        Combined goodness-of-fit metric
    """
    # get the candidate harmonic series
    candidate_h = find_unknown_harmonics(f_n, f_n_err, sigma=3., n_max=4, f_tol=f_tol)
    # determine the best one (based on some goodness of fit parameters)
    max_aa = 0
    tot_sets = 0
    gof = {}
    for k in candidate_h.keys():
        gof[k] = {}
        for n in candidate_h[k].keys():
            harm_len = len(candidate_h[k][n])
            # calculate the frequency divided by the base harmonic to get to the harmonic number n
            harm_base = f_n[candidate_h[k][n]] / (f_n[k] / n)
            harm_n = np.round(harm_base)
            # determine three gof measures
            distance_measure = np.sum((harm_base - harm_n)**2) / harm_len
            completeness = harm_len / np.max(harm_n)
            gof[k][n] = [distance_measure, np.sum(a_n[candidate_h[k][n]]), completeness]
            # take the maximum the amplitudes to later normalise
            max_aa = max(max_aa, np.max(a_n[candidate_h[k][n]]))
            tot_sets += 1
    # calculate final qtt to be minimized
    minimize = np.zeros(tot_sets)
    base_freqs = np.zeros(tot_sets)
    i = 0
    for k in candidate_h.keys():
        for n in candidate_h[k].keys():
            minimize[i] = gof[k][n][1] / max_aa * gof[k][n][2]
            base_freqs[i] = f_n[k] / n
            i += 1
    # select the best set of harmonics and its base period
    p_best = 1 / base_freqs[np.argmax(minimize)]
    return p_best, gof, minimize


@nb.njit()
def base_harmonic_check(f_n, p_orb, t_tot, f_tol=None):
    """Test to find the base harmonic testing multiples of a test period
    
    Parameters
    ----------
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    p_orb: float
        Test period of the eclipsing binary in days
    t_tot: float
        Total time duration of time-series
    f_tol: None, float
        Tolerance in the frequency for accepting harmonics
        If None, use sigma matching instead of pattern matching
    
    Returns
    -------
    p_best: float
        Best fitting base period
    p_test: numpy.ndarray[float]
        Base periods tested
    optimise: numpy.ndarray[float]
        Optimised quantity, the square root of number of matching harmonics
        times the completeness of the found harmonic series
    """
    p_test = np.append(p_orb / np.arange(3, 1, -1), p_orb * np.arange(1, 4))
    p_test = p_test[p_test < t_tot]
    # determine the best one (based on some goodness of fit parameters)
    optimise = np.zeros(len(p_test))
    for i, p in enumerate(p_test):
        harmonics, harmonic_n = find_harmonics_from_pattern(f_n, p, f_tol=min(f_tol, 1 / (2 * p)))
        if (len(harmonics) > 0):
            optimise[i] = len(harmonics)**1.5 / np.max(harmonic_n)
    # select the best set of harmonics and its base period
    p_best = p_test[np.argmax(optimise)]
    return p_best, p_test, optimise


@nb.njit()
def measure_harmonic_period(f_n, f_n_err, p_orb, f_tol):
    """Performs a weighted average of the harmonics found in a set of frequencies.
    
    Parameters
    ----------
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    f_n_err: numpy.ndarray[float]
        Formal errors in the frequencies
    p_orb: float
        The orbital period
    f_tol: float
        Tolerance in the frequency for accepting harmonics
    
    Returns
    -------
    wavg_p_orb: float
        Weighted average orbital period
    werr_p_orb: float
        Error in the weighted average orbital period
    std_p_orb: float
        Standard deviation of the weighted average orbital period
    
    Notes
    -----
    The harmonics are determined using an initial period estimate that has to be
    reasonably close to the actual period.
    The orbital frequency is measured by taking a weighted average of the harmonics.
    The propagated error in the orbital frequency is calculated, and the measured
    standard deviation is also returned (as a separate handle on the error).
    All frequencies are converted to periods.
    """
    # extract the indices of the harmonics and the harmonic number
    harmonics, harmonic_n = find_harmonics_from_pattern(f_n, p_orb, f_tol=f_tol)
    if (len(harmonics) == 0):
        raise ValueError('No harmonic frequencies found')
    # transform the harmonic frequencies to the orbital frequency by dividing by n
    f_orb = f_n[harmonics] / harmonic_n
    f_orb_err = f_n_err[harmonics] / harmonic_n
    # do the weighted average and the error calculation
    n_harm = len(f_orb)
    wavg_f_orb = ut.weighted_mean(f_orb, 1 / f_orb_err**2)
    # wavg_f_orb = np.average(f_orb, weights=1 / f_orb_err**2)
    std_f_orb = np.std(f_orb)
    werr_f_orb = np.sqrt(np.sum(1 / (f_orb_err**2 * n_harm**2))) / np.sum(1 / (f_orb_err**2 * n_harm))
    # convert to periods
    wavg_p_orb = 1 / wavg_f_orb
    werr_p_orb = werr_f_orb / wavg_f_orb**2
    std_p_orb = std_f_orb / wavg_f_orb**2
    return wavg_p_orb, werr_p_orb, std_p_orb


@nb.njit(cache=True)
def curve_walker(signal, peaks, slope_sign, mode='up'):
    """Walk up or down a slope to approach zero or to reach an extremum.
    
    'peaks' are the starting points, 'signal' is the slope to walk
    mode = 'up': walk in the slope sign direction to reach a maximum (minus is left)
    mode = 'down': walk against the slope sign direction to reach a minimum (minus is right)
    mode = 'up_to_zero'/'down_to_zero': same as above, but approaching zero
        as closely as possible without changing direction.
    mode = 'zero': continue until the sign changes
    """
    if 'up' in mode:
        slope_sign = -slope_sign
    max_i = len(signal) - 1
    
    def check_edges(indices):
        return (indices > 0) & (indices < max_i)
    
    def check_condition(prev_signal, cur_signal):
        if 'up' in mode:
            condition = (prev_signal < cur_signal)
        elif 'down' in mode:
            condition = (prev_signal > cur_signal)
        else:
            condition = np.ones(len(cur_signal), dtype=np.bool_)
        if 'zero' in mode:
            condition &= (np.sign(prev_signal) == np.sign(cur_signal))
        return condition
    
    # start at the peaks
    prev_i = peaks
    prev_s = signal[prev_i]
    # step in the desired direction (checking the edges of the array)
    check_cur_edges = check_edges(prev_i + slope_sign)
    cur_i = prev_i + slope_sign * check_cur_edges
    cur_s = signal[cur_i]
    # check that we fulfill the condition (also check next points)
    check_cur_slope = check_condition(prev_s, cur_s)
    # combine the checks for the current indices
    check = (check_cur_slope & check_cur_edges)
    # define the indices to be optimized
    cur_i = prev_i + slope_sign * check
    while np.any(check):
        prev_i = cur_i
        prev_s = signal[prev_i]
        # step in the desired direction (checking the edges of the array)
        check_cur_edges = check_edges(prev_i + slope_sign)
        cur_i = prev_i + slope_sign * check_cur_edges
        cur_s = signal[cur_i]
        # and check that we fulfill the condition
        check_cur_slope = check_condition(prev_s, cur_s)
        check = (check_cur_slope & check_cur_edges)
        # finally, make the actual approved steps
        cur_i = prev_i + slope_sign * check
    return cur_i


@nb.njit()
def measure_harmonic_depths(f_h, a_h, ph_h, t_zero, t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2):
    """Measure the depths of the eclipses from the harmonic model given
    the timing measurements
    
    Parameters
    ----------
    f_h: numpy.ndarray[float]
        Frequencies of the orbital harmonics, in per day
    a_h: numpy.ndarray[float]
        Corresponding amplitudes of the sinusoids
    ph_h: numpy.ndarray[float]
        Corresponding phases of the sinusoids
    t_zero: float
        Time of deepest minimum modulo p_orb
    t_1: float
        Time of primary minimum in domain [0, p_orb)
    t_2: float
        Time of secondary minimum in domain [0, p_orb)
    t_1_1: float
        Time of primary first contact
    t_1_2: float
        Time of primary last contact
    t_2_1: float
        Time of secondary first contact
    t_2_2: float
        Time of secondary last contact
    
    Returns
    -------
    depth_1: float
        Depth of primary minimum
    depth_2: float
        Depth of secondary minimum
    """
    # calculate the harmonic model at the eclipse time points
    t_model = np.array([t_1_1, t_1, t_1_2, t_2_1, t_2, t_2_2])
    model_h = tsf.sum_sines(t_model + t_zero, f_h, a_h, ph_h)
    # calculate depths based on the average level at contacts and the minima
    depth_1 = (model_h[0] + model_h[2]) / 2 - model_h[1]
    depth_2 = (model_h[3] + model_h[5]) / 2 - model_h[4]
    return depth_1, depth_2


def measure_eclipses_dt(p_orb, f_h, a_h, ph_h, noise_level):
    """Determine the eclipse midpoints, depths and widths from the derivatives
    of the harmonic model.
    
    Parameters
    ----------
    p_orb: float
        Orbital period of the eclipsing binary in days
    f_h: numpy.ndarray[float]
        Frequencies containing the orbital harmonics, in per day
    a_h: numpy.ndarray[float]
        Corresponding amplitudes of the sinusoids
    ph_h: numpy.ndarray[float]
        Corresponding phases of the sinusoids
    noise_level: float
        The noise level (standard deviation of the residuals)
    
    Returns
    -------
    t_zero: float, None
        Time of deepest minimum modulo p_orb
    t_1: float, None
        Time of primary minimum in domain [0, p_orb)
        Shifted by t_zero so that deepest minimum occurs at 0
    t_2: float, None
        Time of secondary minimum in domain [0, p_orb)
        Shifted by t_zero so that deepest minimum occurs at 0
    t_contacts: tuple[float], None
        Measurements of the times of contact in the order:
        t_1_1, t_1_2, t_2_1, t_2_2
    depths: numpy.ndarray[float], None
        Measurements of the elcipse depths in units of a_n
    t_int_tan: tuple[float], None
        Measurements of the times near internal tangency in the order:
        t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2
    t_i_1_err: numpy.ndarray[float], None
        Measurement error estimates of the first contact(s)
    t_i_2_err: numpy.ndarray[float], None
        Measurement error estimates of the last contact(s)
    ecl_indices: numpy.ndarray[int], None
        Indices of several important points in the harmonic model
        as generated here (see function for details)
    
    Notes
    -----
    The result is ordered according to depth so that the deepest eclipse is the first.
    Timings are shifted by t_zero so that deepest minimum occurs at 0.
    """
    # make a timeframe from 0 to two P to catch both eclipses in full if present
    t_model = np.arange(0, 2 * p_orb + 0.00001, 0.00001)  # 0.864 second steps if we work in days and per day units
    model_h = tsf.sum_sines(t_model, f_h, a_h, ph_h)
    # the following code utilises a similar idea to find the eclipses as ECLIPSR (except waaay simpler)
    deriv_1 = tsf.sum_sines_deriv(t_model, f_h, a_h, ph_h, deriv=1)
    deriv_2 = tsf.sum_sines_deriv(t_model, f_h, a_h, ph_h, deriv=2)
    # find the first derivative peaks and select the 8 largest ones (those must belong to the four eclipses)
    peaks_1, props = sp.signal.find_peaks(np.abs(deriv_1), height=noise_level, prominence=noise_level)
    if (len(peaks_1) == 0):
        return (None,) * 9  # No eclipse signatures found above the noise level
    ecl_peaks = np.argsort(props['prominences'])[-8:]  # 8 or less (most prominent) peaks
    peaks_1 = np.sort(peaks_1[ecl_peaks])  # sort again to chronological order
    slope_sign = np.sign(deriv_1[peaks_1]).astype(int)  # sign reveals ingress or egress
    # walk outward from peaks_1 to zero in deriv_1
    zeros_1 = curve_walker(deriv_1, peaks_1, slope_sign, mode='zero')
    # find the minima in deriv_2 betweern peaks_1 and zeros_1
    peaks_2_n = [min(p1, z1) + np.argmin(deriv_2[min(p1, z1):max(p1, z1)]) for p1, z1 in zip(peaks_1, zeros_1)]
    peaks_2_n = np.array(peaks_2_n).astype(int)
    # walk outward from the minima in deriv_2 to (local) minima in deriv_1
    minimum_1 = curve_walker(np.abs(deriv_1), peaks_2_n, slope_sign, mode='down')
    # make all the consecutive combinations of peaks (as many as there are peaks)
    indices = np.arange(len(peaks_1))
    combinations = np.vstack(([[i, j] for i, j in zip(indices[:-1], indices[1:])], [indices[-1], 0]))
    # make eclipses
    ecl_indices = np.zeros((0, 11), dtype=int)
    for comb in combinations:
        # can only be an eclipse if ingress and then egress
        if (slope_sign[comb[0]] == -1) & (slope_sign[comb[1]] == 1):
            ecl = [zeros_1[comb[0]], minimum_1[comb[0]], peaks_2_n[comb[0]], peaks_1[comb[0]], 0, 0,
                   0, peaks_1[comb[1]], peaks_2_n[comb[1]], minimum_1[comb[1]], zeros_1[comb[1]]]
            # be careful of wrap-around
            if (ecl[2] > ecl[-3]):
                # check in the harmonic light curve model that all points in eclipse lie beneath the top points
                l_const, l_slope = tsf.linear_slope_two_points(t_model[ecl[2]], model_h[ecl[2]],
                                                               t_model[ecl[-3]] + p_orb, model_h[ecl[-3]])
                t_model_wrapped = np.append(t_model[ecl[2]:], t_model[:ecl[-3] + 1])
                model_h_wrapped = np.append(model_h[ecl[2]:], model_h[:ecl[-3] + 1])
                line = tsf.linear_curve(t_model_wrapped, np.array([l_const]), np.array([l_slope]),
                                        np.array([[0, len(t_model_wrapped)]]))
                ineq = (model_h_wrapped <= line)  # everything has to be below the line (or equal to it)
                # allclose takes into account floating point tolerance (np.all(ineq) does not)
                if np.allclose(model_h_wrapped[np.invert(ineq)], line[np.invert(ineq)]):
                    ecl_indices = np.vstack((ecl_indices, [ecl]))
            else:
                # check in the harmonic light curve model that all points in eclipse lie beneath the top points
                l_const, l_slope = tsf.linear_slope_two_points(t_model[ecl[2]], model_h[ecl[2]],
                                                               t_model[ecl[-3]], model_h[ecl[-3]])
                line = tsf.linear_curve(t_model[ecl[2]:ecl[-3]], np.array([l_const]), np.array([l_slope]),
                                        np.array([[0, ecl[-3] - ecl[2]]]))
                ineq = (model_h[ecl[2]:ecl[-3]] <= line)  # everything has to be below the line (or equal to it)
                # allclose takes into account floating point tolerance (np.all(ineq) does not)
                if np.allclose(model_h[ecl[2]:ecl[-3]][np.invert(ineq)], line[np.invert(ineq)]):
                    ecl_indices = np.vstack((ecl_indices, [ecl]))
    # determine the points of eclipse minima and flat bottoms
    indices_t = np.arange(len(t_model))
    for i, ecl in enumerate(ecl_indices):
        if (ecl[2] > ecl[-3]):  # wrap-around
            i_min = np.argmin(np.abs(np.append(deriv_1[ecl[2]:], deriv_1[:ecl[-3] + 1])))
            ecl_indices[i, 5] = np.append(indices_t[ecl[2]:], indices_t[:ecl[-3] + 1])[i_min]
        else:
            i_min = np.argmin(np.abs(deriv_1[ecl[2]:ecl[-3] + 1]))
            ecl_indices[i, 5] = indices_t[ecl[2]:ecl[-3] + 1][i_min]
        # left bottom turning point (peaks_2_p)
        if (ecl_indices[i, 5] < ecl[1]):  # wrap-around
            i_max = np.argmax(np.append(deriv_2[ecl[1]:], deriv_2[:ecl_indices[i, 5] + 1]))
            ecl_indices[i, 4] = np.append(indices_t[ecl[1]:], indices_t[:ecl_indices[i, 5] + 1])[i_max]
        else:
            i_max = np.argmax(deriv_2[ecl[1]:ecl_indices[i, 5] + 1])
            ecl_indices[i, 4] = indices_t[ecl[1]:ecl_indices[i, 5] + 1][i_max]
        # right bottom turning point (peaks_2_p)
        if (ecl_indices[i, 5] > ecl[-2]):  # wrap-around
            i_max = np.argmax(np.append(deriv_2[ecl_indices[i, 5]:], deriv_2[:ecl[-2] + 1]))
            ecl_indices[i, -5] = np.append(indices_t[ecl_indices[i, 5]:], indices_t[:ecl[-2] + 1])[i_max]
        else:
            i_max = np.argmax(deriv_2[ecl_indices[i, 5]:ecl[-2] + 1])
            ecl_indices[i, -5] = indices_t[ecl_indices[i, 5]:ecl[-2] + 1][i_max]
    # check that we have some eclipses
    if (len(ecl_indices) == 0) | (len(ecl_indices) == 1):
        return (None,) * 8 + (ecl_indices,)
    # correct for not quite reaching the top of the peak in some cases
    ecl_indices[:, 4] = curve_walker(deriv_2, ecl_indices[:, 4], -1, mode='up')
    ecl_indices[:, -5] = curve_walker(deriv_2, ecl_indices[:, -5], 1, mode='up')
    # make the timing measurement - make sure to account for wrap around when an eclipse is divided up
    t_m1_1 = t_model[ecl_indices[:, 1]]  # first times of minimum deriv_1 (from minimum_1)
    t_m1_2 = t_model[ecl_indices[:, -2]]  # last times of minimum deriv_1 (from minimum_1)
    t_p2n_1 = t_model[ecl_indices[:, 2]]  # first times of negative extremum deriv_2 (from peaks_2_n)
    t_p2n_2 = t_model[ecl_indices[:, -3]]  # last times of negative extremum deriv_2 (from peaks_2_n)
    # determine the eclipse edges from the midpoint between peaks_2_n and minimum_1
    t_i_1 = (t_m1_1 + t_p2n_1) / 2 * (t_p2n_1 >= t_m1_1) + (t_m1_1 + t_p2n_1 - 2 * p_orb) / 2 * (t_p2n_1 < t_m1_1)
    t_i_2 = (t_m1_2 + t_p2n_2) / 2 * (t_m1_2 >= t_p2n_2) + (t_m1_2 + t_p2n_2 - 2 * p_orb) / 2 * (t_m1_2 < t_p2n_2)
    indices_t_i_1 = np.searchsorted(t_model, t_i_1)  # if t_model is granular enough, this should be precise enough
    indices_t_i_2 = np.searchsorted(t_model, t_i_2)  # if t_model is granular enough, this should be precise enough
    # use the intervals as 3 sigma limits on either side
    t_i_1_err = (t_p2n_1 - t_m1_1) / 6 * (t_p2n_1 > t_m1_1) + (t_p2n_1 - t_m1_1 + 2 * p_orb) / 6 * (t_p2n_1 < t_m1_1)
    t_i_1_err[t_i_1_err == 0] = 0.00001  # avoid zeros
    t_i_2_err = (t_m1_2 - t_p2n_2) / 6 * (t_m1_2 > t_p2n_2) + (t_m1_2 - t_p2n_2 + 2 * p_orb) / 6 * (t_m1_2 < t_p2n_2)
    t_i_2_err[t_i_2_err == 0] = 0.00001  # avoid zeros
    # convert to midpoints and widths
    ecl_min = t_model[ecl_indices[:, 5]]
    ecl_mid = (t_i_1 + t_i_2) / 2 * (t_i_2 > t_i_1) + (t_i_1 + t_i_2 - 2 * p_orb) / 2 * (t_i_2 < t_i_1)
    widths = (t_i_2 - t_i_1) * (t_i_2 > t_i_1) + (t_i_2 - t_i_1 + 2 * p_orb) * (t_i_2 < t_i_1)
    depths = (model_h[indices_t_i_1] + model_h[indices_t_i_2]) / 2 - model_h[ecl_indices[:, 5]]
    # estimates of flat bottom
    t_b_i_1 = t_model[ecl_indices[:, 4]]
    t_b_i_2 = t_model[ecl_indices[:, -5]]
    ecl_mid_b = ((t_b_i_1 + t_b_i_2) / 2 * (t_b_i_2 > t_b_i_1)
                 + (t_b_i_1 + t_b_i_2 - 2 * p_orb) / 2 * (t_b_i_2 < t_b_i_1))
    widths_b = (t_b_i_2 - t_b_i_1) * (t_b_i_2 > t_b_i_1) + (t_b_i_2 - t_b_i_1 + 2 * p_orb) * (t_b_i_2 < t_b_i_1)
    # now pick out two consecutive, fully covered eclipses
    indices = np.arange(len(ecl_indices))
    combinations = np.array([[i, j] for i, j in zip(indices[:-1], indices[1:])])
    # check overlap of the eclipses
    comb_remove = []
    for i, comb in enumerate(combinations):
        c_dist = abs(abs(ecl_min[comb[0]] - ecl_min[comb[1]]) - p_orb)
        if (c_dist < max(widths[comb[0]] / 2, widths[comb[1]] / 2)):
            comb_remove.append(i)
    combinations = np.delete(combinations, comb_remove, axis=0)
    if (len(combinations) == 0):
        return (None,) * 8 + (ecl_indices,)
    # sum of depths should be largest for the most complete set of eclipses
    comb_d = depths[combinations[:, 0]] + depths[combinations[:, 1]]
    best_comb = combinations[np.argmax(comb_d)]  # argmax automatically picks the first in ties
    ecl_min = ecl_min[best_comb]
    ecl_mid = ecl_mid[best_comb]
    widths = widths[best_comb]
    depths = depths[best_comb]
    ecl_mid_b = ecl_mid_b[best_comb]
    widths_b = widths_b[best_comb]
    t_i_1_err = t_i_1_err[best_comb]
    t_i_2_err = t_i_2_err[best_comb]
    # put the primary (deepest) in front
    sorter = np.argsort(depths)[::-1]
    ecl_min = ecl_min[sorter]
    ecl_mid = ecl_mid[sorter]
    widths = widths[sorter]
    # depths = depths[sorter]  # no longer used from this point
    ecl_mid_b = ecl_mid_b[sorter]
    widths_b = widths_b[sorter]
    t_i_1_err = t_i_1_err[sorter]
    t_i_2_err = t_i_2_err[sorter]
    # put the deepest eclipse at zero (and make sure to get the edge timings in the right spot)
    t_zero = ecl_min[0]
    ecl_min = (ecl_min - t_zero) % p_orb
    ecl_mid = (ecl_mid - t_zero) % p_orb
    if ecl_mid[0] > (p_orb - widths[0] / 2):
        ecl_mid[0] = ecl_mid[0] - p_orb
    ecl_mid_b = (ecl_mid_b - t_zero) % p_orb
    if ecl_mid_b[0] > (p_orb - widths[0] / 2):
        ecl_mid_b[0] = ecl_mid_b[0] - p_orb
    t_zero = t_zero % p_orb
    # define in terms of time points
    t_1, t_2 = ecl_min[0], ecl_min[1]
    t_1_1 = ecl_mid[0] - (widths[0] / 2)  # time of primary first contact
    t_1_2 = ecl_mid[0] + (widths[0] / 2)  # time of primary last contact
    t_2_1 = ecl_mid[1] - (widths[1] / 2)  # time of secondary first contact
    t_2_2 = ecl_mid[1] + (widths[1] / 2)  # time of secondary last contact
    t_contacts = (t_1_1, t_1_2, t_2_1, t_2_2)
    t_b_1_1 = ecl_mid_b[0] - (widths_b[0] / 2)  # time of primary first internal tangency
    t_b_1_2 = ecl_mid_b[0] + (widths_b[0] / 2)  # time of primary last internal tangency
    t_b_2_1 = ecl_mid_b[1] - (widths_b[1] / 2)  # time of secondary first internal tangency
    t_b_2_2 = ecl_mid_b[1] + (widths_b[1] / 2)  # time of secondary last internal tangency
    t_int_tan = (t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2)
    # redetermine depths a tiny bit more precisely
    depths = measure_harmonic_depths(f_h, a_h, ph_h, t_zero, t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2)
    return t_zero, t_1, t_2, t_contacts, depths, t_int_tan, t_i_1_err, t_i_2_err, ecl_indices


@nb.njit()
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
    ν = π / 2 - ω + θ
    """
    nu = np.pi / 2 - w + theta
    return nu


@nb.njit()
def eccentric_anomaly(nu, e):
    """Eccentric anomaly in terms of true anomaly and eccentricity
    
    Parameters
    ----------
    nu: float, np.ndarray[float]
        True anomaly
    e: float, np.ndarray[float]
        Eccentricity of the orbit
    
    Returns
    -------
    : float, np.ndarray[float]
        Eccentric anomaly
    """
    return 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu/2), np.sqrt(1 + e) * np.cos(nu/2))


@nb.njit()
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
    Returns the quantity 2π(t2 - t1)/P given an eccentricity (e) and
    corresponding true anomaly values ν1 and ν2.
    The indefinite integral formula is:
    2 arctan(sqrt(1 - e)sin(nu/2) / (sqrt(1 + e)cos(nu/2))) - e sqrt(1 - e**2)sin(nu) / (1 + e cos(nu))
    """
    def indefinite_integral(nu, e):
        term_1 = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu/2), np.sqrt(1 + e) * np.cos(nu/2))
        term_2 = - e * np.sqrt(1 - e**2) * np.sin(nu) / (1 + e * np.cos(nu))
        return term_1 + term_2
    
    end_boundary = indefinite_integral(nu_2, e)
    start_boundary = indefinite_integral(nu_1, e)
    integral = end_boundary - start_boundary
    return integral


@nb.njit()
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
    For circular orbits, delta has minima at 0 and 180 degrees, but this will deviate for
    eccentric *and* inclined orbits due to conjunction no longer lining up with the minimum
    projected separation between the stars.
    
    Minimize this function w.r.t. theta near zero to get the phase angle of minimum separation
    at primary eclipse (eclipse maximum), or near pi to get it for the secondary eclipse.
    """
    sin_i_2 = np.sin(i)**2
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    # previous (identical except for a factor 1/2 which doesn't matter because it equals zero) formula, from Kopal 1959
    # term_1 = (1 - e * np.sin(theta - w)) * sin_i_2 * np.sin(2*theta)
    # term_2 = 2 * e * np.cos(theta - w) * (1 - np.cos(theta)**2 * sin_i_2)
    minimize = e * np.cos(w) * (1 - sin_i_2) * cos_t + sin_i_2 * cos_t * sin_t + e * np.sin(w) * sin_t
    return minimize


def minima_phase_angles(e, w, i):
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
    
    Notes
    -----
    If theta_2 equals zero, this means no solution was possible
    (no opposite signs), physically there was no minimum separation
    """
    try:
        opt_1 = sp.optimize.root_scalar(delta_deriv, args=(e, w, i), method='brentq', bracket=(-1, 1))
        theta_1 = opt_1.root
    except ValueError:
        theta_1 = 0
    try:
        opt_2 = sp.optimize.root_scalar(delta_deriv, args=(e, w, i), method='brentq', bracket=(np.pi-1, np.pi+1))
        theta_2 = opt_2.root
    except ValueError:
        theta_2 = 0
    return theta_1, theta_2


@nb.njit()
def first_contact_angle_1(phi, e, w, i, phi_0):
    """Find the root of this function to obtain the phase angle between first contact
    and eclipse minimum for the primary eclipse
    
    Parameters
    ----------
    phi: float, numpy.ndarray[float]
        Angle between first contact and eclipse minimum
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit
    phi_0: float
        Auxilary angle (see Kopal 1959)
    
    Returns
    -------
     : float, numpy.ndarray[float]
        Numeric result of the function that should equal 0
    """
    sin_i_2 = np.sin(i)**2
    term_1 = np.sqrt(1 - sin_i_2 * np.cos(phi)**2)
    term_2 = - np.sqrt(1 - sin_i_2 * np.cos(phi_0)**2) * (1 + e * np.sin(w + phi))
    return term_1 + term_2


@nb.njit()
def last_contact_angle_1(phi, e, w, i, phi_0):
    """Find the root of this function to obtain the phase angle between last contact
    and eclipse minimum for the primary eclipse
    
    Parameters
    ----------
    phi: float, numpy.ndarray[float]
        Angle between first contact and eclipse minimum
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit
    phi_0: float
        Auxilary angle (see Kopal 1959)
    
    Returns
    -------
     : float, numpy.ndarray[float]
        Numeric result of the function that should equal 0
    """
    sin_i_2 = np.sin(i)**2
    term_1 = np.sqrt(1 - sin_i_2 * np.cos(phi)**2)
    term_2 = - np.sqrt(1 - sin_i_2 * np.cos(phi_0)**2) * (1 + e * np.sin(w - phi))
    return term_1 + term_2


@nb.njit()
def first_contact_angle_2(phi, e, w, i, phi_0):
    """Find the root of this function to obtain the phase angle between first contact
    and eclipse minimum for the secondary eclipse
    
    Parameters
    ----------
    phi: float, numpy.ndarray[float]
        Angle between first contact and eclipse minimum
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit
    phi_0: float
        Auxilary angle (see Kopal 1959)
    
    Returns
    -------
     : float, numpy.ndarray[float]
        Numeric result of the function that should equal 0
    """
    sin_i_2 = np.sin(i)**2
    term_1 = np.sqrt(1 - sin_i_2 * np.cos(phi)**2)
    term_2 = - np.sqrt(1 - sin_i_2 * np.cos(phi_0)**2) * (1 - e * np.sin(w + phi))
    return term_1 + term_2


@nb.njit()
def last_contact_angle_2(phi, e, w, i, phi_0):
    """Find the root of this function to obtain the phase angle between last contact
    and eclipse minimum for the secondary eclipse
    
    Parameters
    ----------
    phi: float, numpy.ndarray[float]
        Angle between first contact and eclipse minimum
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit
    phi_0: float
        Auxilary angle (see Kopal 1959)
    
    Returns
    -------
     : float, numpy.ndarray[float]
        Numeric result of the function that should equal 0
    """
    sin_i_2 = np.sin(i)**2
    term_1 = np.sqrt(1 - sin_i_2 * np.cos(phi)**2)
    term_2 = - np.sqrt(1 - sin_i_2 * np.cos(phi_0)**2) * (1 - e * np.sin(w - phi))
    return term_1 + term_2


def contact_phase_angles(e, w, i, phi_0):
    """Determine the contact angles for given e, w, i, phi_0
    
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
        
    Returns
    -------
    phi_1_1: float
        First contact angle of primary eclipse
    phi_1_2: float
        Last contact angle of primary eclipse
    phi_2_1: float
        First contact angle of secondary eclipse
    phi_2_2: float
        Last contact angle of secondary eclipse
    """
    q1 = (-10**-5, np.pi/2)
    try:
        opt_3 = sp.optimize.root_scalar(first_contact_angle_1, args=(e, w, i, phi_0), method='brentq', bracket=q1)
        phi_1_1 = opt_3.root
    except ValueError:
        phi_1_1 = 0  # interval likely did not have different signs because it did not quite reach 0 at 0
    try:
        opt_4 = sp.optimize.root_scalar(last_contact_angle_1, args=(e, w, i, phi_0), method='brentq', bracket=q1)
        phi_1_2 = opt_4.root
    except ValueError:
        phi_1_2 = 0  # interval likely did not have different signs because it did not quite reach 0 at 0
    try:
        opt_5 = sp.optimize.root_scalar(first_contact_angle_2, args=(e, w, i, phi_0), method='brentq', bracket=q1)
        phi_2_1 = opt_5.root
    except ValueError:
        phi_2_1 = 0  # interval likely did not have different signs because it did not quite reach 0 at 0
    try:
        opt_6 = sp.optimize.root_scalar(last_contact_angle_2, args=(e, w, i, phi_0), method='brentq', bracket=q1)
        phi_2_2 = opt_6.root
    except ValueError:
        phi_2_2 = 0  # interval likely did not have different signs because it did not quite reach 0 at 0
    return phi_1_1, phi_1_2, phi_2_1, phi_2_2


@nb.njit()
def ecc_omega_approx(p_orb, t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2, i, phi_0):
    """Calculate the eccentricity and argument of periastron from the
    analytic formulae for small eccentricities (among other approximations).

    Parameters
    ----------
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_1: float, numpy.ndarray[float]
        Time of primary minimum in domain [0, p_orb)
        and t_1 < t_2
    t_2: float, numpy.ndarray[float]
        Time of secondary minimum in domain [0, p_orb)
    tau_1_1: float, numpy.ndarray[float]
        Duration of primary first contact to mimimum
    tau_1_2: float, numpy.ndarray[float]
        Duration of primary mimimum to last contact
    tau_2_1: float, numpy.ndarray[float]
        Duration of secondary first contact to mimimum
    tau_2_2: float, numpy.ndarray[float]
        Duration of secondary mimimum to last contact
    i: float
        Inclination of the orbit
    phi_0: float
        Auxilary angle (see Kopal 1959)

    Returns
    -------
    e: float, numpy.ndarray[float]
        Eccentricity for each set of imput parameters
    w: float, numpy.ndarray[float]
        Argument of periastron for each set of imput parameters
    """
    sin_i_2 = np.sin(i)**2
    cos_p0_2 = np.cos(phi_0)**2
    e_cos_w = np.pi * (t_2 / p_orb - t_1 / p_orb - 1/2) * (sin_i_2 / (1 + sin_i_2))
    e_sin_w = np.pi / (2 * np.sin(phi_0) * p_orb) * (tau_1_1 + tau_1_2 - tau_2_1 - tau_2_2)
    e_sin_w = e_sin_w * (sin_i_2 * cos_p0_2 / (1 - sin_i_2 * (1 + cos_p0_2)))
    e = np.sqrt(e_cos_w**2 + e_sin_w**2)
    w = np.arctan2(e_sin_w, e_cos_w) % (2 * np.pi)  # w in interval 0, 2pi
    return e, w


@nb.njit()
def radius_sum_from_phi0(e, i, phi_0):
    """Formula for the sum of radii in units of the semi-major axis
    from the angle phi_0
    
    Parameters
    ----------
    e: float, numpy.ndarray[float]
        Eccentricity
    i: float, numpy.ndarray[float]
        Inclination of the orbit
    phi_0: float, numpy.ndarray[float]
        Auxilary angle, see Kopal 1959
    
    Returns
    -------
    r_sum_sma: float, numpy.ndarray[float]
        Sum of radii in units of the semi-major axis
    """
    r_sum_sma = np.sqrt((1 - np.sin(i)**2 * np.cos(phi_0)**2)) * (1 - e**2)
    return r_sum_sma


@nb.njit()
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
    delta^2 = a^2 (1-e^2)^2(1 - sin^2(i)cos^2(theta))/(1 - e sin(theta - w))^2
    sep = delta/a
    """
    num = (1 - e**2)**2 * (1 - np.sin(i)**2 * np.cos(theta)**2)
    denom = (1 - e * np.sin(theta - w))**2
    sep = np.sqrt(num / denom)
    return sep


@nb.njit()
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
    elif (d <= 1.00001*abs(r_1 - r_2)):
        area = np.pi * min(r_1**2, r_2**2)
    else:
        area = 0
    return area


@nb.njit()
def sb_ratio_from_d_ratio(d_ratio, e, w, i, r_sum_sma, r_ratio, theta_1, theta_2):
    """Surface brightness ratio from the ratio of eclipse depths
    
    Parameters
    ----------
    d_ratio: float
        Measured depth ratio d_2 / d_1
        (d_2 < d_1 by definition)
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
    theta_1: float
        Phase angle of primary minimum
    theta_2: float
        Phase angle of secondary minimum
    
    Returns
    -------
    sb_ratio: float
        Surface brightness ratio sb_2/sb_1
    
    Notes
    -----
    In terms of the eclipse depths (d_i) and the areas
    covered in either eclipse (area_i):
    sb_ratio = (d_2 area_1)/(d_1 area_2)
    """
    sep_1 = projected_separation(e, w, i, theta_1)
    sep_2 = projected_separation(e, w, i, theta_2)
    r_1 = r_sum_sma / (1 + r_ratio)
    r_2 = r_sum_sma * r_ratio / (1 + r_ratio)
    area_1 = covered_area(sep_1, r_1, r_2)
    area_2 = covered_area(sep_2, r_1, r_2)
    if (area_2 > 0):
        sb_ratio = d_ratio * area_1 / area_2
    elif (area_1 > 0):
        sb_ratio = 1000  # we get into territory where the secondary is not visible
    else:
        sb_ratio = 1  # we get into territory where neither eclipse is visible
    return sb_ratio


@nb.njit()
def eclipse_depth(e, w, i, theta, r_sum_sma, r_ratio, sb_ratio):
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
    
    Returns
    -------
    light_lost: float
        Fractional loss of light at the given phase angle
    
    Notes
    -----
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
    # factor sb_ratio depends on primary or secondary, theta ~ 180 is secondary
    if (theta > np.pi/2) & (theta < 3*np.pi/2):
        light_lost = light_lost * sb_ratio
    return light_lost


def objective_inclination(i, r_ratio, p_orb, t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2, d_1, d_2,
                          t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err, d_1_err, d_2_err):
    """Minimise this function to obtain an inclination estimate
    
    Parameters
    ----------
    i: float
        Inclination of the orbit
    r_ratio: float
        Radius ratio r_2/r_1
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_1: float
        Time of primary minimum in domain [0, p_orb)
        and t_1 < t_2
    t_2: float
        Time of secondary minimum in domain [0, p_orb)
    tau_1_1: float
        Duration of primary first contact to mimimum
    tau_1_2: float
        Duration of primary mimimum to last contact
    tau_2_1: float
        Duration of secondary first contact to mimimum
    tau_2_2: float
        Duration of secondary mimimum to last contact
    d_1: float
        Depth of primary minimum
    d_2: float
        Depth of secondary minimum
    t_1_err: float
        Error in t_1 (the time of primary minimum)
    t_2_err: float
        Error in t_1 (the time of secondary minimum)
    tau_1_1_err: float
        Error in tau_1_1 (or in the time of primary first contact)
    tau_1_2_err: float
        Error in tau_1_2 (or in the time of primary last contact)
    tau_2_1_err: float
        Error in tau_2_1 (or in the time of secondary first contact)
    tau_2_2_err: float
        Error in tau_2_2 (or in the time of secondary last contact)
    d_1_err: float
        Error in the depth of primary minimum
    d_2_err: float
        Error in the depth of secondary minimum

    Returns
    -------
    sum_sq_dev: float
        Weighted sum of squared deviations of five outcomes of integrals
        of Kepler's second law for different time spans of the eclipses
        compared to the measured values.
    """
    # obtain phi_0 and the approximate e and w
    phi_0 = np.pi * (tau_1_1 + tau_1_2 + tau_2_1 + tau_2_2) / (2 * p_orb)
    e, w = ecc_omega_approx(p_orb, t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2, i, phi_0)
    # minimise for the phases of minima (theta)
    theta_1, theta_2 = minima_phase_angles(e, w, i)
    if (theta_2 == 0):
        # if no solution is possible (no opposite signs), return infinite
        return 10**9
    # minimise for the contact angles
    phi_1_1, phi_1_2, phi_2_1, phi_2_2 = contact_phase_angles(e, w, i, phi_0)
    # calculate the true anomaly of minima
    nu_1 = true_anomaly(theta_1, w)
    nu_2 = true_anomaly(theta_2, w)
    nu_conj_1 = true_anomaly(0, w)
    nu_conj_2 = true_anomaly(np.pi, w)
    # calculate the integrals (displacement, durations, asymmetries)
    n = 2 * np.pi / p_orb  # the average daily motion
    disp = integral_kepler_2(nu_1, nu_2, e) / n
    # phi angles are for contact points, and are measured from conjunction (theta = 0 or 180 deg)
    integral_tau_1_1 = integral_kepler_2(nu_conj_1 - phi_1_1, nu_conj_1, e) / n
    integral_tau_1_2 = integral_kepler_2(nu_conj_1, nu_conj_1 + phi_1_2, e) / n
    integral_tau_2_1 = integral_kepler_2(nu_conj_2 - phi_2_1, nu_conj_2, e) / n
    integral_tau_2_2 = integral_kepler_2(nu_conj_2, nu_conj_2 + phi_2_2, e) / n
    dur_1 = integral_tau_1_1 + integral_tau_1_2
    dur_2 = integral_tau_2_1 + integral_tau_2_2
    # calculate the depths
    r_sum_sma = radius_sum_from_phi0(e, i, phi_0)
    sb_ratio = sb_ratio_from_d_ratio((d_2/d_1), e, w, i, r_sum_sma, r_ratio, theta_1, theta_2)
    depth_1 = eclipse_depth(e, w, i, theta_1, r_sum_sma, r_ratio, sb_ratio)
    depth_2 = eclipse_depth(e, w, i, theta_2, r_sum_sma, r_ratio, sb_ratio)
    # displacement of the minima, linearly sensitive to e cos(w) (and sensitive to i)
    r_displacement = ((t_2 - t_1) - disp) / np.sqrt(t_1_err**2 + t_2_err**2)
    # difference in duration of the minima, linearly sensitive to e sin(w) (and sensitive to i and phi_0)
    tau_err_tot = np.sqrt(tau_1_1_err**2 + tau_1_2_err**2 + tau_2_1_err**2 + tau_2_2_err**2)
    r_duration_dif = ((tau_1_1 + tau_1_2 - tau_2_1 - tau_2_2) - (dur_1 - dur_2)) / tau_err_tot
    # sum of durations of the minima, linearly sensitive to phi_0 (and sensitive to e sin(w), i and phi_0)
    r_duration_sum = ((tau_1_1 + tau_1_2 + tau_2_1 + tau_2_2) - (dur_1 + dur_2)) / tau_err_tot
    # depths
    r_depth_1 = (d_1 - depth_1) / np.sqrt(d_1_err**2)
    r_depth_2 = (d_2 - depth_2) / np.sqrt(d_2_err**2)
    # calculate the error-normalised residuals
    resid = np.array([r_displacement, r_duration_dif, r_duration_sum, r_depth_1, r_depth_2])
    bic = tsf.calc_bic(resid, 1)
    return bic


def objective_ecl_param(params, p_orb, t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2, d_1, d_2, bottom_1,
                        bottom_2, t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err,
                        d_1_err, d_2_err):
    """Minimise this function to obtain an inclination estimate
    
    Parameters
    ----------
    params: array-like[float]
        i, r_ratio
        Inclination of the orbit
        Radius ratio r_2/r_1
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_1: float
        Time of primary minimum in domain [0, p_orb)
        and t_1 < t_2
    t_2: float
        Time of secondary minimum in domain [0, p_orb)
    tau_1_1: float
        Duration of primary first contact to mimimum
    tau_1_2: float
        Duration of primary mimimum to last contact
    tau_2_1: float
        Duration of secondary first contact to mimimum
    tau_2_2: float
        Duration of secondary mimimum to last contact
    d_1: float
        Depth of primary minimum
    d_2: float
        Depth of secondary minimum
    bottom_1: float
        Duration of 'flat' bottom of primary minimum
    bottom_2: float
        Duration of 'flat' bottom of secondary minimum
    t_1_err: float
        Error in t_1 (the time of primary minimum)
    t_2_err: float
        Error in t_1 (the time of secondary minimum)
    tau_1_1_err: float
        Error in tau_1_1 (or in the time of primary first contact)
    tau_1_2_err: float
        Error in tau_1_2 (or in the time of primary last contact)
    tau_2_1_err: float
        Error in tau_2_1 (or in the time of secondary first contact)
    tau_2_2_err: float
        Error in tau_2_2 (or in the time of secondary last contact)
    d_1_err: float
        Error in the depth of primary minimum
    d_2_err: float
        Error in the depth of secondary minimum

    Returns
    -------
    sum_sq_dev: float
        Weighted sum of squared deviations of five outcomes of integrals
        of Kepler's second law for different time spans of the eclipses
        compared to the measured values.
    """
    ecosw, esinw, i, phi_0, psi_0, r_ratio = params
    e = np.sqrt(ecosw**2 + esinw**2)
    w = np.arctan2(esinw, ecosw) % (2 * np.pi)
    # minimise for the phases of minima (theta)
    theta_1, theta_2 = minima_phase_angles(e, w, i)
    if (theta_2 == 0):
        # if no solution is possible (no opposite signs), return infinite
        return 10**9
    # minimise for the contact angles
    phi_1_1, phi_1_2, phi_2_1, phi_2_2 = contact_phase_angles(e, w, i, phi_0)
    # minimise for the internal tangency angles
    psi_1_1, psi_1_2, psi_2_1, psi_2_2 = contact_phase_angles(e, w, i, psi_0)
    # calculate the true anomaly of minima
    nu_1 = true_anomaly(theta_1, w)
    nu_2 = true_anomaly(theta_2, w)
    nu_conj_1 = true_anomaly(0, w)
    nu_conj_2 = true_anomaly(np.pi, w)
    # calculate the integrals (displacement, durations, asymmetries)
    n = 2 * np.pi / p_orb  # the average daily motion
    disp = integral_kepler_2(nu_1, nu_2, e) / n
    # phi angles are measured from conjunction (theta = 0 or 180 deg)
    integral_tau_1_1 = integral_kepler_2(nu_conj_1 - phi_1_1, nu_conj_1, e) / n
    integral_tau_1_2 = integral_kepler_2(nu_conj_1, nu_conj_1 + phi_1_2, e) / n
    integral_tau_2_1 = integral_kepler_2(nu_conj_2 - phi_2_1, nu_conj_2, e) / n
    integral_tau_2_2 = integral_kepler_2(nu_conj_2, nu_conj_2 + phi_2_2, e) / n
    dur_1 = integral_tau_1_1 + integral_tau_1_2
    dur_2 = integral_tau_2_1 + integral_tau_2_2
    # psi angles are for internal tangency
    integral_bottom_1_1 = integral_kepler_2(nu_conj_1 - psi_1_1, nu_conj_1, e) / n
    integral_bottom_1_2 = integral_kepler_2(nu_conj_1, nu_conj_1 + psi_1_2, e) / n
    integral_bottom_2_1 = integral_kepler_2(nu_conj_2 - psi_2_1, nu_conj_2, e) / n
    integral_bottom_2_2 = integral_kepler_2(nu_conj_2, nu_conj_2 + psi_2_2, e) / n
    bottom_dur_1 = integral_bottom_1_1 + integral_bottom_1_2
    bottom_dur_2 = integral_bottom_2_1 + integral_bottom_2_2
    # calculate the depths
    r_sum_sma = radius_sum_from_phi0(e, i, phi_0)
    sb_ratio = sb_ratio_from_d_ratio((d_2/d_1), e, w, i, r_sum_sma, r_ratio, theta_1, theta_2)
    depth_1 = eclipse_depth(e, w, i, theta_1, r_sum_sma, r_ratio, sb_ratio)
    depth_2 = eclipse_depth(e, w, i, theta_2, r_sum_sma, r_ratio, sb_ratio)
    # displacement of the minima, linearly sensitive to e cos(w) (and sensitive to i)
    r_displacement = ((t_2 - t_1) - disp) / np.sqrt(t_1_err**2 + t_2_err**2)
    # difference in duration of the minima, linearly sensitive to e sin(w) (and sensitive to i and phi_0)
    tau_err_tot = np.sqrt(tau_1_1_err**2 + tau_1_2_err**2 + tau_2_1_err**2 + tau_2_2_err**2)
    r_duration_dif = ((tau_1_1 + tau_1_2 - tau_2_1 - tau_2_2) - (dur_1 - dur_2)) / tau_err_tot
    # sum of durations of the minima, linearly sensitive to phi_0 (and sensitive to e sin(w), i and phi_0)
    r_duration_sum = ((tau_1_1 + tau_1_2 + tau_2_1 + tau_2_2) - (dur_1 + dur_2)) / tau_err_tot
    # depths
    r_depth_1 = (d_1 - depth_1) / np.sqrt(d_1_err**2)
    r_depth_2 = (d_2 - depth_2) / np.sqrt(d_2_err**2)
    # durations of the (flat) bottoms of the minima
    r_bottom_duration_1 = (bottom_1 - bottom_dur_1) / np.sqrt(tau_1_1_err**2 + tau_1_2_err**2)
    r_bottom_duration_2 = (bottom_2 - bottom_dur_2) / np.sqrt(tau_2_1_err**2 + tau_2_2_err**2)
    # calculate the error-normalised residuals
    resid = np.array([r_displacement, r_duration_dif, r_duration_sum, r_depth_1, r_depth_2,
                      r_bottom_duration_1, r_bottom_duration_2])
    bic = tsf.calc_bic(resid, 6)
    return bic


def eclipse_parameters(p_orb, timings_tau, depths, bottom_dur, timing_errs, depths_err):
    """Determine all eclipse parameters using a combination of approximate
    functions and fitting procedures
    
    Parameters
    ----------
    p_orb: float
        The orbital period
    timings_tau: numpy.ndarray[float]
        Eclipse timings of minima and durations from/to first/last contact,
        t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2
    depths: numpy.ndarray[float]
        Eclipse depth of the primary and secondary, depth_1, depth_2
    bottom_dur: numpy.ndarray[float]
        Durations of the possible flat bottom (internal tangency)
        t_b_1_2 - t_b_1_1, t_b_2_2 - t_b_2_1
    timing_errs: numpy.ndarray[float]
        Error estimates for the eclipse timings,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err, or
        t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err
    depths_err: numpy.ndarray[float]
        Error estimates for the depths
    
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
    """
    # use mix of approximate and exact formulae iteratively to get a value for i
    r_ratio = 1
    args_i = (r_ratio, p_orb, *timings_tau, depths[0], depths[1], *timing_errs, *depths_err)
    bounds_i = (np.pi/4, np.pi/2)
    res = sp.optimize.minimize_scalar(objective_inclination, args=args_i, method='bounded', bounds=bounds_i)
    i = res.x
    # calculation phi_0, in durations: (duration_1 + duration_2)/4 = (2pi/P)(tau_1_1 + tau_1_2 + tau_2_1 + tau_2_2)/4
    phi_0 = np.pi * (timings_tau[2] + timings_tau[3] + timings_tau[4] + timings_tau[5]) / (2 * p_orb)
    # psi_0 is like phi_0 but for the eclipse bottom
    psi_0 = np.pi * (bottom_dur[0] + bottom_dur[1]) / (2 * p_orb)
    # values of e and w by approximate formulae
    e, w = ecc_omega_approx(p_orb, *timings_tau, i, phi_0)
    # value for r_sum_sma from ecc, incl and phi_0
    r_sum_sma = radius_sum_from_phi0(e, i, phi_0)
    # value for |r1 - r2|/a = r_dif_sma from ecc, incl and psi_0
    r_dif_sma = radius_sum_from_phi0(e, i, psi_0)
    # r_dif_sma only valid if psi_0 is not zero, otherwise it will give limits on the radii
    r_small = (r_sum_sma - r_dif_sma)/2  # if psi_0=0, this is a lower limit on the smaller radius
    r_large = (r_sum_sma + r_dif_sma)/2  # if psi_0=0, this is an upper limit on the bigger radius
    # fit for e, w, i, phi_0, psi_0 and r_ratio
    ecosw, esinw = e * np.cos(w), e * np.sin(w)
    par_init = (ecosw, esinw, i, phi_0, psi_0, 1)
    args_ep = (p_orb, *timings_tau, *depths, *bottom_dur, *timing_errs, *depths_err)
    bounds_ep = ((max(ecosw - 0.1, -1), min(ecosw + 0.1, 1)), (max(esinw - 0.1, -1), min(esinw + 0.1, 1)),
                 (i - 0.08, min(i + 0.08, np.pi/2)), (phi_0/1.1, min(2*phi_0, np.pi/2)),
                 (psi_0/1.1, min(2*psi_0, np.pi/2)), (r_small/r_large/1.1, r_large/r_small*1.1))  # 0.08 rad ~ 4.5 deg
    res = sp.optimize.minimize(objective_ecl_param, par_init, args=args_ep, method='nelder-mead', bounds=bounds_ep)
    ecosw, esinw, i, phi_0, psi_0, r_ratio = res.x
    e = np.sqrt(ecosw**2 + esinw**2)
    w = np.arctan2(esinw, ecosw) % (2 * np.pi)
    # update radii
    r_sum_sma = radius_sum_from_phi0(e, i, phi_0)
    r_dif_sma = radius_sum_from_phi0(e, i, psi_0)
    # value for sb_ratio from the depths ratio and the other parameters
    theta_1, theta_2 = minima_phase_angles(e, w, i)
    sb_ratio = sb_ratio_from_d_ratio(depths[1]/depths[0], e, w, i, r_sum_sma, r_ratio, theta_1, theta_2)
    return e, w, i, phi_0, psi_0, r_sum_sma, r_dif_sma, r_ratio, sb_ratio


@nb.njit()
def formal_uncertainties(e, w, i, phi_0, p_orb, t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2, p_err, i_err,
                         t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err):
    """Calculates the uncorrelated (formal) uncertainties for the extracted
    parameters (e, w, phi_0, r_sum_sma).
    
    Parameters
    ----------
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit
    phi_0: float
        Auxilary angle, see Kopal 1959
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_1: float
        Time of primary minimum in domain [0, p_orb)
        and t_1 < t_2
    t_2: float
        Time of secondary minimum in domain [0, p_orb)
    tau_1_1: float
        Duration of primary first contact to mimimum
    tau_1_2: float
        Duration of primary mimimum to last contact
    tau_2_1: float
        Duration of secondary first contact to mimimum
    tau_2_2: float
        Duration of secondary mimimum to last contact
    p_err: float
        Error in the orbital period (used as error in the times of minima)
    i_err: float
        Error in the orbital inclination
    t_1_err: float
        Error in t_1 (the time of primary minimum)
    t_2_err: float
        Error in t_1 (the time of secondary minimum)
    tau_1_1_err: float
        Error in tau_1_1 (or in the time of primary first contact)
    tau_1_2_err: float
        Error in tau_1_2 (or in the time of primary last contact)
    tau_2_1_err: float
        Error in tau_2_1 (or in the time of secondary first contact)
    tau_2_2_err: float
        Error in tau_2_2 (or in the time of secondary last contact)

    Returns
    -------
    sigma_e: float
        Formal error in eccentricity
    sigma_w: float
        Formal error in argument of periastron
    sigma_phi_0: float
        Formal error in auxilary angle
    sigma_r_sum_sma: float
        Formal error in sum of radii in units of the semi-major axis
    sigma_ecosw: float
        Formal error in e*cos(w)
    sigma_esinw: float
        Formal error in e*sin(w)
    sigma_f_c: float
        Formal error in sqrt(e)*cos(w)
    sigma_f_s: float
        Formal error in sqrt(e)*sin(w)
    """
    # often used errors
    p_err_2 = p_err**2
    sum_tau_err_2 = tau_1_1_err**2 + tau_1_2_err**2 + tau_2_1_err**2 + tau_2_2_err**2
    # often used sin, cos
    sin_i = np.sin(i)
    sin_i_2 = sin_i**2
    cos_i = np.cos(i)
    cos_i_2 = cos_i**2
    sin_phi0 = np.sin(phi_0)
    cos_phi0_2 = np.cos(phi_0)**2
    cos_w = np.cos(w)
    sin_w = np.sin(w)
    # other often used terms
    term_i = 1 + sin_i_2  # term with i in it
    term_1_i_phi0 = 1 - sin_i_2 * (1 + cos_phi0_2)  # term with i and phi_0 in it
    e_2 = e**2
    term_2_i_phi0 = (1 - sin_i_2 * cos_phi0_2)  # another term with i and phi_0 in it
    # error in phi_0
    s_phi0_p = (np.pi * (tau_1_1 + tau_1_2 + tau_2_1 + tau_2_2) / (2 * p_orb**2))**2 * p_err_2
    s_phi0_tau = (np.pi / (2 * p_orb))**2 * sum_tau_err_2
    sigma_phi_0 = np.sqrt(s_phi0_p + s_phi0_tau)
    # error in e*cos(w)
    s_ecosw_p = (np.pi * (t_2 - t_1) / p_orb**2 * sin_i_2 / term_i)**2 * p_err_2
    s_ecosw_t = (np.pi / p_orb * sin_i_2 / term_i)**2 * (t_1_err**2 + t_2_err**2)
    s_ecosw_i = (2 * sin_i * cos_i / term_i**2)**2 * i_err**2
    sigma_ecosw = np.sqrt(s_ecosw_p + s_ecosw_t + s_ecosw_i)
    # error in e*sin(w)
    s_esinw_p = (np.pi * (tau_1_1 + tau_1_2 - tau_2_1 - tau_2_2) / (2 * sin_phi0 * p_orb**2)
                 * sin_i_2 * cos_phi0_2 / term_1_i_phi0)**2 * p_err_2
    s_esinw_tau = (np.pi / (2 * sin_phi0 * p_orb) * sin_i_2 * cos_phi0_2 / term_1_i_phi0)**2 * sum_tau_err_2
    s_esinw_i = (2 * sin_i * cos_i * cos_phi0_2 / term_1_i_phi0**2)**2 * i_err**2
    s_esinw_phi0 = (2 * sin_i_2 * (1 - sin_i_2) * sin_phi0 * np.cos(phi_0) / term_1_i_phi0**2)**2 * sigma_phi_0**2
    sigma_esinw = np.sqrt(s_esinw_p + s_esinw_tau + s_esinw_i + s_esinw_phi0)
    # error in e
    sigma_e = np.sqrt(cos_w**2 * sigma_ecosw**2 + sin_w**2 * sigma_esinw**2)
    # error in w
    sigma_w = np.sqrt(sin_w**2 / e_2 * sigma_ecosw**2 + cos_w**2 / e_2 * sigma_esinw**2)
    # error in (r1+r2)/a
    s_rs_e = 4 * e_2 * term_2_i_phi0 * sigma_e**2
    s_rs_i = sin_i_2 * cos_i_2 * cos_phi0_2**2 * (1 - e_2)**2 / term_2_i_phi0 * i_err**2
    s_rs_phi0 = sin_i_2**2 * cos_i_2 * cos_phi0_2 * (1 - e_2)**2 / term_2_i_phi0 * sigma_phi_0**2
    sigma_r_sum_sma = np.sqrt(s_rs_e + s_rs_i + s_rs_phi0)
    # error in f_c and f_s (sqrt(e)cos(w) and sqrt(e)sin(w))
    sigma_f_c = np.sqrt(cos_w**2 / (4 * e) * sigma_e**2 + e * sin_w**2 * sigma_w**2)
    sigma_f_s = np.sqrt(sin_w**2 / (4 * e) * sigma_e**2 + e * cos_w**2 * sigma_w**2)
    return sigma_e, sigma_w, sigma_phi_0, sigma_r_sum_sma, sigma_ecosw, sigma_esinw, sigma_f_c, sigma_f_s


def error_estimates_hdi(e, w, i, phi_0, psi_0, r_sum_sma, r_dif_sma, r_ratio, sb_ratio, p_orb, t_zero, f_h, a_h, ph_h,
                        timings_tau, bottom_dur, timing_errs, depths_err, verbose=False):
    """Estimate errors using the highest density interval (HDI)
    
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
    p_orb: float
        Orbital period of the eclipsing binary in days
    f_h: numpy.ndarray[float]
        The frequencies of a number of harmonic sine waves
    a_h: numpy.ndarray[float]
        The amplitudes of a number of harmonic sine waves
    ph_h: numpy.ndarray[float]
        The phases of a number of harmonic sine waves
    t_zero: float
        Time of deepest minimum modulo p_orb
    timings_tau: numpy.ndarray[float]
        Eclipse timings of minima and durations from/to first/last contact,
        t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2
    bottom_dur: numpy.ndarray[float]
        Durations of the possible flat bottom (internal tangency)
        t_b_1_2 - t_b_1_1, t_b_2_2 - t_b_2_1
    timing_errs: numpy.ndarray[float]
        Error estimates for the eclipse timings,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err, or
        t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err
    depths_err: numpy.ndarray[float]
        Error estimates for the depths
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    intervals: tuple[numpy.ndarray[float]]
        The HDIs (hdi_prob=0.683) for the parameters:
        e, w, i, phi_0, psi_0, r_sum_sma, r_dif_sma, r_ratio,
        sb_ratio, e*cos(w), e*sin(w), f_c, f_s
    bounds: tuple[numpy.ndarray[float]]
        The HDIs (hdi_prob=0.997) for the same parameters as intervals
    errors: tuple[numpy.ndarray[float]]
        The (non-symmetric) errors for the same parameters as intervals.
        Derived from the intervals
    dists_in: tuple[numpy.ndarray[float]]
        Full input distributions for: t_1, t_2,
        tau_1_1, tau_1_2, tau_2_1, tau_2_2, d_1, d_2, bot_1, bot_2
    dists_out: tuple[numpy.ndarray[float]]
        Full output distributions for the same parameters as intervals
    
    Notes
    -----
    The HDI is the minimum width Bayesian credible interval (BCI).
    https://arviz-devs.github.io/arviz/api/generated/arviz.hdi.html
    The interval for w can consist of two disjunct intervals due to the
    degeneracy between angles around 90 degrees and 270 degrees.
    """
    t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2 = timings_tau
    bottom_1, bottom_2 = bottom_dur[0], bottom_dur[1]
    t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err = timing_errs
    bot_1_err = np.sqrt(tau_1_1_err**2 + tau_1_2_err**2)  # estimate from the tau errors
    bot_2_err = np.sqrt(tau_2_1_err**2 + tau_2_2_err**2)  # estimate from the tau errors
    # generate input distributions
    rng = np.random.default_rng()
    n_gen = 10**4
    normal_t_1 = rng.normal(t_1, t_1_err, n_gen)
    normal_t_2 = rng.normal(t_2, t_2_err, n_gen)
    # use truncated normals for the rest of the measurements
    normal_tau_1_1 = sp.stats.truncnorm.rvs((0 - tau_1_1) / tau_1_1_err, 10, loc=tau_1_1, scale=tau_1_1_err, size=n_gen)
    normal_tau_1_2 = sp.stats.truncnorm.rvs((0 - tau_1_2) / tau_1_2_err, 10, loc=tau_1_2, scale=tau_1_2_err, size=n_gen)
    normal_tau_2_1 = sp.stats.truncnorm.rvs((0 - tau_2_1) / tau_2_1_err, 10, loc=tau_2_1, scale=tau_2_1_err, size=n_gen)
    normal_tau_2_2 = sp.stats.truncnorm.rvs((0 - tau_2_2) / tau_2_2_err, 10, loc=tau_2_2, scale=tau_2_2_err, size=n_gen)
    normal_bot_1 = sp.stats.truncnorm.rvs((0 - bottom_1) / bot_1_err, 10, loc=bottom_1, scale=bot_1_err, size=n_gen)
    normal_bot_2 = sp.stats.truncnorm.rvs((0 - bottom_2) / bot_2_err, 10, loc=bottom_2, scale=bot_2_err, size=n_gen)
    normal_d_1 = np.zeros(n_gen)
    normal_d_2 = np.zeros(n_gen)
    for k in range(n_gen):
        # depths are re-calculated from the above inputs
        contacts = (normal_t_1[k] - normal_tau_1_1[k], normal_t_1[k] + normal_tau_1_2[k],
                    normal_t_2[k] - normal_tau_2_1[k], normal_t_2[k] + normal_tau_2_2[k])
        depths_k = measure_harmonic_depths(f_h, a_h, ph_h, t_zero, normal_t_1[k], normal_t_2[k], contacts[0],
                                           contacts[1], contacts[2], contacts[3])
        normal_d_1[k] = depths_k[0]
        normal_d_2[k] = depths_k[1]
    # determine the output distributions
    e_vals = np.zeros(n_gen)
    w_vals = np.zeros(n_gen)
    i_vals = np.zeros(n_gen)
    phi0_vals = np.zeros(n_gen)
    psi0_vals = np.zeros(n_gen)
    rsumsma_vals = np.zeros(n_gen)
    rdifsma_vals = np.zeros(n_gen)
    rratio_vals = np.zeros(n_gen)
    sbratio_vals = np.zeros(n_gen)
    i_delete = []  # to be deleted due to out of bounds parameter
    for k in range(n_gen):
        timings_tau_dist = (normal_t_1[k], normal_t_2[k],
                            normal_tau_1_1[k], normal_tau_1_2[k], normal_tau_2_1[k], normal_tau_2_2[k])
        # if sum of tau happens to be larger than p_orb, skip and delete
        if (np.sum(timings_tau_dist[2:]) > p_orb):
            i_delete.append(k)
            continue
        depths_k = np.array([normal_d_1[k], normal_d_2[k]])
        bottom_dur_dist = np.array([normal_bot_1[k], normal_bot_2[k]])
        out = eclipse_parameters(p_orb, timings_tau_dist, depths_k, bottom_dur_dist, timing_errs, depths_err)
        e_vals[k] = out[0]
        w_vals[k] = out[1]
        i_vals[k] = out[2]
        phi0_vals[k] = out[3]
        psi0_vals[k] = out[4]
        rsumsma_vals[k] = out[5]
        rdifsma_vals[k] = out[6]
        rratio_vals[k] = out[7]
        sbratio_vals[k] = out[8]
        if verbose & (k % 100 == 0):
            print(f'parameter calculations {int(k / (n_gen) * 100)}% done')
    # delete the skipped parameters
    normal_t_1 = np.delete(normal_t_1, i_delete)
    normal_t_2 = np.delete(normal_t_2, i_delete)
    normal_tau_1_1 = np.delete(normal_tau_1_1, i_delete)
    normal_tau_1_2 = np.delete(normal_tau_1_2, i_delete)
    normal_tau_2_1 = np.delete(normal_tau_2_1, i_delete)
    normal_tau_2_2 = np.delete(normal_tau_2_2, i_delete)
    normal_d_1 = np.delete(normal_d_1, i_delete)
    normal_d_2 = np.delete(normal_d_2, i_delete)
    normal_bot_1 = np.delete(normal_bot_1, i_delete)
    normal_bot_2 = np.delete(normal_bot_2, i_delete)
    e_vals = np.delete(e_vals, i_delete)
    w_vals = np.delete(w_vals, i_delete)
    i_vals = np.delete(i_vals, i_delete)
    phi0_vals = np.delete(phi0_vals, i_delete)
    psi0_vals = np.delete(psi0_vals, i_delete)
    rsumsma_vals = np.delete(rsumsma_vals, i_delete)
    rdifsma_vals = np.delete(rdifsma_vals, i_delete)
    rratio_vals = np.delete(rratio_vals, i_delete)
    sbratio_vals = np.delete(sbratio_vals, i_delete)
    # Calculate the highest density interval (HDI) for a given probability.
    cos_w = np.cos(w)
    sin_w = np.sin(w)
    # inclination
    i_interval = arviz.hdi(i_vals, hdi_prob=0.683)
    i_bounds = arviz.hdi(i_vals, hdi_prob=0.997)
    i_errs = np.array([i - i_interval[0], i_interval[1] - i])
    # phi_0
    phi0_interval = arviz.hdi(phi0_vals, hdi_prob=0.683)
    phi0_bounds = arviz.hdi(phi0_vals, hdi_prob=0.997)
    phi0_errs = np.array([phi_0 - phi0_interval[0], phi0_interval[1] - phi_0])
    # psi_0
    psi0_interval = arviz.hdi(psi0_vals, hdi_prob=0.683)
    psi0_bounds = arviz.hdi(psi0_vals, hdi_prob=0.997)
    psi0_errs = np.array([psi_0 - psi0_interval[0], psi0_interval[1] - psi_0])
    # eccentricity
    e_interval = arviz.hdi(e_vals, hdi_prob=0.683)
    e_bounds = arviz.hdi(e_vals, hdi_prob=0.997)
    e_errs = np.array([e - e_interval[0], e_interval[1] - e])
    # e*np.cos(w)
    ecosw_interval = arviz.hdi(e_vals*np.cos(w_vals), hdi_prob=0.683)
    ecosw_bounds = arviz.hdi(e_vals*np.cos(w_vals), hdi_prob=0.997)
    ecosw_errs = np.array([e*cos_w - ecosw_interval[0], ecosw_interval[1] - e*cos_w])
    # e*np.sin(w)
    esinw_interval = arviz.hdi(e_vals*np.sin(w_vals), hdi_prob=0.683)
    esinw_bounds = arviz.hdi(e_vals*np.sin(w_vals), hdi_prob=0.997)
    esinw_errs = np.array([e*sin_w - esinw_interval[0], esinw_interval[1] - e*sin_w])
    # sqrt(e)*np.cos(w) (== f_c)
    f_c_interval = arviz.hdi(np.sqrt(e_vals)*np.cos(w_vals), hdi_prob=0.683)
    f_c_bounds = arviz.hdi(np.sqrt(e_vals)*np.cos(w_vals), hdi_prob=0.997)
    f_c_errs = np.array([np.sqrt(e)*cos_w - f_c_interval[0], f_c_interval[1] - np.sqrt(e)*cos_w])
    # sqrt(e)*np.sin(w) (== f_s)
    f_s_interval = arviz.hdi(np.sqrt(e_vals)*np.sin(w_vals), hdi_prob=0.683)
    f_s_bounds = arviz.hdi(np.sqrt(e_vals)*np.sin(w_vals), hdi_prob=0.997)
    f_s_errs = np.array([np.sqrt(e)*sin_w - f_s_interval[0], f_s_interval[1] - np.sqrt(e)*sin_w])
    # omega
    if (abs(w/np.pi*180 - 180) > 80) & (abs(w/np.pi*180 - 180) < 100):
        w_interval = arviz.hdi(w_vals, hdi_prob=0.683, multimodal=True)
        w_bounds = arviz.hdi(w_vals, hdi_prob=0.997, multimodal=True)
        if (len(w_interval) > 1):
            # sign of (w - w_interval) only changes if w is in the interval
            w_in_interval = (np.sign((w - w_interval)[:, 0] * (w - w_interval)[:, 1]) == -1)
            w_errs = np.array([w - w_interval[w_in_interval, 0], w_interval[w_in_interval, 1] - w])
        else:
            w_errs = np.array([w - w_interval[0, 0], w_interval[0, 1] - w])
    else:
        w_interval = arviz.hdi(w_vals - np.pi, hdi_prob=0.683, circular=True) + np.pi
        w_bounds = arviz.hdi(w_vals - np.pi, hdi_prob=0.997, circular=True) + np.pi
        w_errs = np.array([min(abs(w - w_interval[0]), abs(2*np.pi + w - w_interval[0])),
                           min(abs(w_interval[1] - w), abs(2*np.pi + w_interval[1] - w))])
    # r_sum_sma
    rsumsma_interval = arviz.hdi(rsumsma_vals, hdi_prob=0.683)
    rsumsma_bounds = arviz.hdi(rsumsma_vals, hdi_prob=0.997)
    rsumsma_errs = np.array([r_sum_sma - rsumsma_interval[0], rsumsma_interval[1] - r_sum_sma])
    # r_dif_sma
    rdifsma_interval = arviz.hdi(rdifsma_vals, hdi_prob=0.683)
    rdifsma_bounds = arviz.hdi(rdifsma_vals, hdi_prob=0.997)
    rdifsma_errs = np.array([r_dif_sma - rdifsma_interval[0], rdifsma_interval[1] - r_dif_sma])
    # r_ratio
    rratio_interval = arviz.hdi(rratio_vals, hdi_prob=0.683)
    rratio_bounds = arviz.hdi(rratio_vals, hdi_prob=0.997)
    rratio_errs = np.array([r_ratio - rratio_interval[0], rratio_interval[1] - r_ratio])
    # sb_ratio
    sbratio_interval = arviz.hdi(sbratio_vals, hdi_prob=0.683)
    sbratio_bounds = arviz.hdi(sbratio_vals, hdi_prob=0.997)
    sbratio_errs = np.array([sb_ratio - sbratio_interval[0], sbratio_interval[1] - sb_ratio])
    # collect
    intervals = (e_interval, w_interval, i_interval, phi0_interval, psi0_interval, rsumsma_interval, rdifsma_interval,
                 rratio_interval, sbratio_interval, ecosw_interval, esinw_interval, f_c_interval, f_s_interval)
    bounds = (e_bounds, w_bounds, i_bounds, phi0_bounds, psi0_bounds, rsumsma_bounds, rdifsma_bounds, rratio_bounds,
              sbratio_bounds, ecosw_bounds, esinw_bounds, f_c_bounds, f_s_bounds)
    errors = (e_errs, w_errs, i_errs, phi0_errs, psi0_errs, rsumsma_errs, rdifsma_errs, rratio_errs,
              sbratio_errs, ecosw_errs, esinw_errs, f_c_errs, f_s_errs)
    dists_in = (normal_t_1, normal_t_2, normal_tau_1_1, normal_tau_1_2, normal_tau_2_1, normal_tau_2_2,
                normal_d_1, normal_d_2, normal_bot_1, normal_bot_2)
    dists_out = (e_vals, w_vals, i_vals, phi0_vals, psi0_vals, rsumsma_vals, rdifsma_vals, rratio_vals, sbratio_vals)
    return intervals, bounds, errors, dists_in, dists_out
