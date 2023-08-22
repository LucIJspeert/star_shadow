"""STAR SHADOW
Satellite Time series Analysis Routine using
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

import arviz as az

from . import timeseries_functions as tsf
from . import utility as ut


@nb.njit(cache=True)
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
        # all the frequencies are close
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


@nb.njit(cache=True)
def chains_within_rayleigh(f_n, rayleigh):
    """Find all chains of frequencies within each other's Rayleigh criterion.
    
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


@nb.njit(cache=True)
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


@nb.njit(cache=True)
def remove_insignificant_snr(a_n, noise_at_f, n_points):
    """Removes insufficiently significant frequencies in terms of S/N.
    
    Parameters
    ----------
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    noise_at_f: numpy.ndarray[float]
        The noise level at each frequency
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
    
    The noise_at_f here captures the amount of noise on fitting a
    sinusoid of a certain frequency to all data points.
    Not to be confused with the noise on the individual data points of the
    time series.
    """
    snr_threshold = ut.signal_to_noise_threshold(n_points)
    # signal-to-noise below threshold
    a_insig_1 = (a_n / noise_at_f < snr_threshold)
    remove = np.arange(len(a_n))[a_insig_1]
    return remove


@nb.njit(cache=True)
def subtract_sines(a_n_1, ph_n_1, a_n_2, ph_n_2):
    """Analytically subtract a set of sine waves from another set
     with equal frequencies
     
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


@nb.njit(cache=True)
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


@nb.njit(cache=True)
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
    # the frequencies divided by the orbital frequency gives integers for harmonics
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


@nb.njit(cache=True)
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


@nb.njit(cache=True)
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
    harmonics: numpy.ndarray[int]
        Indices of the frequencies in f_n that are deemed harmonics
    harmonic_n: numpy.ndarray[int]
        Corresponding harmonic numbers (base frequency is 1)
    
    Notes
    -----
    A frequency is only accepted as harmonic if it is within 1e-9 of the pattern
    (by default). This can now be user defined for more flexibility.
    """
    # guard against zero period or empty list
    if (p_orb == 0) | (len(f_n) == 0):
        harmonics = np.zeros(0, dtype=np.int_)
        harmonic_n = np.zeros(0, dtype=np.int_)
        return harmonics, harmonic_n
        
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
    m_cn = (d_nn < min(f_tol, 1 / (2 * p_orb)))  # distance must be smaller than tolerance (and never larger than 1/2P)
    # keep the ones left over
    harmonics = sorter[i_nn[m_cn]]
    harmonic_n = harmonic_n[m_cn]
    return harmonics, harmonic_n


@nb.njit(cache=True)
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
    A frequency is only accepted as harmonic if it is within some relative error.
    This can be user defined for flexibility.
    """
    harmonic_n = np.zeros(len(f_n))
    harmonic_n = np.round(f_n * p_orb, 0, harmonic_n)  # closest harmonic (out argument needed in numba atm)
    harmonic_n[harmonic_n == 0] = 1  # avoid zeros resulting from large f_tol
    # get the distances to the nearest pattern frequency
    d_nn = np.abs(f_n - harmonic_n / p_orb)
    # check that the closest neighbours are reasonably close to the harmonic
    m_cn = (d_nn < min(f_tol, 1 / (2 * p_orb)))  # distance smaller than tolerance (or half the harmonic spacing)
    # produce indices and make the right selection
    harmonics = np.arange(len(f_n))[m_cn]
    harmonic_n = harmonic_n[m_cn].astype(np.int_)
    return harmonics, harmonic_n


@nb.njit(cache=True)
def select_harmonics_sigma(f_n, f_n_err, p_orb, f_tol, sigma_f=3):
    """Selects only those frequencies that are probably harmonics

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
    sigma_f: float
        Number of times the error to use for check of significant
        frequency separation
    
    Returns
    -------
     harmonics_passed: numpy.ndarray[int]
        Indices of the frequencies in f_n that are harmonics
     harmonic_n: numpy.ndarray[int]
        Corresponding harmonic numbers (base frequency is 1)

    Notes
    -----
    A frequency is only accepted as harmonic if it is within the
    frequency resolution of the pattern, and if it is within <sigma_f> sigma
    of the frequency uncertainty
    """
    # get the harmonics within f_tol
    harmonics, harm_n = find_harmonics_from_pattern(f_n, p_orb, f_tol=f_tol)
    # frequency error overlaps with theoretical harmonic
    passed_h = np.zeros(len(harmonics), dtype=np.bool_)
    for i, (h, n) in enumerate(zip(harmonics, harm_n)):
        f_theo = n / p_orb
        margin = sigma_f * f_n_err[h]
        overlap_h = (f_n[h] + margin > f_theo) & (f_n[h] - margin < f_theo)
        if overlap_h:
            passed_h[i] = True
    harmonics_passed = harmonics[passed_h]
    harmonic_n = harm_n[passed_h]
    return harmonics_passed, harmonic_n


# @nb.njit()  # won't work due to itertools
def find_combinations(f_n, f_n_err, sigma=1.):
    """Find linear combinations from a set of frequencies.
    
    Parameters
    ----------
    f_n: list[float], numpy.ndarray[float]
        The frequencies of a number of sine waves
    f_n_err: numpy.ndarray[float]
        Formal errors on the frequencies
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
        Formal errors on the frequencies
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


@nb.njit(cache=True)
def harmonic_series_length(f_test, f_n, freq_res, f_nyquist):
    """Find the number of harmonics that a set of frequencies has
    
    Parameters
    ----------
    f_test: numpy.ndarray[float]
        Frequencies to test at
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    freq_res: float
        Frequency resolution
    f_nyquist: float
        Nyquist frequency
    
    Returns
    -------
    n_harm: numpy.ndarray[float]
        Number of harmonics per pattern
    completeness: numpy.ndarray[float]
        Completeness factor of each pattern
    distance: numpy.ndarray[float]
        Sum of squared distances between harmonics
    """
    n_harm = np.zeros(len(f_test))
    completeness = np.zeros(len(f_test))
    distance = np.zeros(len(f_test))
    for i, f in enumerate(f_test):
        harmonics, harmonic_n = find_harmonics_from_pattern(f_n, 1 / f, f_tol=freq_res / 2)
        n_harm[i] = len(harmonics)
        if (n_harm[i] == 0):
            completeness[i] = 1
            distance[i] = 0
        else:
            completeness[i] = n_harm[i] / (f_nyquist // f)
            distance[i] = np.sum((f_n[harmonics] - harmonic_n * f)**2)
    return n_harm, completeness, distance


@nb.njit(cache=True)
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

    Parameters
    ----------
    signal: numpy.ndarray[float]
        The curve to walk along
    peaks: numpy.ndarray[float]
        The starting points
    slope_sign: numpy.ndarray[float]
        Sign of the slope of the curve at the peak locations
    mode: str
        mode='up': walk in the slope sign direction to reach a
            minimum (minus is left)
        mode='down': walk against the slope sign direction to reach
            a maximum (minus is right)
        mode='up_to_zero'/'down_to_zero': same as above, but approaching zero
            as closely as possible without changing direction.
        mode='zero': continue until the sign changes

    Returns
    -------
    cur_i: numpy.ndarray[float]
        End positions of all the walkers

    Notes
    -----
    Assumes a circular curve, so that it can walk from one end
    back onto the other end.
    """
    if 'down' in mode:
        steps = -slope_sign
    else:
        steps = slope_sign
    len_s = len(signal)
    
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
    # step in the desired direction
    cur_i = (prev_i + steps)
    cur_s = signal[cur_i]
    # check that we fulfill the condition
    check = check_condition(prev_s, cur_s) & (cur_i != -1) & (cur_i != len_s)
    # define the indices to be optimized
    cur_i = (prev_i + steps * check)
    while np.any(check):
        prev_i = cur_i
        prev_s = signal[prev_i]
        # step in the desired direction
        cur_i = (prev_i + steps)
        cur_s = signal[cur_i]
        # and check that we fulfill the condition
        check = check_condition(prev_s, cur_s) & (cur_i != -1) & (cur_i != len_s)
        # finally, make the actual approved steps
        cur_i = (prev_i + steps * check)
    return cur_i


@nb.njit(cache=True)
def curve_walker_circular(signal, peaks, slope_sign, mode='up'):
    """Walk up or down a slope to approach zero or to reach an extremum.
    
    Parameters
    ----------
    signal: numpy.ndarray[float]
        The curve to walk along
    peaks: numpy.ndarray[float]
        The starting points
    slope_sign: numpy.ndarray[float]
        Sign of the slope of the curve at the peak locations
    mode: str
        mode='up': walk in the slope sign direction to reach a
            minimum (minus is left)
        mode='down': walk against the slope sign direction to reach
            a maximum (minus is right)
        mode='up_to_zero'/'down_to_zero': same as above, but approaching zero
            as closely as possible without changing direction.
        mode='zero': continue until the sign changes
    
    Returns
    -------
    cur_i: numpy.ndarray[float]
        End positions of all the walkers
    
    Notes
    -----
    Assumes a circular curve, so that it can walk from one end
    back onto the other end.
    """
    if 'down' in mode:
        steps = -slope_sign
    else:
        steps = slope_sign
    len_s = len(signal)
    
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
    # step in the desired direction
    cur_i = (prev_i + steps) % len_s
    cur_s = signal[cur_i]
    # check that we fulfill the condition
    check = check_condition(prev_s, cur_s)
    # define the indices to be optimized
    cur_i = (prev_i + steps * check) % len_s
    while np.any(check):
        prev_i = cur_i
        prev_s = signal[prev_i]
        # step in the desired direction
        cur_i = (prev_i + steps) % len_s
        cur_s = signal[cur_i]
        # and check that we fulfill the condition
        check = check_condition(prev_s, cur_s)
        # finally, make the actual approved steps
        cur_i = (prev_i + steps * check) % len_s
    return cur_i


@nb.njit(cache=True)
def curve_explorer_root_angle(func, x0, walk_sign, args):
    """Walk up or down a slope to approach zero for angles

    Parameters
    ----------
    func: function
        Function that generates the curve to walk along.
    x0: numpy.ndarray[float]
        The starting positions
    walk_sign: numpy.ndarray[float]
        Sign of the direction to walk in (minus is left)
    args: tuple
        Arguments for func

    Returns
    -------
    x_interp: numpy.ndarray[float]
        Root positions for each x0

    Notes
    -----
    Assumes a circular curve with a period of 2 pi, so that it
    can walk from one end back onto the other end.
    """
    two_pi = 2 * np.pi
    step = 0.001  # step in rad (does not determine final precision)
    # start at x0
    cur_x = x0
    cur_y = func(x0, *args)
    f_sign_x0 = np.sign(cur_y).astype(np.int_)  # sign of function at initial position
    # step in the desired direction
    try_x = (x0 + step * walk_sign)
    try_y = func(try_x, *args)
    # check whether the sign stays the same
    check = (np.sign(cur_y) == np.sign(try_y))
    # if we take this many steps, we've gone full circle
    for _ in range(two_pi // step + 1):
        if not np.any(check):
            break
        # make the approved steps and continue if any were approved
        cur_x[check] = try_x[check]
        cur_y[check] = try_y[check]
        # try the next steps
        try_x[check] = cur_x[check] + step * walk_sign[check]
        try_y[check] = func(try_x[check], *args)
        # check whether the sign stays the same
        check[check] = (np.sign(cur_y[check]) == np.sign(try_y[check]))
    # non-wrapped try positions
    try_x = (cur_x + step * walk_sign)
    try_y = func(try_x, *args)
    # interpolate for better precision than the angle step
    condition = (f_sign_x0 == 1)
    xp1 = np.where(condition, try_y, cur_y)
    yp1 = np.where(condition, try_x, cur_x)
    xp2 = np.where(condition, cur_y, try_y)
    yp2 = np.where(condition, cur_x, try_x)
    x_interp = ut.interp_two_points(np.zeros(len(x0)), xp1, yp1, xp2, yp2)
    x_interp = x_interp % two_pi
    return x_interp


@nb.njit(cache=True)
def measure_harmonic_depths(f_h, a_h, ph_h, t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2):
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
    t_1: float
        Time of primary minimum with respect to the mean time
    t_2: float
        Time of secondary minimum with respect to t_1
    t_1_1: float
        Time of primary first contact
    t_1_2: float
        Time of primary last contact
    t_2_1: float
        Time of secondary first contact
    t_2_2: float
        Time of secondary last contact
    t_b_1_1: float
        Time of primary first internal tangency
    t_b_1_2: float
        Time of primary last internal tangency
    t_b_2_1: float
        Time of secondary first internal tangency
    t_b_2_2: float
        Time of secondary last internal tangency
    
    Returns
    -------
    depth_1: float
        Depth of primary minimum
    depth_2: float
        Depth of secondary minimum
    """
    # measure heights at the bottom
    if (t_b_1_2 - t_b_1_1 > 0):
        t_model_b_1 = np.linspace(t_b_1_1, t_b_1_2, 1000)
    else:
        t_model_b_1 = np.array([t_1])
    if (t_b_2_2 - t_b_2_1 > 0):
        t_model_b_2 = np.linspace(t_b_2_1, t_b_2_2, 1000)
    else:
        t_model_b_2 = np.array([t_2])
    model_h_b_1 = tsf.sum_sines(t_model_b_1, f_h, a_h, ph_h, t_shift=False)
    model_h_b_2 = tsf.sum_sines(t_model_b_2, f_h, a_h, ph_h, t_shift=False)
    # calculate the harmonic model at the eclipse edges
    t_model = np.array([t_1_1, t_1_2, t_2_1, t_2_2])
    model_h = tsf.sum_sines(t_model, f_h, a_h, ph_h, t_shift=False)
    # calculate depths based on the average level at contacts and the minima
    depth_1 = (model_h[0] + model_h[1]) / 2 - np.mean(model_h_b_1)
    depth_2 = (model_h[2] + model_h[3]) / 2 - np.mean(model_h_b_2)
    return depth_1, depth_2


def mark_eclipse_peaks(t_model, deriv_1, deriv_2, noise_level, t_gaps, n_prominent=24):
    """Find and refine the prominent eclipse signatures
    
    Parameters
    ----------
    t_model: numpy.ndarray[float]
        Set of time points for a model of sinudoids
    deriv_1: numpy.ndarray[float]
        Derivative of the sinusoids at t_model
    deriv_2: numpy.ndarray[float]
        Second derivative of the sinusoids at t_model
    noise_level: float
        The noise level (standard deviation of the residuals)
    t_gaps: numpy.ndarray[float]
        Gap timestamps in pairs
    n_prominent: int
        Number of most prominent peaks to initially mark
        Advised to leave at 24 except for local search then 4.

    Returns
    -------
    peaks_1: numpy.ndarray[int]
        Set of indices indicating extrema in deriv_1
    slope_sign: numpy.ndarray[int]
        Sign of deriv_1 at peaks_1 (the sign of the slope)
    zeros_1: numpy.ndarray[int]
        Set of indices indicating zero points in deriv_1
    peaks_2_n: numpy.ndarray[int]
        Set of indices indicating minima in deriv_2
    minimum_1: numpy.ndarray[int]
        Set of indices indicating local minima in deriv_1
    zeros_1_in: numpy.ndarray[int]
        Set of indices indicating inner zero points in deriv_1
    peaks_2_p: numpy.ndarray[int]
        Set of indices indicating maxima in deriv_2
    minimum_1_in: numpy.ndarray[int]
        Set of indices indicating inner local minima in deriv_1
    
    Notes
    -----
    Intended for use in detect_eclipses,
    with a fine grid of time points spanning
    two times the orbital period.
    """
    # find the first derivative peaks and select the 8 largest ones (those must belong to the four eclipses)
    peaks_1, props = sp.signal.find_peaks(np.abs(deriv_1), height=noise_level, prominence=noise_level)
    if (len(peaks_1) == 0):
        return (None,) * 8  # No eclipse signatures found above the noise level
    # 24 or fewer (most prominent) peaks (increased from 8 as side peaks of the primary were higher than the secondary)
    ecl_peaks = np.argsort(props['prominences'])[-n_prominent:]
    # increase the amount if any peaks fell into gaps
    mask_gaps = tsf.mask_timestamps(t_model, t_gaps)  # masks everything but the gaps
    mask_peaks = np.zeros(len(t_model), dtype=bool)
    mask_peaks[peaks_1[ecl_peaks]] = True  # mask everything but the eclipse peaks
    n_pk_in_gap = np.sum(mask_gaps & mask_peaks)  # number of peaks falling in gaps
    if (n_pk_in_gap > 0):
        ecl_peaks = np.argsort(props['prominences'])[-n_prominent - n_pk_in_gap:]
    # now convert ecl_peaks to peaks in t_model and get slope signs
    peaks_1 = np.sort(peaks_1[ecl_peaks])  # sort again to chronological order
    slope_sign = np.sign(deriv_1[peaks_1]).astype(int)  # sign reveals ingress or egress
    # walk outward from peaks_1 to zero in deriv_1
    zeros_1 = curve_walker(deriv_1, peaks_1, slope_sign, mode='zero')
    # find the minima in deriv_2 between peaks_1 and zeros_1
    peaks_2_n = [min(p1, z1) + np.argmin(deriv_2[min(p1, z1):max(p1, z1)]) for p1, z1 in zip(peaks_1, zeros_1)]
    peaks_2_n = np.array(peaks_2_n).astype(int)
    # adjust slightly to account for any misalignment with peaks_1
    peaks_2_n = curve_walker(deriv_2, peaks_2_n, -slope_sign, mode='down')
    # walk outward from the minima in deriv_2 to (local) minima in deriv_1
    minimum_1 = curve_walker(np.abs(deriv_1), peaks_2_n, -slope_sign, mode='down')
    # walk inward from peaks_1 to zero in deriv_1
    zeros_1_in = curve_walker(deriv_1, peaks_1, -slope_sign, mode='zero')
    # find the maxima in deriv_2 between peaks_1 and zeros_1_in
    peaks_2_p = [min(p1, z1) + np.argmax(deriv_2[min(p1, z1):max(p1, z1)]) for p1, z1 in zip(peaks_1, zeros_1_in)]
    peaks_2_p = np.array(peaks_2_p).astype(int)
    # adjust slightly to account for any misalignment with peaks_1
    peaks_2_p = curve_walker(deriv_2, peaks_2_p, -slope_sign, mode='up')
    # walk inward from the maxima in deriv_2 to (local) minima in deriv_1
    minimum_1_in = curve_walker(np.abs(deriv_1), peaks_2_p, slope_sign, mode='down')
    return peaks_1, slope_sign, zeros_1, peaks_2_n, minimum_1, zeros_1_in, peaks_2_p, minimum_1_in


@nb.njit(cache=True)
def assemble_eclipses(p_orb, t_model, deriv_1, deriv_2, model_h, t_gaps, peaks_1, slope_sign, zeros_1, peaks_2_n,
                      minimum_1, zeros_1_in, peaks_2_p, minimum_1_in):
    """Make a list of indices indicating eclipses out of lists of peaks
    
    Parameters
    ----------
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_model: numpy.ndarray[float]
        Set of time points for a model of sinudoids
    deriv_1: numpy.ndarray[float]
        Derivative of the sinusoids at t_model
    deriv_2: numpy.ndarray[float]
        Second derivative of the sinusoids at t_model
    model_h: numpy.ndarray[float]
        Model of harmonic sinusoids at t_model
    t_gaps: numpy.ndarray[float]
        Gap timestamps in pairs
    peaks_1: numpy.ndarray[int]
        Set of indices indicating extrema in deriv_1
    slope_sign: numpy.ndarray[int]
        Sign of deriv_1 at peaks_1 (the sign of the slope)
    zeros_1: numpy.ndarray[int]
        Set of indices indicating zero points in deriv_1
    peaks_2_n: numpy.ndarray[int]
        Set of indices indicating minima in deriv_2
    minimum_1: numpy.ndarray[int]
        Set of indices indicating local minima in deriv_1
    zeros_1_in: numpy.ndarray[int]
        Set of indices indicating inner zero points in deriv_1
    peaks_2_p: numpy.ndarray[int]
        Set of indices indicating maxima in deriv_2
    minimum_1_in: numpy.ndarray[int]
        Set of indices indicating inner local minima in deriv_1

    Returns
    -------
    ecl_indices: numpy.ndarray[int]
        Two dimensional array of eclipse indices.
        Each eclipse has indices corresponding to
        several prominent points.
    
    Notes
    -----
    Intended for use in detect_eclipses,
    in conjunction with mark_eclipse_peaks.
    """
    # group peaks into pairs with negative and then positive slopes
    indices = np.arange(len(peaks_1))
    i_neg = indices[slope_sign == -1]  # negative slope is candidate ingress
    i_pos = indices[slope_sign == 1]  # positive slope is candidate egress
    combinations = np.zeros((0, 2), dtype=np.int_)
    for i in i_neg:
        egress = i_pos[i_pos > i]  # select all points after i
        ingress = np.repeat(i, len(egress))
        combinations = np.append(combinations, np.column_stack((ingress, egress)), axis=0)
    # make eclipses
    ecl_indices = np.zeros((0, 15), dtype=np.int_)
    for comb in combinations:
        # restrict duration to half the orbital period
        duration = t_model[minimum_1[comb[1]]] - t_model[minimum_1[comb[0]]]
        condition = (duration < p_orb / 2)
        # eclipses may not be in gaps in the phase coverage
        t_ingress = t_model[peaks_2_n[comb[0]]]
        t_egress = t_model[peaks_2_n[comb[1]]]
        gap_cond = True
        for tg in t_gaps:
            gap_cond_1 = (t_ingress > tg[0]) & (t_ingress < tg[-1])  # ingress not in gap
            gap_cond_2 = (t_egress > tg[0]) & (t_egress < tg[-1])  # egress not in gap
            gap_cond_3 = gap_cond_1 | gap_cond_2 | ((t_ingress < tg[0]) & (t_egress > tg[-1]))
            gap_cond_3 &= ((tg[-1] - tg[0]) / (t_egress - t_ingress)) > 0.5  # too much eclipse in gap
            if gap_cond_1 | gap_cond_2 | gap_cond_3:
                gap_cond = False  # at least one side in gap or too big a gap
        condition &= gap_cond
        if condition:
            # decide about the presence of a flat bottom (deriv_2 has to become negative)
            if (np.min(deriv_2[peaks_2_p[comb[0]]:peaks_2_p[comb[1]] + 1]) > 0):
                p_2_p_1 = zeros_1_in[comb[0]]  # adjust peaks_2_p for consistency
                p_2_p_2 = zeros_1_in[comb[1]]
                m_1_i_1 = zeros_1_in[comb[0]]  # adjust minimum_1_in for consistency
                m_1_i_2 = zeros_1_in[comb[1]]
            else:
                p_2_p_1 = peaks_2_p[comb[0]]
                p_2_p_2 = peaks_2_p[comb[1]]
                m_1_i_1 = minimum_1_in[comb[0]]
                m_1_i_2 = minimum_1_in[comb[1]]
            # put the eclipse minimum at or in the middle between the minimum_1_in
            minimum_1_in_mid = (m_1_i_1 + m_1_i_2) // 2
            # assemble eclipse indices
            ecl = [zeros_1[comb[0]], minimum_1[comb[0]], peaks_2_n[comb[0]], peaks_1[comb[0]],
                   p_2_p_1, m_1_i_1, zeros_1_in[comb[0]], minimum_1_in_mid, zeros_1_in[comb[1]], m_1_i_2, p_2_p_2,
                   peaks_1[comb[1]], peaks_2_n[comb[1]], minimum_1[comb[1]], zeros_1[comb[1]]]
            # check in the harmonic light curve model that all points in eclipse lie beneath the top points
            i_mid_ecl = (ecl[2] + ecl[-3]) // 2
            line_check_1 = np.all(model_h[ecl[2]:i_mid_ecl] <= model_h[ecl[2]])
            line_check_2 = np.all(model_h[i_mid_ecl:ecl[-3]] <= model_h[ecl[-3]])
            if line_check_1 & line_check_2:
                ecl_indices = np.vstack((ecl_indices, np.array([ecl])))
    # check overlap and pick the highest peaks in deriv_1 in case of overlap
    indices = np.arange(len(ecl_indices))
    combinations = np.zeros((0, 2), dtype=np.int_)
    for i in indices:
        ecl_2 = indices[indices > i]  # select only eclipses after ecl_1 to avoid doubles
        ecl_1 = np.repeat(i, len(ecl_2))
        combinations = np.append(combinations, np.column_stack((ecl_1, ecl_2)), axis=0)
    ecl_remove = []
    for comb in combinations:
        overlap_1 = (ecl_indices[comb[0], 1] >= ecl_indices[comb[1], 1])
        overlap_1 &= (ecl_indices[comb[0], 1] < ecl_indices[comb[1], -2])
        overlap_2 = (ecl_indices[comb[0], 1] <= ecl_indices[comb[1], 1])
        overlap_2 &= (ecl_indices[comb[0], -2] > ecl_indices[comb[1], 1])
        if (overlap_1 | overlap_2):
            peak_height = deriv_1[ecl_indices[comb, -4]] - deriv_1[ecl_indices[comb, 3]]
            ecl_remove.append(comb[np.argmin(peak_height)])
    remove_mask = np.ones(len(ecl_indices), dtype=np.bool_)
    remove_mask[np.array(ecl_remove).astype(np.int_)] = False
    ecl_indices = ecl_indices[remove_mask]  # delete only first two args in numba
    return ecl_indices


@nb.njit(cache=True)
def measure_eclipses(t_model, model_h, ecl_indices, noise_level):
    """Measure the times, durations and depths of the eclipses
    
    Parameters
    ----------
    t_model: numpy.ndarray[float]
        Set of time points for a model of sinudoids
    model_h: numpy.ndarray[float]
        Model of harmonic sinusoids at t_model
    ecl_indices: numpy.ndarray[int]
        Two dimensional array of eclipse indices.
        Each eclipse has indices corresponding to
        several prominent points.
    noise_level: float
        The noise level (standard deviation of the residuals)

    Returns
    -------
    ecl_min: numpy.ndarray[float]
        Times of the eclipse minima
    ecl_mid: numpy.ndarray[float]
        Times of the eclipse midpoints
    widths: numpy.ndarray[float]
        Durations of the eclipses
    depths: numpy.ndarray[float]
        Depths of the eclipses
    ecl_mid_b: numpy.ndarray[float]
        Times of the flat bottom midpoints
    widths_b: numpy.ndarray[float]
        Durations of the flat bottoms
    t_i_1_err: numpy.ndarray[float]
        Timing error estimates for the eclipse edges
    t_i_2_err: numpy.ndarray[float]
        Timing error estimates for the eclipse edges
    t_b_i_1_err: numpy.ndarray[float]
        Timing error estimates for the eclipse bottoms
    t_b_i_2_err: numpy.ndarray[float]
        Timing error estimates for the eclipse bottoms
    
    Notes
    -----
    Intended for use in detect_eclipses, in conjunction
    with mark_eclipse_peaks.
    Will remove eclipses below noise_level/4
    """
    # make the timing measurements
    t_m1_1 = t_model[ecl_indices[:, 1]]  # first times of minimum deriv_1 (from minimum_1)
    t_m1_2 = t_model[ecl_indices[:, -2]]  # last times of minimum deriv_1 (from minimum_1)
    t_p2n_1 = t_model[ecl_indices[:, 2]]  # first times of negative extremum deriv_2 (from peaks_2_n)
    t_p2n_2 = t_model[ecl_indices[:, -3]]  # last times of negative extremum deriv_2 (from peaks_2_n)
    t_m1i_1 = t_model[ecl_indices[:, 5]]  # first times of inner minimum deriv_1 (from minimum_1_in)
    t_m1i_2 = t_model[ecl_indices[:, -6]]  # last times of inner minimum deriv_1 (from minimum_1_in)
    t_p2p_1 = t_model[ecl_indices[:, 4]]  # first times of positive extremum deriv_2 (from peaks_2_p)
    t_p2p_2 = t_model[ecl_indices[:, -5]]  # last times of positive extremum deriv_2 (from peaks_2_p)
    ecl_min = t_model[ecl_indices[:, 7]]  # minimum taken from minimum_1_in_mid
    # determine the eclipse edges from the midpoint between peaks_2_n and minimum_1
    t_i_1 = (t_m1_1 + t_p2n_1) / 2
    t_i_2 = (t_m1_2 + t_p2n_2) / 2
    indices_t_i_1 = np.searchsorted(t_model, t_i_1)  # if t_model is granular enough, this should be precise enough
    indices_t_i_2 = np.searchsorted(t_model, t_i_2)  # if t_model is granular enough, this should be precise enough
    # determine the bottom edges from the midpoint between peaks_2_p and minimum_1_in
    t_b_i_1 = (t_m1i_1 + t_p2p_1) / 2
    t_b_i_2 = (t_m1i_2 + t_p2p_2) / 2
    indices_t_b_i_1 = np.searchsorted(t_model, t_b_i_1)  # if t_model is granular enough, this should be precise enough
    indices_t_b_i_2 = np.searchsorted(t_model, t_b_i_2)  # if t_model is granular enough, this should be precise enough
    # use the intervals as 3 sigma limits on either side
    t_i_1_err = (t_p2n_1 - t_m1_1) / 6
    t_i_1_err[t_i_1_err == 0] = 0.00001  # avoid zeros
    t_i_2_err = (t_m1_2 - t_p2n_2) / 6
    t_i_2_err[t_i_2_err == 0] = 0.00001  # avoid zeros
    # do the same for the inner intervals (flat bottom)
    t_b_i_1_err = (t_m1i_1 - t_p2p_1) / 6
    t_b_i_1_err[t_b_i_1_err == 0] = 0.00001  # avoid zeros
    t_b_i_2_err = (t_p2p_2 - t_m1i_2) / 6
    t_b_i_2_err[t_b_i_2_err == 0] = 0.00001  # avoid zeros
    # convert to midpoints and widths and take the deepest depth measured in two ways
    ecl_mid = (t_i_1 + t_i_2) / 2
    widths = (t_i_2 - t_i_1)
    ecl_mid_b = (t_b_i_1 + t_b_i_2) / 2
    widths_b = (t_b_i_2 - t_b_i_1)
    depths_1 = (model_h[indices_t_i_1] + model_h[indices_t_i_2]) / 2 - model_h[ecl_indices[:, 6]]
    depths_2 = (model_h[indices_t_i_1] - model_h[indices_t_b_i_1]
                + model_h[indices_t_i_2] - model_h[indices_t_b_i_2]) / 2
    depths = np.copy(depths_1)
    d2_larger = (depths_2 > depths_1)
    depths[d2_larger] = depths_2[d2_larger]  # max no support for optional args numba
    # remove too shallow eclipses
    remove_shallow = (depths > noise_level / 4)
    ecl_min = ecl_min[remove_shallow]
    ecl_mid = ecl_mid[remove_shallow]
    widths = widths[remove_shallow]
    depths = depths[remove_shallow]
    t_i_1_err = t_i_1_err[remove_shallow]
    t_i_2_err = t_i_2_err[remove_shallow]
    t_b_i_1_err = t_b_i_1_err[remove_shallow]
    t_b_i_2_err = t_b_i_2_err[remove_shallow]
    return ecl_min, ecl_mid, widths, depths, ecl_mid_b, widths_b, t_i_1_err, t_i_2_err, t_b_i_1_err, t_b_i_2_err


@nb.njit(cache=True)
def select_eclipses(p_orb, ecl_min, widths, depths):
    """Select the best combination of primary and secondary
    
    Parameters
    ----------
    p_orb: float
        Orbital period of the eclipsing binary in days
    ecl_min: numpy.ndarray[float]
        Times of the eclipse minima
    widths: numpy.ndarray[float]
        Durations of the eclipses
    depths: numpy.ndarray[float]
        Depths of the eclipses
    
    Returns
    -------
    best_comb: numpy.ndarray[int]
        Best combination of eclipses based on
        the combined depth.
    """
    n_ecl = len(ecl_min)
    # now pick out two consecutive, fully covered eclipses
    indices = np.arange(n_ecl)
    combinations = np.zeros((n_ecl - 1, 2), dtype=np.int_)
    for i, (j, k) in enumerate(zip(indices[:-1], indices[1:])):
        combinations[i] = [j, k]  # numba no like array initialisation of a list comprehension
    # also pick out two candidates skipping one candidate
    if (n_ecl > 1):
        combinations_2 = np.zeros((n_ecl - 2, 2), dtype=np.int_)
        for i, (j, k) in enumerate(zip(indices[:-2], indices[2:])):
            combinations_2[i] = [j, k]  # numba no like array initialisation of a list comprehension
        if (len(combinations_2) > 0):
            combinations = np.append(combinations, combinations_2, axis=0)
    # also pick out two candidates skipping two candidates
    if (n_ecl > 2):
        combinations_3 = np.zeros((n_ecl - 3, 2), dtype=np.int_)
        for i, (j, k) in enumerate(zip(indices[:-3], indices[3:])):
            combinations_3[i] = [j, k]  # numba no like array initialisation of a list comprehension
        if (len(combinations_3) > 0):
            combinations = np.append(combinations, combinations_3, axis=0)
    # check overlap of the eclipses
    comb_remove = np.zeros(0, dtype=np.int_)
    for i, comb in enumerate(combinations):
        comb_dist_direct = abs(ecl_min[comb[0]] - ecl_min[comb[1]])
        comb_dist_wrapped = abs(comb_dist_direct - p_orb)
        max_width = max(widths[comb[0]] / 2, widths[comb[1]] / 2)
        if (comb_dist_direct < max_width) | (comb_dist_wrapped < max_width):
            comb_remove = np.append(comb_remove, i)
    remove_mask = np.ones(len(combinations), dtype=np.bool_)
    remove_mask[comb_remove] = False
    combinations = combinations[remove_mask]  # delete only has first two args in numba
    if (len(combinations) == 0):
        return np.zeros(0, dtype=np.int_)
    # sum of depths should be largest for the most complete set of eclipses
    comb_d = depths[combinations[:, 0]] + depths[combinations[:, 1]]
    best_comb = combinations[np.argmax(comb_d)]  # argmax automatically picks the first in ties
    return best_comb


@nb.njit(cache=True)
def match_eclipses(ecl_indices_1, ecl_indices_2, depths_2):
    """Finds which eclipses in 2 match the eclipses in 1
    
    Parameters
    ----------
    ecl_indices_1: numpy.ndarray[int]
        Two dimensional array of eclipse indices.
        Each eclipse has indices corresponding to
        several prominent points.
    ecl_indices_2: numpy.ndarray[int]
        Two dimensional array of eclipse indices.
        Each eclipse has indices corresponding to
        several prominent points.
    depths_2: numpy.ndarray[float]
        Depths of the eclipses
    
    Returns
    -------
    matches: numpy.ndarray[float]
        Indices of array 2 for each of array 1
    unmatched: numpy.ndarray[bool]
        Mask of unmatched ecl in array 2
    """
    n_ecl_1 = len(ecl_indices_1)
    n_ecl_2 = len(ecl_indices_2)
    indices_1 = np.arange(n_ecl_1)
    indices_2 = np.arange(n_ecl_2)
    # find the matches in 2 for each ecl in 1
    matches = -np.ones(n_ecl_1, dtype=np.int_)
    unmatched = np.ones(n_ecl_2, dtype=np.bool_)
    for i in indices_1:
        match = (ecl_indices_1[i, 1] < ecl_indices_2[:, 2]) & (ecl_indices_1[i, -2] > ecl_indices_2[:, -3])
        unmatched[match] = False  # keeps track of matched 2s
        if np.sum(match) > 1:
            j = indices_2[match][np.argmax(depths_2[match])]
        elif np.sum(match) == 1:
            j = indices_2[match][0]
        else:
            j = -1
        matches[i] = j
    return matches, unmatched


def detect_eclipses(p_orb, f_n, a_n, ph_n, noise_level, t_gaps):
    """Determine the eclipse midpoints, depths and widths from the derivatives
    of the harmonic model.
    
    Parameters
    ----------
    p_orb: float
        Orbital period of the eclipsing binary in days
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
    noise_level: float
        The noise level (standard deviation of the residuals)
    t_gaps: numpy.ndarray[float]
        Gap timestamps in pairs
    
    Returns
    -------
    ecl_indices: numpy.ndarray[int], None
        Indices of several important points in the harmonic model
        as generated here (see function for details)
    ecl_indices: numpy.ndarray[int], None
        Indices of several important points in the harmonic model
        refined by using all harmonics (may or may not equal ecl_indices)
    
    Notes
    -----
    The result is ordered according to depth so that the deepest eclipse is first.
    
    The code in this function utilises a similar idea to find the eclipses
    as ECLIPSR (except somewhat simpler due to analytic functions instead
    of raw data). See IJspeert 2021.
    """
    # make a timeframe from 0 to two P to catch both eclipses in full if present
    t_model = np.linspace(0, 2 * p_orb, 10**6)
    harmonics, harmonic_n = find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    f_h, a_h, ph_h = f_n[harmonics], a_n[harmonics], ph_n[harmonics]
    for i, n in enumerate([20, 40, np.max(harmonic_n)]):
        low_h = (harmonic_n <= n)  # restrict harmonics to avoid interference of high frequencies
        model_h = tsf.sum_sines(t_model, f_h[low_h], a_h[low_h], ph_h[low_h])
        deriv_1 = tsf.sum_sines_deriv(t_model, f_h[low_h], a_h[low_h], ph_h[low_h], deriv=1)
        deriv_2 = tsf.sum_sines_deriv(t_model, f_h[low_h], a_h[low_h], ph_h[low_h], deriv=2)
        # find the eclipses
        output_a = mark_eclipse_peaks(t_model, deriv_1, deriv_2, noise_level, t_gaps, n_prominent=24)
        peaks_1, slope_sign, zeros_1, peaks_2_n, minimum_1, zeros_1_in, peaks_2_p, minimum_1_in = output_a
        ecl_indices = assemble_eclipses(p_orb, t_model, deriv_1, deriv_2, model_h, t_gaps, peaks_1, slope_sign, zeros_1,
                                        peaks_2_n, minimum_1, zeros_1_in, peaks_2_p, minimum_1_in)
        # measure them up
        output_b = measure_eclipses(t_model, model_h, ecl_indices, noise_level)
        ecl_min, ecl_mid, widths, depths, ecl_mid_b, widths_b, t_i_1_err, t_i_2_err, t_b_i_1_err, t_b_i_2_err = output_b
        if (len(ecl_min) == 0):
            continue
        # pick the best pair
        best_comb = select_eclipses(p_orb, ecl_min, widths, depths)
        if (len(best_comb) == 0):
            continue
        else:
            # if we found a best pair, move on to stage two
            break
    # if still nothing was found, return
    if (len(ecl_min) == 0):
        return (None,) * 9 + (ecl_indices,)
    elif (len(best_comb) == 0):
        return (None,) * 9 + (ecl_indices,)
    # select the best combination of eclipses
    ecl_indices = ecl_indices[best_comb]
    # refine measurements for the selected eclipses by using all harmonics
    if (i < 2):
        model_h = tsf.sum_sines(t_model, f_h, a_h, ph_h)
        deriv_1 = tsf.sum_sines_deriv(t_model, f_h, a_h, ph_h, deriv=1)
        deriv_2 = tsf.sum_sines_deriv(t_model, f_h, a_h, ph_h, deriv=2)
        ecl_1 = ecl_indices[0]
        ecl_2 = ecl_indices[1]
        output_c = mark_eclipse_peaks(t_model[ecl_1[1]:ecl_1[-2]], deriv_1[ecl_1[1]:ecl_1[-2]],
                                      deriv_2[ecl_1[1]:ecl_1[-2]], noise_level, t_gaps, n_prominent=4)
        output_d = mark_eclipse_peaks(t_model[ecl_2[1]:ecl_2[-2]], deriv_1[ecl_2[1]:ecl_2[-2]],
                                      deriv_2[ecl_2[1]:ecl_2[-2]], noise_level, t_gaps, n_prominent=4)
        peaks_1 = np.append(output_c[0] + ecl_1[1], output_d[0] + ecl_2[1])
        slope_sign = np.append(output_c[1], output_d[1])
        zeros_1 = np.append(output_c[2] + ecl_1[1], output_d[2] + ecl_2[1])
        peaks_2_n = np.append(output_c[3] + ecl_1[1], output_d[3] + ecl_2[1])
        minimum_1 = np.append(output_c[4] + ecl_1[1], output_d[4] + ecl_2[1])
        zeros_1_in = np.append(output_c[5] + ecl_1[1], output_d[5] + ecl_2[1])
        peaks_2_p = np.append(output_c[6] + ecl_1[1], output_d[6] + ecl_2[1])
        minimum_1_in = np.append(output_c[7] + ecl_1[1], output_d[7] + ecl_2[1])
        ecl_indices_ref = assemble_eclipses(p_orb, t_model, deriv_1, deriv_2, model_h, t_gaps, peaks_1, slope_sign,
                                            zeros_1, peaks_2_n, minimum_1, zeros_1_in, peaks_2_p, minimum_1_in)
    else:
        ecl_indices_ref = np.copy(ecl_indices)
    return ecl_indices, ecl_indices_ref


def timings_from_ecl_indices(ecl_indices, p_orb, f_n, a_n, ph_n):
    """Translate the eclipse indices to timings and depths
    
    Parameters
    ----------
    ecl_indices: numpy.ndarray[int], None
        Indices of several important points in the harmonic model
    p_orb: float
        Orbital period of the eclipsing binary in days
    f_n: numpy.ndarray[float]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[float]
        The phases of a number of sine waves
        
    Returns
    -------
    t_1: float, None
        Time of primary minimum with respect to the mean time
    t_2: float, None
        Time of secondary minimum with respect to the mean time
    t_contacts: tuple[float], None
        Measurements of the times of contact:
        t_1_1, t_1_2, t_2_1, t_2_2
    t_int_tan: tuple[float], None
        Measurements of the times near internal tangency:
        t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2
    depths: numpy.ndarray[float], None
        Measurements of the eclipse depths in units of a_n
    t_i_1_err: numpy.ndarray[float], None
        Measurement error estimates of the first contacts
    t_i_2_err: numpy.ndarray[float], None
        Measurement error estimates of the last contacts
    t_b_i_1_err: numpy.ndarray[float]
        Measurement error estimates of the first internal tangencies
    t_b_i_2_err: numpy.ndarray[float]
        Measurement error estimates of the last internal tangencies
    """
    # make a timeframe from 0 to two P to catch both eclipses in full if present (has to match detect_eclipses)
    t_model = np.linspace(0, 2 * p_orb, 10**6)
    harmonics, harmonic_n = find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    f_h, a_h, ph_h = f_n[harmonics], a_n[harmonics], ph_n[harmonics]
    model_h = tsf.sum_sines(t_model, f_h, a_h, ph_h)
    # measure up the eclipses
    for n in [np.max(harmonic_n), 40, 20]:
        low_h = (harmonic_n <= n)
        model_h = tsf.sum_sines(t_model, f_h[low_h], a_h[low_h], ph_h[low_h])
        output = measure_eclipses(t_model, model_h, ecl_indices, 0)
        ecl_min, ecl_mid, widths, depths, ecl_mid_b, widths_b, t_i_1_err, t_i_2_err, t_b_i_1_err, t_b_i_2_err = output
        # negative depths could occur, returning <2 eclipses - n_h 20 should be guaranteed to give 2
        if (len(ecl_min) == 2):
            break
    # put the deepest eclipse at zero (and make sure to get the edge timings in the right spot)
    sorter = np.argsort(depths)[::-1]
    p, s = sorter[0], sorter[1]  # primary, secondary
    t_1 = ecl_min[p]
    t_2 = (ecl_min[s] - t_1) % p_orb
    ecl_mid = (ecl_mid - t_1) % p_orb
    if ecl_mid[p] > (p_orb - widths[p] / 2):
        ecl_mid[p] = ecl_mid[p] - p_orb
    ecl_mid_b = (ecl_mid_b - t_1) % p_orb
    if ecl_mid_b[p] > (p_orb - widths[p] / 2):
        ecl_mid_b[p] = ecl_mid_b[p] - p_orb
    # define in terms of time points
    t_1_1 = ecl_mid[p] - (widths[p] / 2)  # time of primary first contact
    t_1_2 = ecl_mid[p] + (widths[p] / 2)  # time of primary last contact
    t_2_1 = ecl_mid[s] - (widths[s] / 2)  # time of secondary first contact
    t_2_2 = ecl_mid[s] + (widths[s] / 2)  # time of secondary last contact
    t_b_1_1 = ecl_mid_b[p] - (widths_b[p] / 2)  # time of primary first internal tangency
    t_b_1_2 = ecl_mid_b[p] + (widths_b[p] / 2)  # time of primary last internal tangency
    t_b_2_1 = ecl_mid_b[s] - (widths_b[s] / 2)  # time of secondary first internal tangency
    t_b_2_2 = ecl_mid_b[s] + (widths_b[s] / 2)  # time of secondary last internal tangency
    # translate the timings by the time of primary minimum with respect to the mean time
    t_2 = t_2 + t_1
    t_contacts = (t_1_1 + t_1, t_1_2 + t_1, t_2_1 + t_1, t_2_2 + t_1)
    t_int_tan = (t_b_1_1 + t_1, t_b_1_2 + t_1, t_b_2_1 + t_1, t_b_2_2 + t_1)
    # redetermine depths a tiny bit more precisely
    depths = measure_harmonic_depths(f_h[low_h], a_h[low_h], ph_h[low_h], t_1, t_2, *t_contacts, *t_int_tan)
    return t_1, t_2, t_contacts, t_int_tan, depths, t_i_1_err, t_i_2_err, t_b_i_1_err, t_b_i_2_err


def linear_regression_uncertainty(p_orb, t_tot, sigma_t=1):
    """Calculates the linear regression errors on period and t_zero

    Parameters
    ---------
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_tot: float
        Total time base of observations
    sigma_t: float
        Error in the individual time measurements

    Returns
    -------
    p_err: float
        Error in the period
    t_err: float
        Error in t_zero
    p_t_cov: float
        Covariance between the period and t_zero
    
    Notes
    -----
    The number of eclipses, computed from the period and
    time base, is taken to be a contiguous set.
    var_matrix:
    [[std[0]**2          , std[0]*std[1]*corr],
     [std[0]*std[1]*corr,           std[1]**2]]
    """
    # number of observed eclipses (technically contiguous)
    n = int(abs(t_tot // p_orb)) + 1
    # M
    matrix = np.column_stack((np.ones(n, dtype=int), np.arange(n, dtype=int)))
    # M^-1
    matrix_inv = np.linalg.pinv(matrix)  # inverse (of a general matrix)
    # M^-1 S M^-1^T, S unit matrix times some sigma (no covariance in the data)
    var_matrix = matrix_inv @ matrix_inv.T
    var_matrix = var_matrix * sigma_t**2
    # errors in the period and t_zero
    t_err = np.sqrt(var_matrix[0, 0])
    p_err = np.sqrt(var_matrix[1, 1])
    p_t_corr = var_matrix[0, 1] / (t_err * p_err)  # or [1, 0]
    return p_err, t_err, p_t_corr


@nb.njit(cache=True)
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
    ν = π / 2 - ω + θ
    """
    nu = np.pi / 2 - w + theta
    return nu


@nb.njit(cache=True)
def eccentric_anomaly(nu, e):
    """Eccentric anomaly in terms of true anomaly and eccentricity
    
    Parameters
    ----------
    nu: float, numpy.ndarray[float]
        True anomaly
    e: float, numpy.ndarray[float]
        Eccentricity of the orbit
    
    Returns
    -------
    : float, numpy.ndarray[float]
        Eccentric anomaly
    """
    return 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))


@nb.njit(cache=True)
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
    
    def indefinite_integral(nu, ecc):
        term_1 = 2 * np.arctan2(np.sqrt(1 - ecc) * np.sin(nu / 2), np.sqrt(1 + ecc) * np.cos(nu / 2))
        term_2 = - ecc * np.sqrt(1 - ecc**2) * np.sin(nu) / (1 + ecc * np.cos(nu))
        mod_term = 4 * np.pi * ((nu // (2 * np.pi) + 1) // 2)  # correction term for going over 2pi
        return term_1 + term_2 + mod_term
    
    end_boundary = indefinite_integral(nu_2, e)
    start_boundary = indefinite_integral(nu_1, e)
    integral = end_boundary - start_boundary
    return integral


@nb.njit(cache=True)
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
    # previous (identical except for a factor 1/2 which doesn't matter because it equals zero) formula, from Kopal 1959
    # term_1 = (1 - e * np.sin(theta - w)) * sin_i_2 * np.sin(2*theta)
    # term_2 = 2 * e * np.cos(theta - w) * (1 - np.cos(theta)**2 * sin_i_2)
    minimize = e * np.cos(theta - w) + sin_i_2 * np.cos(theta) * (np.sin(theta) - e * np.cos(w))
    return minimize


@nb.njit(cache=True)
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
    """
    sin_i_2 = np.sin(i)**2
    deriv = -e * np.cos(w) * (1 - sin_i_2) * np.sin(theta) + e * np.sin(w) * np.cos(theta) + sin_i_2 * np.cos(2 * theta)
    return deriv


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
    theta_3: float
        Phase angle of maximum separation between 1 and 2
    theta_4: float
        Phase angle of maximum separation between 2 and 1
    """
    try:
        opt_1 = sp.optimize.root_scalar(delta_deriv, args=(e, w, i), method='brentq', bracket=(-1, 1))
        theta_1 = opt_1.root
    except ValueError:
        theta_1 = 0
    try:
        opt_2 = sp.optimize.root_scalar(delta_deriv, args=(e, w, i), method='brentq', bracket=(np.pi - 1, np.pi + 1))
        theta_2 = opt_2.root
    except ValueError:
        theta_2 = np.pi
    try:
        opt_3 = sp.optimize.root_scalar(delta_deriv, args=(e, w, i), method='brentq',
                                        bracket=(np.pi / 2 - 1, np.pi / 2 + 1))
        theta_3 = opt_3.root
    except ValueError:
        theta_3 = np.pi / 2
    try:
        opt_4 = sp.optimize.root_scalar(delta_deriv, args=(e, w, i), method='brentq',
                                        bracket=(3 * np.pi / 2 - 1, 3 * np.pi / 2 + 1))
        theta_4 = opt_4.root
    except ValueError:
        theta_4 = 3 * np.pi / 2
    return theta_1, theta_2, theta_3, theta_4


@nb.njit(cache=True)
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
    Other implementation for minima_phase_angles that can be JIT-ted.
    On its own it is 10x slower, but as part of other functions it can be faster
    if it means that other function can then also be JIT-ted
    """
    x0 = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])  # initial theta values
    if (e == 0):
        # this would break, so return the defaults for circular orbits
        return 0.0, np.pi, np.pi / 2, 3 * np.pi / 2
    # use the derivative of the projected distance to get theta angles
    deriv_1 = delta_deriv(x0, e, w, i)  # value of the projected distance derivative
    deriv_2 = delta_deriv_2(x0, e, w, i)  # value of the second derivative
    walk_sign = -np.sign(deriv_1).astype(np.int_) * np.sign(deriv_2).astype(np.int_)
    # walk along the curve to find zero points
    two_pi = 2 * np.pi
    step = 0.001  # step in rad (does not determine final precision)
    # start at x0
    cur_x = x0
    cur_y = delta_deriv(cur_x, e, w, i)
    f_sign_x0 = np.sign(cur_y).astype(np.int_)  # sign of delta_deriv at initial position
    # step in the desired direction
    try_x = cur_x + step * walk_sign
    try_y = delta_deriv(try_x, e, w, i)
    # check whether the sign stays the same
    check = (np.sign(cur_y) == np.sign(try_y))
    # if we take this many steps, we've gone full circle
    for _ in range(two_pi // step + 1):
        if not np.any(check):
            break
        # make the approved steps and continue if any were approved
        cur_x[check] = try_x[check]
        cur_y[check] = try_y[check]
        # try the next steps
        try_x[check] = cur_x[check] + step * walk_sign[check]
        try_y[check] = delta_deriv(try_x[check], e, w, i)
        # check whether the sign stays the same
        check[check] = (np.sign(cur_y[check]) == np.sign(try_y[check]))
    # interpolate for better precision than the angle step
    condition = (f_sign_x0 == 1)
    xp1 = np.where(condition, try_y, cur_y)
    yp1 = np.where(condition, try_x, cur_x)
    xp2 = np.where(condition, cur_y, try_y)
    yp2 = np.where(condition, cur_x, try_x)
    thetas_interp = ut.interp_two_points(np.zeros(len(x0)), xp1, yp1, xp2, yp2)
    thetas_interp[np.isnan(thetas_interp)] = 0
    thetas_interp = thetas_interp % two_pi
    # theta_1 is primary minimum, theta_2 is secondary minimum, the others are at the furthest projected distance
    theta_1, theta_3, theta_2, theta_4 = thetas_interp
    return theta_1, theta_2, theta_3, theta_4


@nb.njit(cache=True)
def contact_angles(phi, e, w, i, phi_0, ecl=1, contact=1):
    """Find the root of this function to obtain the phase angle between first/last contact
    and eclipse minimum for the primary/secondary eclipse

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
        Auxiliary angle (see Kopal 1959)
    ecl: int
        Primary or secondary eclipse (1 or 2)
    contact: int
        First or last contact (1 or 2)

    Returns
    -------
     eqn: float, numpy.ndarray[float]
        Numeric result of the function that should equal 0
    """
    sin_i_2 = np.sin(i)**2
    term_1 = np.sqrt(1 - sin_i_2 * np.cos(phi)**2)
    if (ecl == 1) & (contact == 1):
        eqn = term_1 - np.sqrt(1 - sin_i_2 * np.cos(phi_0)**2) * (1 + e * np.sin(w + phi))
    elif (ecl == 1) & (contact == 2):
        eqn = term_1 - np.sqrt(1 - sin_i_2 * np.cos(phi_0)**2) * (1 + e * np.sin(w - phi))
    elif (ecl == 2) & (contact == 1):
        eqn = term_1 - np.sqrt(1 - sin_i_2 * np.cos(phi_0)**2) * (1 - e * np.sin(w + phi))
    elif (ecl == 2) & (contact == 2):
        eqn = term_1 - np.sqrt(1 - sin_i_2 * np.cos(phi_0)**2) * (1 - e * np.sin(w - phi))
    else:
        print(f'ecl={ecl} and contact={contact} are not valid choices.')
        eqn = term_1 - np.sqrt(1 - sin_i_2 * np.cos(phi_0)**2)
    return eqn


@nb.njit(cache=True)
def contact_angles_radii(phi, e, w, i, r_sum_sma, ecl=1, contact=1):
    """Find the root of this function to obtain the phase angle between first/last contact
    and eclipse minimum for the primary/secondary eclipse using the radii

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
    r_sum_sma: float
        Sum of radii in units of the semi-major axis
        (can also be r_dif_sma for internal tangency)
    ecl: int
        Primary or secondary eclipse (1 or 2)
    contact: int
        First or last contact (1 or 2)

    Returns
    -------
     eqn: float, numpy.ndarray[float]
        Numeric result of the function that should equal 0
    """
    sin_i_2 = np.sin(i)**2
    term_1 = np.sqrt(1 - sin_i_2 * np.cos(phi)**2)
    if (ecl == 1) & (contact == 1):
        eqn = term_1 - r_sum_sma / (1 - e**2) * (1 + e * np.sin(w + phi))
    elif (ecl == 1) & (contact == 2):
        eqn = term_1 - r_sum_sma / (1 - e**2) * (1 + e * np.sin(w - phi))
    elif (ecl == 2) & (contact == 1):
        eqn = term_1 - r_sum_sma / (1 - e**2) * (1 - e * np.sin(w + phi))
    elif (ecl == 2) & (contact == 2):
        eqn = term_1 - r_sum_sma / (1 - e**2) * (1 - e * np.sin(w - phi))
    else:
        print(f'ecl={ecl} and contact={contact} are not valid choises.')
        eqn = term_1 - r_sum_sma / (1 - e**2)
    return eqn


def root_contact_phase_angles(e, w, i, phi_0):
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
        Auxiliary angle (see Kopal 1959)
        
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
    q1 = (-10**-5, np.pi / 2)
    try:
        opt_3 = sp.optimize.root_scalar(contact_angles, args=(e, w, i, phi_0, 1, 1), method='brentq', bracket=q1)
        phi_1_1 = opt_3.root
    except ValueError:
        phi_1_1 = 0  # interval likely did not have different signs because it did not quite reach 0 at 0
    try:
        opt_4 = sp.optimize.root_scalar(contact_angles, args=(e, w, i, phi_0, 1, 2), method='brentq', bracket=q1)
        phi_1_2 = opt_4.root
    except ValueError:
        phi_1_2 = 0  # interval likely did not have different signs because it did not quite reach 0 at 0
    try:
        opt_5 = sp.optimize.root_scalar(contact_angles, args=(e, w, i, phi_0, 2, 1), method='brentq', bracket=q1)
        phi_2_1 = opt_5.root
    except ValueError:
        phi_2_1 = 0  # interval likely did not have different signs because it did not quite reach 0 at 0
    try:
        opt_6 = sp.optimize.root_scalar(contact_angles, args=(e, w, i, phi_0, 2, 2), method='brentq', bracket=q1)
        phi_2_2 = opt_6.root
    except ValueError:
        phi_2_2 = 0  # interval likely did not have different signs because it did not quite reach 0 at 0
    return phi_1_1, phi_1_2, phi_2_1, phi_2_2


@nb.njit(cache=True)
def root_contact_phase_angles_2(e, w, i, phi_0):
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
        Auxiliary angle (see Kopal 1959)

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
    # walk along the curve to find zero points
    pi_two = np.pi / 2
    step = 0.001  # step in rad (does not determine final precision)
    # repeat for all phis
    phis = []
    for m, n in [[1, 1], [1, 2], [2, 1], [2, 2]]:
        cur_x = 0
        cur_y = contact_angles(0, e, w, i, phi_0, m, n)
        f_sign_x0 = int(np.sign(cur_y))  # sign of delta_deriv at initial position
        # step in the desired direction
        try_x = cur_x + step
        try_y = contact_angles(try_x, e, w, i, phi_0, m, n)
        # check whether the sign stays the same
        check = (np.sign(cur_y) == np.sign(try_y))
        # if we take this many steps, we should have found the root
        for _ in range(pi_two // step + 1):
            if not check:
                break
            # make the approved steps and continue if any were approved
            cur_x = try_x
            cur_y = try_y
            # try the next steps
            try_x = cur_x + step
            try_y = contact_angles(try_x, e, w, i, phi_0, m, n)
            # check whether the sign stays the same
            check = (np.sign(cur_y) == np.sign(try_y))
        # interpolate for better precision than the angle step
        xp1 = cur_y * (f_sign_x0 == -1) + try_y * (f_sign_x0 == 1)
        yp1 = cur_x * (f_sign_x0 == -1) + try_x * (f_sign_x0 == 1)
        xp2 = cur_y * (f_sign_x0 == 1) + try_y * (f_sign_x0 == -1)
        yp2 = cur_x * (f_sign_x0 == 1) + try_x * (f_sign_x0 == -1)
        phis.append(ut.interp_two_points(0, xp1, yp1, xp2, yp2))
    phi_1_1, phi_1_2, phi_2_1, phi_2_2 = phis
    return phi_1_1, phi_1_2, phi_2_1, phi_2_2


def root_contact_phase_angles_radii(e, w, i, r_sum_sma):
    """Determine the contact angles for given e, w, i, r_sum_sma

    Parameters
    ----------
    e: float
        Eccentricity of the orbit
    w: float
        Argument of periastron
    i: float
        Inclination of the orbit
    r_sum_sma: float
        Sum of radii in units of the semi-major axis
        (can also be r_dif_sma for internal tangency)

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
    q1 = (-10**-5, np.pi / 2)
    try:
        opt_3 = sp.optimize.root_scalar(contact_angles_radii, args=(e, w, i, r_sum_sma, 1, 1), method='brentq',
                                        bracket=q1)
        phi_1_1 = opt_3.root
    except ValueError:
        phi_1_1 = 0  # interval likely did not have different signs because it did not quite reach 0 at 0
    try:
        opt_4 = sp.optimize.root_scalar(contact_angles_radii, args=(e, w, i, r_sum_sma, 1, 2), method='brentq',
                                        bracket=q1)
        phi_1_2 = opt_4.root
    except ValueError:
        phi_1_2 = 0  # interval likely did not have different signs because it did not quite reach 0 at 0
    try:
        opt_5 = sp.optimize.root_scalar(contact_angles_radii, args=(e, w, i, r_sum_sma, 2, 1), method='brentq',
                                        bracket=q1)
        phi_2_1 = opt_5.root
    except ValueError:
        phi_2_1 = 0  # interval likely did not have different signs because it did not quite reach 0 at 0
    try:
        opt_6 = sp.optimize.root_scalar(contact_angles_radii, args=(e, w, i, r_sum_sma, 2, 2), method='brentq',
                                        bracket=q1)
        phi_2_2 = opt_6.root
    except ValueError:
        phi_2_2 = 0  # interval likely did not have different signs because it did not quite reach 0 at 0
    return phi_1_1, phi_1_2, phi_2_1, phi_2_2


@nb.njit(cache=True)
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
        Duration of primary first contact to minimum
    tau_1_2: float, numpy.ndarray[float]
        Duration of primary minimum to last contact
    tau_2_1: float, numpy.ndarray[float]
        Duration of secondary first contact to minimum
    tau_2_2: float, numpy.ndarray[float]
        Duration of secondary minimum to last contact
    i: float
        Inclination of the orbit
    phi_0: float
        Auxiliary angle (see Kopal 1959)

    Returns
    -------
    e: float, numpy.ndarray[float]
        Eccentricity for each set of input parameters
    w: float, numpy.ndarray[float]
        Argument of periastron for each set of input parameters
    """
    sin_i_2 = np.sin(i)**2
    cos_p0_2 = np.cos(phi_0)**2
    e_cos_w = np.pi * (t_2 / p_orb - t_1 / p_orb - 1 / 2) * (sin_i_2 / (1 + sin_i_2))
    e_sin_w = np.pi / (2 * np.sin(phi_0) * p_orb) * (tau_1_1 + tau_1_2 - tau_2_1 - tau_2_2)
    e_sin_w = e_sin_w * (sin_i_2 * cos_p0_2 / (1 - sin_i_2 * (1 + cos_p0_2)))
    e = np.sqrt(e_cos_w**2 + e_sin_w**2)
    w = np.arctan2(e_sin_w, e_cos_w) % (2 * np.pi)  # w in interval 0, 2pi
    return e, w


@nb.njit(cache=True)
def r_sum_sma_from_phi_0(e, i, phi_0):
    """Formula for the sum of radii in units of the semi-major axis
     from the angle phi_0
    
    Parameters
    ----------
    e: float, numpy.ndarray[float]
        Eccentricity
    i: float, numpy.ndarray[float]
        Inclination of the orbit
    phi_0: float, numpy.ndarray[float]
        Auxiliary angle, see Kopal 1959
    
    Returns
    -------
    r_sum_sma: float, numpy.ndarray[float]
        Sum of radii in units of the semi-major axis
    
    Notes
    -----
    Becomes 0 for e = 1, and then negative when e > 1.
    """
    r_sum_sma = np.sqrt((1 - np.sin(i)**2 * np.cos(phi_0)**2)) * (1 - e**2)  # (1 - e**2) not sqrt as in Kopal
    return r_sum_sma


@nb.njit(cache=True)
def phi_0_from_r_sum_sma(e, i, r_sum_sma):
    """Formula for the angle phi_0 from the sum of radii in units
    of the semi-major axis

    Parameters
    ----------
    e: float, numpy.ndarray[float]
        Eccentricity
    i: float, numpy.ndarray[float]
        Inclination of the orbit
    r_sum_sma: float, numpy.ndarray[float]
        Sum of radii in units of the semi-major axis

    Returns
    -------
    phi_0: float, numpy.ndarray[float]
        Auxiliary angle, see Kopal 1959
    
    Raises
    ------
    ZeroDivisionError
        If either e == 1 or i == 0 and all inputs are float.
        If inputs involve arrays, gives RuntimeWarning instead.
    """
    phi_0 = np.arccos(np.sqrt(1 - r_sum_sma**2 / (1 - e**2)**2) / np.sin(i))
    return phi_0


@nb.njit(cache=True)
def r_ratio_from_rho_0(e, w, i, phi_0, rho_0):
    """Formula for the sum of radii in units of the semi-major axis
     from the angle phi_0

    Parameters
    ----------
    e: float, numpy.ndarray[float]
        Eccentricity
    w: float, numpy.ndarray[float]
        Argument of periastron
    i: float, numpy.ndarray[float]
        Inclination of the orbit
    phi_0: float, numpy.ndarray[float]
        Auxiliary angle, see Kopal 1959
    rho_0: float, numpy.ndarray[float]
        Auxiliary scaled distance

    Returns
    -------
    r_ratio: float, numpy.ndarray[float]
        Radius ratio r_2/r_1
    
    Notes
    -----
    Using rho_0 can be advantagous in case of partial eclipses, however,
    some information is given up: we don't know which star is which size.
    (if we assume main sequence, star 1 - the primary - is the bigger one)
    """
    esinw = e * np.sin(w)
    r_sum = r_sum_sma_from_phi_0(e, i, phi_0)
    d_term = (1 - e**2)**2 * np.cos(phi_0) * np.sin(phi_0) * np.cos(i) / (1 + esinw)**2
    # we lose some information: we don't know which star is which size
    r_1 = r_sum / 2 + np.sqrt(rho_0 + d_term - r_sum**2) / 2
    r_2 = r_sum / 2 - np.sqrt(rho_0 + d_term - r_sum**2) / 2
    r_ratio = r_2 / r_1
    return r_ratio


@nb.njit(cache=True)
def rho_0_from_r_ratio(e, w, i, r_sum_sma, r_ratio):
    """Formula for the angle phi_0 from the sum of radii in units
    of the semi-major axis

    Parameters
    ----------
    e: float, numpy.ndarray[float]
        Eccentricity
    w: float, numpy.ndarray[float]
        Argument of periastron
    i: float, numpy.ndarray[float]
        Inclination of the orbit
    r_sum_sma: float, numpy.ndarray[float]
        Sum of radii in units of the semi-major axis
    r_ratio: float, numpy.ndarray[float]
        Radius ratio r_2/r_1

    Returns
    -------
    rho_0: float, numpy.ndarray[float]
        Auxiliary scaled distance
    """
    r_1 = r_sum_sma / (1 + r_ratio)
    r_2 = r_sum_sma * r_ratio / (1 + r_ratio)
    esinw = e * np.sin(w)
    phi_0 = phi_0_from_r_sum_sma(e, i, r_sum_sma)
    rho_0 = 2 * r_1**2 + 2 * r_2**2 - (1 - e**2)**2 * np.cos(phi_0) * np.sin(phi_0) * np.cos(i) / (1 + esinw)**2
    return rho_0


@nb.njit(cache=True)
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
    delta^2 = a^2 (1-e^2)^2(1 - sin^2(i)cos^2(theta))/(1 - e sin(theta - w))^2
    sep = delta/a
    """
    num = (1 - e**2)**2 * (1 - np.sin(i)**2 * np.cos(theta)**2)
    denom = (1 - e * np.sin(theta - w))**2
    sep = np.sqrt(num / denom)
    return sep


@nb.njit(cache=True)
def covered_area(d, r_1, r_2):
    """Area covered for two overlapping circles separated by a certain distance
    
    Parameters
    ----------
    d: numpy.ndarray[float]
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
    # define conditions for separating parameter space
    cond_1 = (d > 1.00001 * abs(r_1 - r_2)) & (d < (r_1 + r_2))
    cond_2 = (d <= 1.00001 * abs(r_1 - r_2)) & np.invert(cond_1)
    cond_3 = np.invert(cond_1) & np.invert(cond_2)
    area = np.zeros(len(d))
    # formula for condition 1
    term_1 = r_1**2 * np.arccos((d[cond_1]**2 + r_1**2 - r_2**2) / (2 * d[cond_1] * r_1))
    term_2 = r_2**2 * np.arccos((d[cond_1]**2 + r_2**2 - r_1**2) / (2 * d[cond_1] * r_2))
    term_3 = - r_1 * r_2 * np.sqrt(1 - ((r_1**2 + r_2**2 - d[cond_1]**2) / (2 * r_1 * r_2))**2)
    area[cond_1] = term_1 + term_2 + term_3
    # value for condition 2
    area[cond_2] = np.pi * min(r_1**2, r_2**2)
    # value for condition 3
    area[cond_3] = 0
    return area


@nb.njit(cache=True)
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
    area_1 = covered_area(np.array([sep_1]), r_1, r_2)[0]
    area_2 = covered_area(np.array([sep_2]), r_1, r_2)[0]
    if (area_2 > 0) & (area_1 > 0):
        sb_ratio = d_ratio * area_1 / area_2
    elif (area_1 > 0):
        sb_ratio = 1000  # we get into territory where the secondary is not visible
    elif (area_2 > 0):
        sb_ratio = 0.001  # we get into territory where the primary is not visible
    else:
        sb_ratio = 1  # we get into territory where neither eclipse is visible
    return sb_ratio


@nb.njit(cache=True)
def eclipse_depth(theta, e, w, i, r_sum_sma, r_ratio, sb_ratio, theta_3, theta_4):
    """Theoretical eclipse depth in the assumption of uniform brightness
    
    Parameters
    ----------
    theta: numpy.ndarray[float]
        Phase angle (0 or pi degrees at conjunction)
        Around 0, the light of the primary is blocked,
        around pi, the light of the secondary is blocked.
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
    light_lost(1) = covered_area / (pi r_1^2 + pi r_2^2 sb_ratio)
    light_lost(2) = covered_area sb_ratio / (pi r_1^2 + pi r_2^2 sb_ratio)
    """
    # calculate radii and projected separation
    r_1 = r_sum_sma / (1 + r_ratio)
    r_2 = r_sum_sma * r_ratio / (1 + r_ratio)
    sep = projected_separation(e, w, i, theta)
    # with those, calculate the covered area and light lost due to that
    if (r_sum_sma == 0):
        light_lost = np.zeros(len(theta))
    else:
        area = covered_area(sep, r_1, r_2)
        light_lost = area / (np.pi * r_1**2 + np.pi * r_2**2 * sb_ratio)
    # factor sb_ratio depends on primary or secondary, theta ~ 180 is secondary
    cond_1 = (theta > theta_3) & (theta < theta_4)
    light_lost[cond_1] = light_lost[cond_1] * sb_ratio
    return light_lost


def eclipse_times(p_orb, t_zero, e, w, i, r_sum_sma, r_ratio):
    """Theoretical eclipse timings

    Parameters
    ----------
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_zero: float
        Time of deepest minimum with respect to the mean time
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
    
    Returns
    -------
    t_1: float
        Time of primary minimum in domain [0, p_orb)
    t_2: float
        Time of secondary minimum in domain [0, p_orb)
    t_1_1: float
        Duration of primary first contact to minimum
    t_1_2: float
        Duration of primary minimum to last contact
    t_2_1: float
        Duration of secondary first contact to minimum
    t_2_2: float
        Duration of secondary minimum to last contact
    t_b_1_1: float
        Time of primary first internal tangency to minimum
    t_b_1_2: float
        Time of primary minimum to second internal tangency
    t_b_2_1: float
        Time of secondary first internal tangency to minimum
    t_b_2_2: float
        Time of secondary minimum to second internal tangency
    """
    # minimise for the phases of minima (theta)
    theta_1, theta_2, theta_3, theta_4 = minima_phase_angles(e, w, i)
    if (theta_2 == 0):
        # if no solution is possible (no opposite signs), return infinite
        return 10**9
    # minimise for the contact angles
    phi_1_1, phi_1_2, phi_2_1, phi_2_2 = root_contact_phase_angles_radii(e, w, i, r_sum_sma)
    # minimise for the internal tangency angles
    r_dif_sma = abs(r_sum_sma * (1 - r_ratio) / (1 + r_ratio))
    psi_1_1, psi_1_2, psi_2_1, psi_2_2 = root_contact_phase_angles_radii(e, w, i, r_dif_sma)
    # calculate the true anomaly of minima
    nu_1 = true_anomaly(theta_1, w)
    nu_2 = true_anomaly(theta_2, w)
    nu_conj_1 = true_anomaly(0, w)
    nu_conj_2 = true_anomaly(np.pi, w)
    # calculate the integrals (displacement, durations, asymmetries)
    n = 2 * np.pi / p_orb  # the average daily motion
    disp = integral_kepler_2(nu_1, nu_2, e) / n
    integral_conj_1 = integral_kepler_2(nu_conj_1, nu_1, e) / n  # - if nu_conj_1 > nu_1, else +
    integral_conj_2 = integral_kepler_2(nu_conj_2, nu_2, e) / n  # - if nu_conj_2 > nu_2, else +
    # phi angles are measured from conjunction (theta = 0 or 180 deg)
    integral_tau_1_1 = integral_kepler_2(nu_conj_1 - phi_1_1, nu_conj_1, e) / n
    integral_tau_1_2 = integral_kepler_2(nu_conj_1, nu_conj_1 + phi_1_2, e) / n
    integral_tau_2_1 = integral_kepler_2(nu_conj_2 - phi_2_1, nu_conj_2, e) / n
    integral_tau_2_2 = integral_kepler_2(nu_conj_2, nu_conj_2 + phi_2_2, e) / n
    # psi angles are for internal tangency
    integral_bottom_1_1 = integral_kepler_2(nu_conj_1 - psi_1_1, nu_conj_1, e) / n
    integral_bottom_1_2 = integral_kepler_2(nu_conj_1, nu_conj_1 + psi_1_2, e) / n
    integral_bottom_2_1 = integral_kepler_2(nu_conj_2 - psi_2_1, nu_conj_2, e) / n
    integral_bottom_2_2 = integral_kepler_2(nu_conj_2, nu_conj_2 + psi_2_2, e) / n
    # convert intervals to times (adjust for difference between minimum and conjunction)
    t_1 = t_zero
    t_2 = t_zero + disp
    t_1_1 = t_1 - integral_conj_1 - integral_tau_1_1
    t_1_2 = t_1 - integral_conj_1 + integral_tau_1_2
    t_2_1 = t_2 - integral_conj_2 - integral_tau_2_1
    t_2_2 = t_2 - integral_conj_2 + integral_tau_2_2
    t_b_1_1 = t_1 - integral_conj_1 - integral_bottom_1_1
    t_b_1_2 = t_1 - integral_conj_1 + integral_bottom_1_2
    t_b_2_1 = t_2 - integral_conj_2 - integral_bottom_2_1
    t_b_2_2 = t_2 - integral_conj_2 + integral_bottom_2_2
    return t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2


def eclipse_depths(e, w, i, r_sum_sma, r_ratio, sb_ratio):
    """Theoretical eclipse depths at minimum
    
    Parameters
    ----------
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
    sb_ratio: float
        Surface brightness ratio sb_2/sb_1
    
    Returns
    -------
    depth_1: float
        Depth of primary minimum
    depth_2: float
        Depth of secondary minimum
    
    Notes
    -----
    Not to be confused with eclipse_depth
    """
    # minimise for the phases of minima (theta)
    theta_1, theta_2, theta_3, theta_4 = minima_phase_angles(e, w, i)
    phases_min = np.array([theta_1, theta_2])
    depth_1, depth_2 = eclipse_depth(phases_min, e, w, i, r_sum_sma, r_ratio, sb_ratio, theta_3, theta_4)
    return depth_1, depth_2


def objective_inclination(cosi, p_orb, timings_tau, depths, timings_err, depths_err):
    """Minimise this function to obtain an inclination estimate
    
    Parameters
    ----------
    cosi: float
        Cosine of the inclination of the orbit
    p_orb: float
        Orbital period of the eclipsing binary in days
    timings_tau: numpy.ndarray[float]
        Eclipse timings of minima, durations from/to first/last contact,
        and durations from/to first/last internal tangency:
        t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2,
        tau_b_1_1, tau_b_1_2, tau_b_2_1, tau_b_2_2
    depths: numpy.ndarray[float]
        Eclipse depth of the primary and secondary, depth_1, depth_2
    timings_err: numpy.ndarray[float]
        Error estimates for the eclipse timings,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err
        t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err
    depths_err: numpy.ndarray[float]
        Error estimates for the depths

    Returns
    -------
    likelihood: float
        Minus the likelihood with weighted sum of squared deviations of five
        outcomes of integrals of Kepler's second law for different time spans
        of the eclipses compared to the measured values.
    
    Notes
    -----
    r_ratio is set to 1 for this objective function
    """
    # unpack (parameter and) arguments
    i = np.arccos(cosi)
    t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2 = timings_tau[:6]
    d_1, d_2 = depths
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err = timings_err[:6]
    d_1_err, d_2_err = depths_err
    # obtain phi_0 and the approximate e and w
    phi_0 = np.pi * (tau_1_1 + tau_1_2 + tau_2_1 + tau_2_2) / (2 * p_orb)
    e, w = ecc_omega_approx(p_orb, t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2, i, phi_0)
    if (e >= 1):
        return 10**9  # we want to stay in orbit
    # minimise for the phases of minima (theta)
    theta_1, theta_2, theta_3, theta_4 = minima_phase_angles(e, w, i)
    if (theta_2 == 0):
        # if no solution is possible (no opposite signs), return infinite
        return 10**9
    # minimise for the contact angles
    phi_1_1, phi_1_2, phi_2_1, phi_2_2 = root_contact_phase_angles(e, w, i, phi_0)
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
    r_sum_sma = r_sum_sma_from_phi_0(e, i, phi_0)
    r_ratio = 1
    sb_ratio = sb_ratio_from_d_ratio((d_2 / d_1), e, w, i, r_sum_sma, r_ratio, theta_1, theta_2)
    phases_min = np.array([theta_1, theta_2])
    depth_1, depth_2 = eclipse_depth(phases_min, e, w, i, r_sum_sma, r_ratio, sb_ratio, theta_3, theta_4)
    # displacement of the minima, linearly sensitive to e cos(w) (and sensitive to i)
    r_displacement = ((t_2 - t_1) - disp) / np.sqrt(t_1_err**2 + t_2_err**2)
    # difference in duration of the minima, linearly sensitive to e sin(w) (and sensitive to i and phi_0)
    tau_err_tot = np.sqrt(t_1_1_err**2 + t_1_2_err**2 + t_2_1_err**2 + t_2_2_err**2)
    r_duration_dif = ((tau_1_1 + tau_1_2 - tau_2_1 - tau_2_2) - (dur_1 - dur_2)) / tau_err_tot
    # sum of durations of the minima, linearly sensitive to phi_0 (and sensitive to e sin(w), i and phi_0)
    r_duration_sum = ((tau_1_1 + tau_1_2 + tau_2_1 + tau_2_2) - (dur_1 + dur_2)) / tau_err_tot
    # depths
    r_depth_dif = ((d_1 - d_2) - (depth_1 - depth_2)) / np.sqrt(d_1_err**2 + d_2_err**2)
    r_depth_sum = ((d_1 + d_2) - (depth_1 + depth_2)) / np.sqrt(d_1_err**2 + d_2_err**2)
    # calculate the error-normalised residuals
    resid = np.array([r_displacement, r_duration_dif, r_duration_sum, r_depth_dif, r_depth_sum])
    likelihood = -tsf.calc_likelihood(resid)  # minus for minimisation
    return likelihood


def objective_incl_plus(params, p_orb, ecosw, esinw, timings_tau, depths, timings_err, depths_err):
    """Minimise this function to obtain an estimate of cosi, phi_0,
    ratio of radii and ratio of surface brightness

    Parameters
    ----------
    params: array-like[float]
        cosi, phi_0, log_rr, log_sb
    p_orb: float
        Orbital period of the eclipsing binary in days
    ecosw: float
        Eccentricity times cosine of omega
    esinw: float
        Eccentricity times sine of omega
    timings_tau: numpy.ndarray[float]
        Eclipse timings of minima, durations from/to first/last contact,
        and durations from/to first/last internal tangency:
        t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2,
        tau_b_1_1, tau_b_1_2, tau_b_2_1, tau_b_2_2
    depths: numpy.ndarray[float]
        Eclipse depth of the primary and secondary, depth_1, depth_2
    timings_err: numpy.ndarray[float]
        Error estimates for the eclipse timings,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err
        t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err
    depths_err: numpy.ndarray[float]
        Error estimates for the depths

    Returns
    -------
    likelihood: float
        Minus the likelihood with weighted sum of squared deviations of
        five outcomes of integrals of Kepler's second law for different
        time spans of the eclipses compared to the measured values.
    """
    # unpack parameters and arguments
    cosi, phi_0, log_rr, log_sb = params
    e = np.sqrt(ecosw**2 + esinw**2)
    w = np.arctan2(esinw, ecosw) % (2 * np.pi)
    i = np.arccos(cosi)
    r_sum_sma = r_sum_sma_from_phi_0(e, i, phi_0)
    r_ratio = 10**log_rr
    sb_ratio = 10**log_sb
    t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2 = timings_tau[:6]
    tau_b_1_1, tau_b_1_2, tau_b_2_1, tau_b_2_2 = timings_tau[6:]
    d_1, d_2 = depths
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err = timings_err[:6]
    t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err = timings_err[6:]
    d_1_err, d_2_err = depths_err
    if (e >= 1):
        return 10**9  # we want to stay in orbit
    # minimise for the phases of minima (theta)
    theta_1, theta_2, theta_3, theta_4 = minima_phase_angles(e, w, i)
    if (theta_2 == 0):
        # if no solution is possible (no opposite signs), return infinite
        return 10**9
    # minimise for the contact angles
    phi_1_1, phi_1_2, phi_2_1, phi_2_2 = root_contact_phase_angles_radii(e, w, i, r_sum_sma)
    # minimise for the internal tangency angles
    r_dif_sma = abs(r_sum_sma * (1 - r_ratio) / (1 + r_ratio))
    psi_1_1, psi_1_2, psi_2_1, psi_2_2 = root_contact_phase_angles_radii(e, w, i, r_dif_sma)
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
    # sb_ratio = sb_ratio_from_d_ratio((d_2 / d_1), e, w, i, r_sum_sma, r_ratio, theta_1, theta_2)
    phases_min = np.array([theta_1, theta_2])
    depth_1, depth_2 = eclipse_depth(phases_min, e, w, i, r_sum_sma, r_ratio, sb_ratio, theta_3, theta_4)
    # displacement of the minima, linearly sensitive to e cos(w) (and sensitive to i)
    r_displacement = ((t_2 - t_1) - disp) / np.sqrt(t_1_err**2 + t_2_err**2)
    # difference in duration of the minima, linearly sensitive to e sin(w) (and sensitive to i and phi_0)
    tau_err_tot = np.sqrt(t_1_1_err**2 + t_1_2_err**2 + t_2_1_err**2 + t_2_2_err**2)
    r_duration_dif = ((tau_1_1 + tau_1_2 - tau_2_1 - tau_2_2) - (dur_1 - dur_2)) / tau_err_tot
    # sum of durations of the minima, linearly sensitive to phi_0 (and sensitive to e sin(w), i and phi_0)
    r_duration_sum = ((tau_1_1 + tau_1_2 + tau_2_1 + tau_2_2) - (dur_1 + dur_2)) / tau_err_tot
    # depths
    r_depth_dif = ((d_1 - d_2) - (depth_1 - depth_2)) / np.sqrt(d_1_err**2 + d_2_err**2)
    r_depth_sum = ((d_1 + d_2) - (depth_1 + depth_2)) / np.sqrt(d_1_err**2 + d_2_err**2)
    # durations of the (flat) bottoms of the minima
    r_b_duration_sum = ((tau_b_1_1 + tau_b_1_2 + tau_b_2_1 + tau_b_2_2) - (bottom_dur_1 + bottom_dur_2)) / tau_err_tot
    r_b_duration_dif = ((tau_b_1_1 + tau_b_1_2 - tau_b_2_1 - tau_b_2_2) - (bottom_dur_1 - bottom_dur_2)) / tau_err_tot
    # calculate the error-normalised residuals
    resid = np.array([r_displacement, r_duration_dif, r_duration_sum, r_depth_dif, r_depth_sum, r_b_duration_dif,
                      r_b_duration_sum])
    likelihood = -tsf.calc_likelihood(resid)  # minus for minimisation
    return likelihood


def objective_ecl_param(params, p_orb, timings_tau, depths, timings_err, depths_err):
    """Minimise this function to obtain an estimate of all eclipse parameters
    
    Parameters
    ----------
    params: array-like[float]
        ecosw, esinw, cosi, phi_0, log_rr, log_sb
    p_orb: float
        Orbital period of the eclipsing binary in days
    timings_tau: numpy.ndarray[float]
        Eclipse timings of minima, durations from/to first/last contact,
        and durations from/to first/last internal tangency:
        t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2,
        tau_b_1_1, tau_b_1_2, tau_b_2_1, tau_b_2_2
    depths: numpy.ndarray[float]
        Eclipse depth of the primary and secondary, depth_1, depth_2
    timings_err: numpy.ndarray[float]
        Error estimates for the eclipse timings,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err
        t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err
    depths_err: numpy.ndarray[float]
        Error estimates for the depths

    Returns
    -------
    likelihood: float
        Minus the likelihood with weighted sum of squared deviations of
        five outcomes of integrals of Kepler's second law for different
        time spans of the eclipses compared to the measured values.
    """
    ecosw, esinw, cosi, phi_0, log_rr, log_sb = params
    e = np.sqrt(ecosw**2 + esinw**2)
    w = np.arctan2(esinw, ecosw) % (2 * np.pi)
    i = np.arccos(cosi)
    r_sum_sma = r_sum_sma_from_phi_0(e, i, phi_0)
    r_ratio = 10**log_rr
    sb_ratio = 10**log_sb
    t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2 = timings_tau[:6]
    tau_b_1_1, tau_b_1_2, tau_b_2_1, tau_b_2_2 = timings_tau[6:]
    d_1, d_2 = depths
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err = timings_err[:6]
    t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err = timings_err[6:]
    d_1_err, d_2_err = depths_err
    if (e >= 1):
        return 10**9  # we want to stay in orbit
    # minimise for the phases of minima (theta)
    theta_1, theta_2, theta_3, theta_4 = minima_phase_angles(e, w, i)
    if (theta_2 == 0):
        # if no solution is possible (no opposite signs), return infinite
        return 10**9
    # minimise for the contact angles
    phi_1_1, phi_1_2, phi_2_1, phi_2_2 = root_contact_phase_angles_radii(e, w, i, r_sum_sma)
    # minimise for the internal tangency angles
    r_dif_sma = abs(r_sum_sma * (1 - r_ratio) / (1 + r_ratio))
    psi_1_1, psi_1_2, psi_2_1, psi_2_2 = root_contact_phase_angles_radii(e, w, i, r_dif_sma)
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
    phases_min = np.array([theta_1, theta_2])
    depth_1, depth_2 = eclipse_depth(phases_min, e, w, i, r_sum_sma, r_ratio, sb_ratio, theta_3, theta_4)
    # displacement of the minima, linearly sensitive to e cos(w) (and sensitive to i)
    r_displacement = ((t_2 - t_1) - disp) / np.sqrt(t_1_err**2 + t_2_err**2)
    # difference in duration of the minima, linearly sensitive to e sin(w) (and sensitive to i and phi_0)
    tau_err_tot = np.sqrt(t_1_1_err**2 + t_1_2_err**2 + t_2_1_err**2 + t_2_2_err**2)
    r_duration_dif = ((tau_1_1 + tau_1_2 - tau_2_1 - tau_2_2) - (dur_1 - dur_2)) / tau_err_tot
    # sum of durations of the minima, linearly sensitive to phi_0 (and sensitive to e sin(w), i and phi_0)
    r_duration_sum = ((tau_1_1 + tau_1_2 + tau_2_1 + tau_2_2) - (dur_1 + dur_2)) / tau_err_tot
    # depths
    r_depth_dif = ((d_1 - d_2) - (depth_1 - depth_2)) / np.sqrt(d_1_err**2 + d_2_err**2)
    r_depth_sum = ((d_1 + d_2) - (depth_1 + depth_2)) / np.sqrt(d_1_err**2 + d_2_err**2)
    # durations of the (flat) bottoms of the minima
    r_b_duration_sum = ((tau_b_1_1 + tau_b_1_2 + tau_b_2_1 + tau_b_2_2) - (bottom_dur_1 + bottom_dur_2)) / tau_err_tot
    r_b_duration_dif = ((tau_b_1_1 + tau_b_1_2 - tau_b_2_1 - tau_b_2_2) - (bottom_dur_1 - bottom_dur_2)) / tau_err_tot
    # calculate the error-normalised residuals
    resid = np.array([r_displacement, r_duration_dif, r_duration_sum, r_depth_dif, r_depth_sum, r_b_duration_dif,
                      r_b_duration_sum])
    likelihood = -tsf.calc_likelihood(resid)  # minus for minimisation
    return likelihood


def eclipse_parameters(p_orb, timings_tau, depths, timings_err, depths_err, verbose=False):
    """Determine all eclipse parameters using a combination of approximate
    functions and fitting procedures
    
    Parameters
    ----------
    p_orb: float
        The orbital period
    timings_tau: numpy.ndarray[float]
        Eclipse timings of minima, durations from/to first/last contact,
        and durations from/to first/last internal tangency:
        t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2,
        tau_b_1_1, tau_b_1_2, tau_b_2_1, tau_b_2_2
    depths: numpy.ndarray[float]
        Eclipse depth of the primary and secondary, depth_1, depth_2
    timings_err: numpy.ndarray[float]
        Error estimates for the eclipse timings,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err
        t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err
    depths_err: numpy.ndarray[float]
        Error estimates for the depths
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    ecosw: float
        Eccentricity times cosine of omega
    esinw: float
        Eccentricity times sine of omega
    cosi: float
        Cosine of the inclination of the orbit
    phi_0: float
        Auxiliary angle, see Kopal 1959
    log_rr: float
        Logarithm of the radius ratio r_2/r_1
    log_sb: float
        Logarithm of the surface brightness ratio sb_2/sb_1
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
    sb_ratio: float
        Surface brightness ratio sb_2/sb_1
    
    Notes
    -----
    phi_0: Auxiliary angle (see Kopal 1959)
    psi_0: Auxiliary angle like phi_0 but for the eclipse bottoms
    """
    # use mix of approximate and exact formulae iteratively to get a value for i
    args_a = (p_orb, timings_tau, depths, timings_err, depths_err)
    bounds_a = (0, 1)
    result_a = sp.optimize.minimize_scalar(objective_inclination, args=args_a, method='bounded', bounds=bounds_a)
    cosi = result_a.x
    i = np.arccos(cosi)
    # calculation phi_0, in durations: (duration_1 + duration_2)/4 = (2pi/P)(tau_1_1 + tau_1_2 + tau_2_1 + tau_2_2)/4
    phi_0 = np.pi * (timings_tau[2] + timings_tau[3] + timings_tau[4] + timings_tau[5]) / (2 * p_orb)
    # psi_0 is like phi_0 but for the eclipse bottom
    psi_0 = np.pi * (timings_tau[6] + timings_tau[7] + timings_tau[8] + timings_tau[9]) / (2 * p_orb)
    # values of e and w by approximate formulae
    e, w = ecc_omega_approx(p_orb, *timings_tau[:6], i, phi_0)
    # value for r_sum_sma from ecc, incl and phi_0
    r_sum_sma = r_sum_sma_from_phi_0(e, i, phi_0)
    # value for |r1 - r2|/a = r_dif_sma from ecc, incl and psi_0
    r_dif_sma = r_sum_sma_from_phi_0(e, i, psi_0)
    # r_dif_sma only valid if psi_0 is not zero, otherwise it will give limits on the radii
    r_small = (r_sum_sma - r_dif_sma) / 2  # if psi_0=0, this is a lower limit on the smaller radius
    r_large = (r_sum_sma + r_dif_sma) / 2  # if psi_0=0, this is an upper limit on the bigger radius
    if (r_small == 0) | (r_large == 0):
        log_rr_bounds = (-3.0, 3.0)
    else:
        log_rr_bounds = (np.log10(r_small / r_large / 1.1), np.log10(r_large / r_small * 1.1))
    # lower bound on cosi needs to be loose, upper bound can be more restrictive
    cosi_bounds = (0, min(cosi * 1.2, 1))
    # bounds for phi_0 can be fairly tight as the measurement is robust
    phi_0_bounds = (phi_0 / 1.1, min(phi_0 * 1.1, 1))
    # fit globally for: cosi, phi_0, r_ratio and sb_ratio
    ecosw, esinw = e * np.cos(w), e * np.sin(w)
    args_b = (p_orb, ecosw, esinw, timings_tau, depths, timings_err, depths_err)
    bounds_b = (cosi_bounds, phi_0_bounds, log_rr_bounds, (-3, 3))
    result_b = sp.optimize.shgo(objective_incl_plus, args=args_b, bounds=bounds_b)
    # local fit of: ecosw, esinw, cosi, phi_0, r_ratio and sb_ratio
    par_init_c = np.append([ecosw, esinw], result_b.x)
    args_c = (p_orb, timings_tau, depths, timings_err, depths_err)
    bounds_c = ((-1, 1), (-1, 1), cosi_bounds, phi_0_bounds, log_rr_bounds, (-3, 3))
    result_c = sp.optimize.minimize(objective_ecl_param, par_init_c, args=args_c, method='L-BFGS-B', bounds=bounds_c)
    ecosw, esinw, cosi, phi_0, log_rr, log_sb = result_c.x
    e = np.sqrt(ecosw**2 + esinw**2)
    w = np.arctan2(esinw, ecosw) % (2 * np.pi)
    i = np.arccos(cosi)
    r_sum_sma = r_sum_sma_from_phi_0(e, i, phi_0)
    r_ratio = 10**log_rr
    sb_ratio = 10**log_sb
    if verbose:
        print(f'Fit convergence: {result_c.success} - BIC: {result_c.fun:1.2f}. N_iter: {int(result_c.nit)}.')
    return ecosw, esinw, cosi, phi_0, log_rr, log_sb, e, w, i, r_sum_sma, r_ratio, sb_ratio


@nb.njit(cache=True)
def formal_uncertainties(e, w, i, p_orb, t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2, p_err, i_err,
                         t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err):
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
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_1: float
        Time of primary minimum in domain [0, p_orb)
        and t_1 < t_2
    t_2: float
        Time of secondary minimum in domain [0, p_orb)
    tau_1_1: float
        Duration of primary first contact to minimum
    tau_1_2: float
        Duration of primary minimum to last contact
    tau_2_1: float
        Duration of secondary first contact to minimum
    tau_2_2: float
        Duration of secondary minimum to last contact
    p_err: float
        Error in the orbital period (used as error in the times of minima)
    i_err: float
        Error in the orbital inclination
    t_1_err: float
        Error in t_1 (the time of primary minimum)
    t_2_err: float
        Error in t_1 (the time of secondary minimum)
    t_1_1_err: float
        Error in tau_1_1 (or in the time of primary first contact)
    t_1_2_err: float
        Error in tau_1_2 (or in the time of primary last contact)
    t_2_1_err: float
        Error in tau_2_1 (or in the time of secondary first contact)
    t_2_2_err: float
        Error in tau_2_2 (or in the time of secondary last contact)

    Returns
    -------
    sigma_e: float
        Formal error in eccentricity
    sigma_w: float
        Formal error in argument of periastron
    sigma_phi_0: float
        Formal error in auxiliary angle
    sigma_r_sum_sma: float
        Formal error in sum of radii in units of the semi-major axis
    sigma_ecosw: float
        Formal error in e*cos(w)
    sigma_esinw: float
        Formal error in e*sin(w)
    """
    # often used errors/parameter
    phi_0 = np.pi * (tau_1_1 + tau_1_2 + tau_2_1 + tau_2_2) / (2 * p_orb)
    p_err_2 = p_err**2
    sum_t_err_2 = t_1_1_err**2 + t_1_2_err**2 + t_2_1_err**2 + t_2_2_err**2
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
    if e == 0:
        e = 10**-9
    e_2 = e**2
    term_2_i_phi0 = (1 - sin_i_2 * cos_phi0_2)  # another term with i and phi_0 in it
    # error in phi_0
    s_phi0_p = (np.pi * (tau_1_1 + tau_1_2 + tau_2_1 + tau_2_2) / (2 * p_orb**2))**2 * p_err_2
    s_phi0_tau = (np.pi / (2 * p_orb))**2 * sum_t_err_2
    sigma_phi_0 = np.sqrt(s_phi0_p + s_phi0_tau)
    # error in e*cos(w)
    s_ecosw_p = (np.pi * (t_2 - t_1) / p_orb**2 * sin_i_2 / term_i)**2 * p_err_2
    s_ecosw_t = (np.pi / p_orb * sin_i_2 / term_i)**2 * (t_1_err**2 + t_2_err**2)
    s_ecosw_i = (2 * sin_i * cos_i / term_i**2)**2 * i_err**2
    sigma_ecosw = np.sqrt(s_ecosw_p + s_ecosw_t + s_ecosw_i)
    # error in e*sin(w)
    s_esinw_p = (np.pi * (tau_1_1 + tau_1_2 - tau_2_1 - tau_2_2) / (2 * sin_phi0 * p_orb**2)
                 * sin_i_2 * cos_phi0_2 / term_1_i_phi0)**2 * p_err_2
    s_esinw_tau = (np.pi / (2 * sin_phi0 * p_orb) * sin_i_2 * cos_phi0_2 / term_1_i_phi0)**2 * sum_t_err_2
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
    # sigma_f_c = np.sqrt(cos_w**2 / (4 * e) * sigma_e**2 + e * sin_w**2 * sigma_w**2)
    # sigma_f_s = np.sqrt(sin_w**2 / (4 * e) * sigma_e**2 + e * cos_w**2 * sigma_w**2)
    return sigma_e, sigma_w, sigma_phi_0, sigma_r_sum_sma, sigma_ecosw, sigma_esinw


def error_estimates_hdi(e, w, i, r_sum, r_rat, sb_rat, ecosw, esinw, cosi, phi_0, log_rr, log_sb, p_orb,
                        timings, depths, p_err, timings_err, depths_err, p_t_corr, verbose=False):
    """Estimate errors using importance sampling and
    the highest density interval (HDI)
    
    Parameters
    ----------
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
    ecosw: float
        Eccentricity times cosine of omega
    esinw: float
        Eccentricity times sine of omega
    cosi: float
        Cosine of the inclination of the orbit
    phi_0: float
        Auxiliary angle, see Kopal 1959
    log_rr: float
        Logarithm of the radius ratio r_2/r_1
    log_sb: float
        Logarithm of the surface brightness ratio sb_2/sb_1
    p_orb: float
        Orbital period of the eclipsing binary in days
    timings: numpy.ndarray[float]
        Eclipse timings of minima and first and last contact points,
        Timings of the possible flat bottom (internal tangency),
        t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2
        t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2
    depths: numpy.ndarray[float]
        Primary and secondary eclipse depth
    p_err: float
        Error in the orbital period
    timings_err: numpy.ndarray[float]
        Error estimates for the eclipse timings,
        t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err,
        t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err
    depths_err: numpy.ndarray[float]
        Error estimates for the depths
    p_t_corr: float
        Correlation between orbital period and t_zero/t_1/t_2
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    intervals: tuple[numpy.ndarray[float]]
        The HDIs (hdi_prob=0.683) for the parameters:
        e, w, i, r_sum, r_rat, sb_rat, ecosw, esinw, cosi, phi_0, log_rr, log_sb
    errors: tuple[numpy.ndarray[float]]
        The (non-symmetric) errors for the same parameters as intervals.
        These are computed from the intervals.
    dists_in: tuple[numpy.ndarray[float]]
        Full input distributions for: p, t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2,
        t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, d_1, d_2
    dists_out: tuple[numpy.ndarray[float]]
        Full output distributions for the same parameters as intervals
    
    Notes
    -----
    The HDI is the minimum width Bayesian credible interval (BCI).
    https://arviz-devs.github.io/arviz/api/generated/arviz.hdi.html
    The interval for w can consist of two disjunct intervals due to the
    degeneracy between angles around 90 degrees and 270 degrees.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings
    depth_1, depth_2 = depths
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err = timings_err[:6]
    t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err = timings_err[6:]
    depth_1_err, depth_2_err = depths_err
    # generate input distributions
    rng = np.random.default_rng()
    n_gen = 10**3  # 10**4
    # variance matrix of t_1, t_2 and period
    var_matrix = np.zeros((3, 3))
    var_matrix[0, 0] = t_1_err**2
    var_matrix[1, 1] = t_2_err**2
    var_matrix[2, 2] = p_err**2
    var_matrix[0, 2] = p_t_corr * t_1_err * p_err
    var_matrix[2, 0] = var_matrix[0, 2]
    var_matrix[1, 2] = p_t_corr * t_2_err * p_err
    var_matrix[2, 1] = var_matrix[1, 2]
    mvn = rng.multivariate_normal([t_1, t_2, p_orb], var_matrix, 1000)
    normal_t_1 = mvn[:, 0]
    normal_t_2 = mvn[:, 1]
    normal_p = mvn[:, 2]
    # the edges cannot surpass the midpoints so use truncnorm
    normal_t_1_1 = sp.stats.truncnorm.rvs(((normal_t_2 - p_orb) - t_1_1) / t_1_1_err, (normal_t_1 - t_1_1) / t_1_1_err,
                                          loc=t_1_1, scale=t_1_1_err, size=n_gen)
    normal_t_1_2 = sp.stats.truncnorm.rvs((normal_t_1 - t_1_2) / t_1_2_err, (normal_t_2 - t_1_2) / t_1_2_err,
                                          loc=t_1_2, scale=t_1_2_err, size=n_gen)
    normal_t_2_1 = sp.stats.truncnorm.rvs((normal_t_1 - t_2_1) / t_2_1_err, (normal_t_2 - t_2_1) / t_2_1_err,
                                          loc=t_2_1, scale=t_2_1_err, size=n_gen)
    normal_t_2_2 = sp.stats.truncnorm.rvs((normal_t_2 - t_2_2) / t_2_2_err, (normal_t_1 + p_orb - t_2_2) / t_2_2_err,
                                          loc=t_2_2, scale=t_2_2_err, size=n_gen)
    # if we have wide eclipses, they possibly overlap, fix by putting point in the middle
    overlap_1_2 = (normal_t_1_2 > normal_t_2_1)
    if np.any(overlap_1_2):
        middle = (normal_t_1_2[overlap_1_2] + normal_t_2_1[overlap_1_2]) / 2
        normal_t_1_2[overlap_1_2] = middle
        normal_t_2_1[overlap_1_2] = middle
    overlap_2_1 = (normal_t_2_2 > normal_t_1_1 + p_orb)
    if np.any(overlap_2_1):
        middle = (normal_t_1_1[overlap_2_1] + p_orb + normal_t_2_2[overlap_2_1]) / 2
        normal_t_1_1[overlap_2_1] = middle - p_orb
        normal_t_2_2[overlap_2_1] = middle
    # the bottom points are truncated at the edge points
    normal_t_b_1_1 = sp.stats.truncnorm.rvs((normal_t_1_1 - t_b_1_1) / t_b_1_1_err,
                                            (normal_t_1_2 - t_b_1_1) / t_b_1_1_err,
                                            loc=t_b_1_1, scale=t_b_1_1_err, size=n_gen)
    normal_t_b_1_2 = sp.stats.truncnorm.rvs((normal_t_1_1 - t_b_1_2) / t_b_1_2_err,
                                            (normal_t_1_2 - t_b_1_2) / t_b_1_2_err,
                                            loc=t_b_1_2, scale=t_b_1_2_err, size=n_gen)
    normal_t_b_2_1 = sp.stats.truncnorm.rvs((normal_t_2_1 - t_b_2_1) / t_b_2_1_err,
                                            (normal_t_2_2 - t_b_2_1) / t_b_2_1_err,
                                            loc=t_b_2_1, scale=t_b_2_1_err, size=n_gen)
    normal_t_b_2_2 = sp.stats.truncnorm.rvs((normal_t_2_1 - t_b_2_2) / t_b_2_2_err,
                                            (normal_t_2_2 - t_b_2_2) / t_b_2_2_err,
                                            loc=t_b_2_2, scale=t_b_2_2_err, size=n_gen)
    # likely to overlap, fixed by putting them in the middle (this is more physical than truncating in the middle)
    overlap_b_1 = (normal_t_b_1_1 > normal_t_b_1_2)
    if np.any(overlap_b_1):
        middle = (normal_t_b_1_1[overlap_b_1] + normal_t_b_1_2[overlap_b_1]) / 2
        normal_t_b_1_1[overlap_b_1] = middle
        normal_t_b_1_2[overlap_b_1] = middle
    overlap_b_2 = (normal_t_b_2_1 > normal_t_b_2_2)
    if np.any(overlap_b_2):
        middle = (normal_t_b_2_1[overlap_b_2] + normal_t_b_2_2[overlap_b_2]) / 2
        normal_t_b_2_1[overlap_b_2] = middle
        normal_t_b_2_2[overlap_b_2] = middle
    # calculate the tau
    normal_tau_1_1 = normal_t_1 - normal_t_1_1
    normal_tau_1_2 = normal_t_1_2 - normal_t_1
    normal_tau_2_1 = normal_t_2 - normal_t_2_1
    normal_tau_2_2 = normal_t_2_2 - normal_t_2
    normal_tau_b_1_1 = normal_t_1 - normal_t_b_1_1
    normal_tau_b_1_2 = normal_t_b_1_2 - normal_t_1
    normal_tau_b_2_1 = normal_t_2 - normal_t_b_2_1
    normal_tau_b_2_2 = normal_t_b_2_2 - normal_t_2
    # depths are truncated at zero and upper limit of five sigma
    normal_d_1 = sp.stats.truncnorm.rvs((0 - depth_1) / depth_1_err, 5, loc=depth_1, scale=depth_1_err, size=n_gen)
    normal_d_2 = sp.stats.truncnorm.rvs((0 - depth_2) / depth_2_err, 5, loc=depth_2, scale=depth_2_err, size=n_gen)
    # determine the output distributions
    ecosw_vals = np.zeros(n_gen)
    esinw_vals = np.zeros(n_gen)
    cosi_vals = np.zeros(n_gen)
    phi_0_vals = np.zeros(n_gen)
    log_rr_vals = np.zeros(n_gen)
    log_sb_vals = np.zeros(n_gen)
    e_vals = np.zeros(n_gen)
    w_vals = np.zeros(n_gen)
    i_vals = np.zeros(n_gen)
    r_sum_vals = np.zeros(n_gen)
    r_rat_vals = np.zeros(n_gen)
    sb_rat_vals = np.zeros(n_gen)
    i_delete = []  # to be deleted due to out of bounds parameter
    for k in range(n_gen):
        timings_tau_dist = (normal_t_1[k], normal_t_2[k],
                            normal_tau_1_1[k], normal_tau_1_2[k], normal_tau_2_1[k], normal_tau_2_2[k],
                            normal_tau_b_1_1[k], normal_tau_b_1_2[k], normal_tau_b_2_1[k], normal_tau_b_2_2[k])
        # if sum of tau happens to be larger than p_orb, skip and delete
        if (np.sum(timings_tau_dist[2:6]) > normal_p[k]) | (normal_d_1[k] < 0) | (normal_d_2[k] < 0):
            i_delete.append(k)
            continue
        depths_k = np.array([normal_d_1[k], normal_d_2[k]])
        out = eclipse_parameters(normal_p[k], timings_tau_dist, depths_k, timings_err, depths_err, verbose=False)
        ecosw_vals[k] = out[0]
        esinw_vals[k] = out[1]
        cosi_vals[k] = out[2]
        phi_0_vals[k] = out[3]
        log_rr_vals[k] = out[4]
        log_sb_vals[k] = out[5]
        e_vals[k] = out[6]
        w_vals[k] = out[7]
        i_vals[k] = out[8]
        r_sum_vals[k] = out[9]
        r_rat_vals[k] = out[10]
        sb_rat_vals[k] = out[11]
        if verbose & (k % 50 == 0):
            print(f'Parameter calculations {int(k / (n_gen) * 100)}% done', end='\r')
    if verbose:
        print(f'Parameter calculations 100% done')
    # delete the skipped parameters
    normal_p = np.delete(normal_p, i_delete)
    normal_t_1 = np.delete(normal_t_1, i_delete)
    normal_t_2 = np.delete(normal_t_2, i_delete)
    normal_t_1_1 = np.delete(normal_t_1_1, i_delete)
    normal_t_1_2 = np.delete(normal_t_1_2, i_delete)
    normal_t_2_1 = np.delete(normal_t_2_1, i_delete)
    normal_t_2_2 = np.delete(normal_t_2_2, i_delete)
    normal_t_b_1_1 = np.delete(normal_t_b_1_1, i_delete)
    normal_t_b_1_2 = np.delete(normal_t_b_1_2, i_delete)
    normal_t_b_2_1 = np.delete(normal_t_b_2_1, i_delete)
    normal_t_b_2_2 = np.delete(normal_t_b_2_2, i_delete)
    normal_d_1 = np.delete(normal_d_1, i_delete)
    normal_d_2 = np.delete(normal_d_2, i_delete)
    ecosw_vals = np.delete(ecosw_vals, i_delete)
    esinw_vals = np.delete(esinw_vals, i_delete)
    cosi_vals = np.delete(cosi_vals, i_delete)
    phi_0_vals = np.delete(phi_0_vals, i_delete)
    log_rr_vals = np.delete(log_rr_vals, i_delete)
    log_sb_vals = np.delete(log_sb_vals, i_delete)
    e_vals = np.delete(e_vals, i_delete)
    w_vals = np.delete(w_vals, i_delete)
    i_vals = np.delete(i_vals, i_delete)
    r_sum_vals = np.delete(r_sum_vals, i_delete)
    r_rat_vals = np.delete(r_rat_vals, i_delete)
    sb_rat_vals = np.delete(sb_rat_vals, i_delete)
    # Calculate the highest density interval (HDI) for a given probability.
    # e cos(w)
    ecosw_interval = az.hdi(ecosw_vals, hdi_prob=0.683)
    ecosw_bounds = az.hdi(ecosw_vals, hdi_prob=0.997)
    ecosw_err = np.array([ecosw - ecosw_interval[0], ecosw_interval[1] - ecosw])
    # e sin(w)
    esinw_interval = az.hdi(esinw_vals, hdi_prob=0.683)
    esinw_bounds = az.hdi(esinw_vals, hdi_prob=0.997)
    esinw_err = np.array([esinw - esinw_interval[0], esinw_interval[1] - esinw])
    # cos(i)
    cosi_interval = az.hdi(cosi_vals, hdi_prob=0.683)
    cosi_bounds = az.hdi(cosi_vals, hdi_prob=0.997)
    cosi_err = np.array([cosi - cosi_interval[0], cosi_interval[1] - cosi])
    # phi_0
    phi_0_interval = az.hdi(phi_0_vals, hdi_prob=0.683)
    phi_0_bounds = az.hdi(phi_0_vals, hdi_prob=0.997)
    phi_0_err = np.array([phi_0 - phi_0_interval[0], phi_0_interval[1] - phi_0])
    # log_rr
    log_rr_interval = az.hdi(log_rr_vals, hdi_prob=0.683)
    log_rr_bounds = az.hdi(log_rr_vals, hdi_prob=0.997)
    log_rr_err = np.array([log_rr - log_rr_interval[0], log_rr_interval[1] - log_rr])
    # log_sb
    log_sb_interval = az.hdi(log_sb_vals, hdi_prob=0.683)
    log_sb_bounds = az.hdi(log_sb_vals, hdi_prob=0.997)
    log_sb_err = np.array([log_sb - log_sb_interval[0], log_sb_interval[1] - log_sb])
    # eccentricity
    e_interval = az.hdi(e_vals, hdi_prob=0.683)
    e_bounds = az.hdi(e_vals, hdi_prob=0.997)
    e_err = np.array([e - e_interval[0], e_interval[1] - e])
    # omega
    if (abs(w / np.pi * 180 - 180) > 80) & (abs(w / np.pi * 180 - 180) < 100):
        w_interval = az.hdi(w_vals, hdi_prob=0.683, multimodal=True)
        w_bounds = az.hdi(w_vals, hdi_prob=0.997, multimodal=True)
    else:
        w_interval = az.hdi(w_vals - np.pi, hdi_prob=0.683, circular=True) + np.pi
        w_bounds = az.hdi(w_vals - np.pi, hdi_prob=0.997, circular=True) + np.pi
    w_inter, w_inter_2 = ut.bounds_multiplicity_check(w_interval, w)
    # w_bds, w_bds_2 = ut.bounds_multiplicity_check(w_bounds, w)
    w_err = np.array([w - w_inter[0], (w_inter[1] - w) % (2 * np.pi)])  # %2pi for if w_inter wrapped around
    # inclination
    i_interval = az.hdi(i_vals, hdi_prob=0.683)
    i_bounds = az.hdi(i_vals, hdi_prob=0.997)
    i_err = np.array([i - i_interval[0], i_interval[1] - i])
    # r_sum_sma
    r_sum_interval = az.hdi(r_sum_vals, hdi_prob=0.683)
    r_sum_bounds = az.hdi(r_sum_vals, hdi_prob=0.997)
    r_sum_err = np.array([r_sum - r_sum_interval[0], r_sum_interval[1] - r_sum])
    # r_ratio
    r_rat_interval = az.hdi(r_rat_vals, hdi_prob=0.683)
    r_rat_bounds = az.hdi(r_rat_vals, hdi_prob=0.997)
    r_rat_err = np.array([r_rat - r_rat_interval[0], r_rat_interval[1] - r_rat])
    # sb_ratio
    sb_rat_interval = az.hdi(sb_rat_vals, hdi_prob=0.683)
    sb_rat_bounds = az.hdi(sb_rat_vals, hdi_prob=0.997)
    sb_rat_err = np.array([sb_rat - sb_rat_interval[0], sb_rat_interval[1] - sb_rat])
    # collect
    intervals = (e_interval, w_interval, i_interval, r_sum_interval, r_rat_interval, sb_rat_interval,
                 ecosw_interval, esinw_interval, cosi_interval, phi_0_interval, log_rr_interval, log_sb_interval)
    bounds = (e_bounds, w_bounds, i_bounds, r_sum_bounds, r_rat_bounds, sb_rat_bounds,
              ecosw_bounds, esinw_bounds, cosi_bounds, phi_0_bounds, log_rr_bounds, log_sb_bounds)
    errors = (e_err, w_err, i_err, r_sum_err, r_rat_err, sb_rat_err,
              ecosw_err, esinw_err, cosi_err, phi_0_err, log_rr_err, log_sb_err)
    dists_in = (normal_p, normal_t_1, normal_t_2, normal_t_1_1, normal_t_1_2, normal_t_2_1, normal_t_2_2,
                normal_t_b_1_1, normal_t_b_1_2, normal_t_b_2_1, normal_t_b_2_2, normal_d_1, normal_d_2)
    dists_out = (e_vals, w_vals, i_vals, r_sum_vals, r_rat_vals, sb_rat_vals)
    return intervals, errors, dists_in, dists_out
