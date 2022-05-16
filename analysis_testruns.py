"""Script for testing general analysis of binary pulsators"""

import os
import time
import fnmatch

import numpy as np
import scipy as sp
import scipy.stats
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
import astropy.io as aio
import itertools as iter
import ellc
import importlib as ilib

import eclipsr as ecl
import main_functions as mf
import timeseries_functions as tsf
import timeseries_fitting as tsfit
import analysis_functions as af
import utility as ut
import visualisation as vis


# laptop
tic_dir = '/lhome/lijspeert/data/TIC'
target_dir = '/lhome/lijspeert/data/TIC_targets'
lc_dir = '/lhome/lijspeert/data/TESS_lc_downloads/mastDownload'
lc_dir_hlsp = '/lhome/lijspeert/data/TESS_lc_downloads/mastDownload/HLSP'
analysis_dir = '/lhome/lijspeert/data/TESS_lc_downloads/analysis'
test_dir = '/lhome/lijspeert/data/test_data/prewhitening'


# binning folded eclipse signal - shift the bins
# testing on EB from sample - load in catalogue
EB_catalogue = pd.read_csv(os.path.join(analysis_dir, 'catalogue', 'EB_catalogue_var_cor.csv'))
# get the files
all_files = []
for root, dirs, files in os.walk(lc_dir_hlsp):
    for file in files:
        all_files.append(os.path.join(root, file))

spoc_files = []
for file in all_files:
    if 'spoc' in file:
        spoc_files.append(file)
qlp_files = []
for file in all_files:
    if 'qlp' in file:
        qlp_files.append(file)

tic_numbers_spoc, n_sectors_spoc = np.loadtxt('/lhome/lijspeert/data/TESS_lc_downloads/analysis/n_sectors_spoc.dat',
                                              unpack=True, dtype=int)
tic_numbers_qlp, n_sectors_qlp = np.loadtxt('/lhome/lijspeert/data/TESS_lc_downloads/analysis/n_sectors_qlp.dat',
                                            unpack=True, dtype=int)
spoc_in_cat = [tic in EB_catalogue['TIC_ID'].values for tic in tic_numbers_spoc]
qlp_in_cat = [tic in EB_catalogue['TIC_ID'].values for tic in tic_numbers_qlp]


print(tic_numbers_spoc[spoc_in_cat][n_sectors_spoc[spoc_in_cat] > 10])
tic = 269696438
period = float(EB_catalogue['eclipse_period'][EB_catalogue['TIC_ID'] == tic])
p_orb = period
t_0 = float(EB_catalogue['t_supcon'][EB_catalogue['TIC_ID'] == tic])
# other period(s)
# EB_catalogue_refined = pd.read_csv(os.path.join(analysis_dir, 'catalogue', 'EB_catalogue_refined_p.csv'))
p_profile =  2.9790051063382035  # float(EB_catalogue_refined['eclipse_period'][EB_catalogue['TIC_ID'] == tic])
p_profile_e = 1.0698330040970204e-05  # float(EB_catalogue_refined['p_interval_right'][EB_catalogue['TIC_ID'] == tic])
p_pdm = 2.979014958393902
# get the SPOC data
times, sap_signal, signal, signal_err, sectors, t_sectors, crowd = ut.load_tess_lc(tic, spoc_files, apply_flags=True)
i_sectors = ut.convert_tess_t_sectors(times, t_sectors)
times, signal, sector_medians, t_start, t_combined, i_half_s = ut.stitch_tess_sectors(times, signal, i_sectors)
t_0 -= t_start
folded_t = ((times - t_0) / period) % 1
sort_fold = np.argsort(folded_t)
sorted_folded_t = folded_t[sort_fold]
sorted_folded_s = signal[sort_fold]
n_dpoints = len(times)
if (n_dpoints / 10 > 1000):
    n_bins = 1000
else:
    n_bins = n_dpoints // 10  # at least 10 data points per bin on average
    

def phase_dispersion(phases, signal, n_bins):
    """Phase dispersion, as in PDM, without overlapping bins."""
    def var_no_avg(a):
        return np.sum(np.abs(a - np.mean(a))**2)  # if mean instead of sum, this is variance
    bins = np.linspace(-0.5, 0.5, n_bins + 1)
    binned_var, edges, number = sp.stats.binned_statistic(phases, signal, statistic=var_no_avg, bins=bins)
    total_var = np.sum(binned_var) / len(signal)
    overall_var = np.var(signal)
    return total_var / overall_var


def phase_dispersion_minimisation(times, signal, p_start, p_stop, n_p, t_0, n_bins):
    """"""
    periods = np.linspace(p_start, p_stop, n_p)
    pd_all = np.zeros(len(periods))
    for i, p in enumerate(periods):
        fold = tsf.fold_time_series(times, p, t_0)
        pd_all[i] = phase_dispersion(fold, signal, n_bins)
    return pd_all


# new one based on the prewhitened frequencies
def phase_dispersion_minimisation(times, signal, f_n):
    """"""
    # number of bins for dispersion calculation
    n_dpoints = len(times)
    if (n_dpoints / 10 > 1000):
        n_bins = 1000
    else:
        n_bins = n_dpoints // 10  # at least 10 data points per bin on average
    # determine where to look based on the frequencies
    periods = np.zeros(7 * len(f_n))
    for i, f in enumerate(f_n):
        periods[7*i:7*i+7] = np.arange(1, 8) / f
    # stay below the maximum
    p_max = np.ptp(times)
    periods = periods[periods < p_max]
    # compute the dispersion measures
    pd_all = np.zeros(len(periods))
    for i, p in enumerate(periods):
        fold = tsf.fold_time_series(times, p, 0)
        pd_all[i] = phase_dispersion(fold, signal, n_bins)
    return periods, pd_all



fold_1 = tsf.fold_time_series(times, period, t_0)
pd_1 = phase_dispersion(fold_1, signal, n_bins)
fold_2 = tsf.fold_time_series(times, p_profile, t_0)
pd_2 = phase_dispersion(fold_2, signal, n_bins)
fold_3 = tsf.fold_time_series(times, p_avg_harm, t_0)
pd_3 = phase_dispersion(fold_3, signal, n_bins)
fold_4 = tsf.fold_time_series(times, 1/wavg_f_orb, t_0)
pd_4 = phase_dispersion(fold_4, signal, n_bins)
p_start = np.min([period, p_profile, p_avg_harm, 1 / wavg_f_orb]) - 0.0001
p_stop = np.max([period, p_profile, p_avg_harm, 1 / wavg_f_orb]) + 0.0001
pd_range = phase_dispersion_minimisation(times, signal, p_start, p_stop, 10000, t_0, n_bins)
plt.scatter(1/np.linspace(p_start, p_stop, 10000), pd_range, c='grey')
plt.scatter([1/period], [pd_1])
plt.scatter([1 / p_profile], [pd_2])
plt.scatter([1 / p_avg_harm], [pd_3])
plt.scatter([wavg_f_orb], [pd_4])
plt.xlabel('orbital frequency')
plt.ylabel('phase dispersion')


# make an average profile
def regularise(times, const, slope, f_n, a_n, ph_n, i_sectors):
    """Use the sum of sines model of a light curve to convert the measurement
    times and model to a regular interval.
    
    Intended for space telescope time series, which are already pretty regular.
    The time series can then be used in a regular wavelet transform.
    """
    med_timestep = np.median(np.diff(times))
    n_steps = int(np.round(np.ptp(times) / med_timestep)) + 1
    reg_times = np.linspace(times[0], times[-1], n_steps)
    reg_model = linear_curve(reg_times, const, slope, i_sectors)  # the linear part of the model
    reg_model += sum_sines(reg_times, f_n, a_n, ph_n)  # the sinusoid part of the model
    return reg_times, reg_model


def build_profile(times, signal, t_0, period):
    """Builds an average eclipse profile given the ephemeris.
    
    Takes 100 equal bins over the phase domain and shifts it 10 times
    within a bin width to build a profile of the eclipse.
    """
    # prepare folded versions of the light curve
    folded_t = ((times - t_0) / period) % 1  # phases in [0, 1]
    sort_fold = np.argsort(folded_t)
    s_f_t = folded_t[sort_fold]  # sorted folded times
    s_f_s = signal[sort_fold]  # sorted folded signal
    # extend these by sticking half the phase interval to both ends
    s_f_t_ext = np.concatenate((s_f_t[s_f_t > 0.5] - 1, s_f_t, s_f_t[s_f_t < 0.5] + 1))
    s_f_s_ext = np.concatenate((s_f_s[s_f_t > 0.5], s_f_s, s_f_s[s_f_t < 0.5]))
    # set up binning process
    n_bins = 100
    n_shifts = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_diff = bins[1] - bins[0]
    # extend the bins by 1 on both ends to catch excess points left and right
    bins_ext = np.concatenate(([bins[0] - bin_diff], bins, [bins[-1] + bin_diff]))
    bin_mid = np.array([])
    profile = np.array([])
    shifts = np.linspace(-bin_diff * (1 - 1/n_shifts), bin_diff * (1 - 1/n_shifts), n_shifts)
    for i in range(n_shifts):
        binned_s, edges, number = sp.stats.binned_statistic(s_f_t_ext, s_f_s_ext, statistic='mean',
                                                            bins=bins_ext + shifts[i])
        # remove the two extra outer bins
        bin_mid = np.append(bin_mid, (edges[1:-1][1:] + edges[1:-1][:-1]) / 2)
        profile = np.append(profile, binned_s[1:-1])
    # remove the tail ends that overshoot interval [0, 1]
    in_interval = (bin_mid >= 0) & (bin_mid <= 1)
    bin_mid = bin_mid[in_interval]
    profile = profile[in_interval]
    # sort the profile
    sorter = np.argsort(bin_mid)
    bin_mid = bin_mid[sorter]
    profile = profile[sorter]
    return bin_mid, profile

bins = np.linspace(0, 1, n_bins + 1)
# numpy.digitize(x, bins, right=False)
binned_s, edges, number = sp.stats.binned_statistic(sorted_folded_t, sorted_folded_s, statistic='mean', bins=bins)
plt.scatter((bins[1:] + bins[:-1]) / 2, binned_s, marker='.')
# try by shifting bins
bin_mid, profile = build_profile(times, signal, t_0, period)
plt.scatter(bin_mid, profile, marker='.')
# remove from lc
interp_s = np.interp(folded_t, bin_mid, profile)
reduced_s = signal - interp_s
# look at freq spectrum
freqs_s, ampls_s = tsf.scargle(times, signal - np.mean(signal))
freqs_r, ampls_r = tsf.scargle(times, reduced_s)
fig, ax = plt.subplots()
ax.plot(freqs_s, ampls_s)
ax.plot(freqs_r, ampls_r)
plt.tight_layout()
plt.show()
# take out the rest of the frequencies
const, f_n, a_n, ph_n = tsf.extract_all(times, reduced_s)
model = tsf.sum_sines(times, const, f_n, a_n, ph_n)
const_full, f_n_full, a_n_full, ph_n_full = tsf.extract_all(times, signal)
model_full = tsf.sum_sines(times, const_full, f_n_full, a_n_full, ph_n_full)
const_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, signal - model_full, a_n_full)
plt.scatter(times, signal, marker='.')
plt.scatter(times, model + interp_s, marker='.')
plt.scatter(times, model_full, marker='.')
# compare freqs to see harmonics?
test_int = f_n * period
test_int_full = f_n_full * period
f_profile_e = p_profile_e / p_profile**2
f_harm, n_harm = tsf.find_harmonics_from_pattern(f_n, period)
f_harm_full, n_harm_full = tsf.find_harmonics_from_pattern(f_n_full, period)
fig, ax = plt.subplots(figsize=(16, 5))
for i in range(1, np.ceil(np.max(test_int_full)).astype(int)):
    ax.plot([i, i], [-0.4, 0.5], '--', c='tab:green')
    # ax.plot([i+0.05, i+0.05], [-0.4, 0.5], '--', c='tab:pink')
    # ax.plot([i-0.05, i-0.05], [-0.4, 0.5], '--', c='tab:pink')
ax.scatter(test_int, np.zeros(len(f_n))+0.1, label='profile subtracted')
ax.scatter(test_int[f_harm], np.zeros(len(test_int[f_harm]))+0.1, label='harmonics')
ax.scatter(test_int_full, np.zeros(len(f_n_full)), label='full lc')
ax.scatter(test_int_full[f_harm_full], np.zeros(len(f_n_full[f_harm_full])), label='harmonics')
# ax.scatter(f_n * p_2, np.zeros(len(f_n))+0.1, label='p_2')
# ax.scatter(f_n_full * p_2, np.zeros(len(f_n_full)), label='p_2')
plt.legend()
plt.tight_layout()
plt.show()
# now improve period from harmonics
f_orb = f_n[f_harm] / n_harm
f_orb_full = f_n_full[f_harm_full] / n_harm_full
f_orb_err_full = f_n_err[f_harm_full] / n_harm_full
avg_f_orb = np.average(f_orb_full)
wavg_f_orb = np.average(f_orb_full, weights=1/f_orb_err_full**2)
std_f_orb = np.std(f_orb_full)
err_f_orb = np.sqrt(np.sum(f_orb_err_full**2)) / len(f_orb_err_full)
werr_f_orb = np.sqrt(np.sum(1/f_orb_err_full**2 / len(f_orb_err_full)**2)) / np.sum(1/f_orb_err_full**2/ len(f_orb_err_full))
p_avg_harm = 1 / avg_f_orb
p_avg_harm_err = std_f_orb * p_avg_harm**2  # same as p * f_err / f
period_unc_init = 0.00152628
fig, ax = plt.subplots(figsize=(9, 8))
ax.errorbar(f_orb_full, n_harm_full, xerr=f_orb_err_full, marker='o', c='grey')
ax.errorbar([1/period, 1/period], [-0.6, np.max(n_harm_full)+0.6], xerr=[period_unc_init, period_unc_init],
            fmt='--', label='initial period and error estimate')
ax.errorbar([1 / p_profile, 1 / p_profile], [0.3, np.max(n_harm_full) - 0.3], xerr=[f_profile_e, f_profile_e],
            fmt='--', label='averaged eclipse profile period')
ax.errorbar([1 / p_avg_harm, 1 / p_avg_harm], [0, np.max(n_harm_full)], xerr=[std_f_orb, std_f_orb],
            fmt='--', label='average harmonic and standard deviation')
ax.errorbar([1 / p_avg_harm, 1 / p_avg_harm], [-0.3, np.max(n_harm_full) + 0.3], xerr=[err_f_orb, err_f_orb],
            fmt='--', label='average harmonic and propagated error')
ax.errorbar([wavg_f_orb, wavg_f_orb], [0.6, np.max(n_harm_full)-0.6], xerr=[werr_f_orb, werr_f_orb],
            fmt='--', label='w average harmonic and propagated w error')
ax.errorbar([1/p_pdm, 1/p_pdm], [0.9, np.max(n_harm_full)-0.9], xerr=[0, 0],
            fmt='--', label='PDM')
plt.xlabel('frequency (1/d)', fontsize=14)
plt.ylabel('harmonic', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
# plot models - residuals are indistinguishable
model_harm = tsf.sum_sines(times, const_full, f_n_full[f_harm_full], a_n_full[f_harm_full], ph_n_full[f_harm_full])
fig, ax = plt.subplots(figsize=(15, 10), nrows=3, sharex=True)
ax[0].scatter(times, signal, marker='.', label='signal')
ax[0].scatter(times, model_full, marker='.', label='sum of sines')
ax[0].scatter(times, model + interp_s, marker='.', label='profile plus sines')
ax[0].legend()
ax[1].scatter(times, model_harm, marker='.', label='harmonics only')
ax[1].scatter(times, interp_s, marker='.', label='eclipse profile')
ax[1].legend()
ax[2].scatter(times, signal - model_full, marker='.', label='residual of sum of sines')
ax[2].scatter(times, signal - model - interp_s, marker='.', label='residual of profile plus sines')
ax[2].legend()
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()
# plot of folded signals
fold_orig = tsf.fold_time_series(times, period, t_0)
fold_p_2 = tsf.fold_time_series(times, p_profile, t_0)
fold_p_harm = tsf.fold_time_series(times, p_avg_harm, t_0)
fold_p_harm_w = tsf.fold_time_series(times, 1/wavg_f_orb, t_0)
fig, ax = plt.subplots()
ax.scatter(fold_orig, signal-0.002, label='original period')
ax.scatter(fold_p_2, signal, label='eclipse profile')
ax.scatter(fold_p_harm, signal+0.002, label='averaged harmonic')
ax.scatter(fold_p_harm_w, signal+0.004, label='w averaged harmonic')
plt.xlabel('phase', fontsize=14)
plt.ylabel('signal (offset)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
# plot periodograms of residuals
model_harm = tsf.sum_sines(times, const_full, f_n_full[f_harm_full], a_n_full[f_harm_full], ph_n_full[f_harm_full])
model_harm_2 = tsf.sum_sines(times, const_full, wavg_f_orb * n_harm_full, a_n_full[f_harm_full], ph_n_full[f_harm_full])
ph_new = tsf.scargle_phase(times, signal - model, wavg_f_orb * n_harm_full)
model_harm_3 = tsf.sum_sines(times, const_full, wavg_f_orb * n_harm_full, a_n_full[f_harm_full], ph_new)
freqs1, ampls1 = tsf.scargle(times, signal - model_full)
freqs2, ampls2 = tsf.scargle(times, signal - model - model_harm_2)
# freqs3, ampls3 = tsf.scargle(times, signal - model - model_harm_3)
fig, ax = plt.subplots()
ax.plot(freqs1, ampls1, label='full model')
ax.plot(freqs2, ampls2, alpha=0.5, label='model plus harmonics')
# ax.plot(freqs2, ampls3, alpha=0.5, label='model plus harmonics 2')
plt.xlabel('frequency (1/d)', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()


# refining period with this method of eclipse profile making
def refine_period(times, signal, t_0, period, plot=False):
    """Refine the period of an eclipsing binary system.
    
    Uses phase folding by the initial period to make an average profile,
    then shifts the period and looks at the residuals of the newly folded
    light curve minus the profile.
    Returns the best period, credible interval and best BIC value.
    The credible interval is based on a delta BIC value of 10.
    """
    # build the averaged profile
    bin_mid, profile = build_profile(times, signal, t_0, period)
    # remove nan values in the profile
    finites = np.isfinite(profile)
    bin_mid = bin_mid[finites]
    profile = profile[finites]
    # initial rough sweep
    n_points = 1000
    delta_bic = 10  # delta BIC to use for the credible interval
    bics_1 = np.zeros(n_points)
    periods_1 = period + np.linspace(-0.001*period, 0.001*period, n_points)
    for i in range(n_points):
        folded_t_i = ((times - t_0) / periods_1[i]) % 1  # fold to the new period
        interp_s_i = np.interp(folded_t_i, bin_mid, profile)  # interpolate the profile
        bics_1[i] = calc_bic(signal - interp_s_i, len(profile))
    # best period found in round one
    best_i = np.argmin(bics_1)
    best_p = periods_1[best_i]
    # rough interval to refine the search in round two
    p_interval = periods_1[(bics_1 - bics_1[best_i]) < delta_bic]
    if (len(p_interval) == 1):
        delta_p = np.ptp(periods_1) / n_points
    else:
        delta_p = max(abs(p_interval[[0, -1]] - best_p))
    # round two
    bics_2 = np.zeros(n_points)
    periods_2 = best_p + np.linspace(-2 * delta_p, 2 * delta_p, n_points)
    for i in range(n_points):
        folded_t_i = ((times - t_0) / periods_2[i]) % 1
        interp_s_i = np.interp(folded_t_i, bin_mid, profile)
        bics_2[i] = calc_bic(signal - interp_s_i, len(profile))
    best_i_2 = np.argmin(bics_2)
    best_p_2 = periods_2[best_i_2]
    best_bic_2 = bics_2[best_i_2]
    p_interval_2 = periods_2[(bics_2 - bics_2[best_i_2]) < delta_bic][[0, -1]]
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(periods_1, bics_1)
        ax.scatter(periods_2, bics_2)
        ax.plot(p_interval_2, [bics_2[best_i_2], bics_2[best_i_2]], marker='|', label='interval p')
        ax.scatter(best_p_2, bics_2[best_i_2], label='best p')
        ax.scatter(period, bics_2[best_i_2], label='initial p')
        plt.legend()
        plt.show()
    return best_p_2, p_interval_2, best_bic_2


best_p, p_interval, bic = refine_period(times, signal, t_0, period, plot=True)
folded_t = tsf.fold_time_series(times, period, t_0)
folded_t_2 = tsf.fold_time_series(times, best_p, t_0)
plt.scatter(folded_t, signal, marker='.')
plt.scatter(folded_t_2, signal, marker='.')

"""x -> orbital period ; y -> min and max frequencies extracted
x -> orbital period ; y -> frequency of highest amplitude ; color -> Teff
x -> orbital period ; y -> frequency of highest amplitude ; color -> depth eclipse 1 / depth eclipse 2
x -> frequency ;      y -> amplitude, but order all of them in terms of ascending orbital period
"""


# 10 random light curves for Cole to look for 'sub harmonic' (0.5 orbital frequency)
rng = np.random.default_rng(seed=11211)
print(EB_catalogue['TIC_ID'].values[rng.integers(0, len(EB_catalogue['TIC_ID'].values), 10)])


# fixing orbital freq to one value and redetermining a and ph for harmonics
harmonics, harmonic_n = tsf.find_harmonics_from_pattern(f_n_full, 1 / wavg_f_orb)
const_h, f_n_h, a_n_h, ph_n_h = tsf.fix_harmonic_frequency(times, signal, 1 / wavg_f_orb, const_full, f_n_full,
                                                           a_n_full, ph_n_full, i_half_s)
model_h = tsf.sum_sines(times, const_h, f_n_h, a_n_h, ph_n_h)
fig, ax = plt.subplots(figsize=(15, 10), nrows=3, sharex=True)
ax[0].scatter(times, signal, marker='.', label='signal')
ax[0].scatter(times, model_full, marker='.', label='harmonics not fixed')
ax[0].scatter(times, model_h, marker='.', label='harmonics fixed')
ax[0].legend()
ax[1].scatter(times, tsf.sum_sines(times, const_full, f_n_full[harmonics], a_n_full[harmonics], ph_n_full[harmonics]),
              marker='.', label='harmonics (not fixed)')
ax[1].scatter(times, tsf.sum_sines(times, const_h, f_n_h[harmonics], a_n_h[harmonics], ph_n_h[harmonics]),
              marker='.', label='harmonics (fixed)')
ax[1].legend()
ax[2].scatter(times, signal - model_full, marker='.', label='residual with harmonics not fixed')
ax[2].scatter(times, signal - model_h, marker='.', label='residual with harmonics fixed')
ax[2].legend()
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()
# bigger difference between the models and the data than between the two models
print(tsf.calc_bic(signal - model_full, 3 * len(f_n_full) + 1))
print(tsf.calc_bic(signal - model_h, 3 * len(f_n_h) + 1),
      tsf.calc_bic(signal - model_h, 3 * len(f_n_h) + 1 - (len(harmonics) - 1)))
# may need to change other freqs too though
freqs1, ampls1 = tsf.scargle(times, signal - model_full)
freqs2, ampls2 = tsf.scargle(times, signal - model_h)
fig, ax = plt.subplots()
ax.plot(freqs_s, ampls_s, label='signal')
ax.plot(freqs1, ampls1, label='full model')
ax.plot(freqs2, ampls2, alpha=0.5, label='model plus harmonics')
plt.xlabel('frequency (1/d)', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
# not too bad actually - doesn't even get better by re-doing the other freqs
# now try to extract more harmonics
const_h2, f_n_h2, a_n_h2, ph_n_h2 = tsf.extract_additional_harmonics(times, signal, 1/wavg_f_orb,
                                                                     const_h, f_n_h, a_n_h, ph_n_h)
model_h2 = tsf.sum_sines(times, const_h2, f_n_h2, a_n_h2, ph_n_h2)
harmonics_2 = np.append(harmonics, np.arange(len(f_n_h), len(f_n_h2)))
model_t = np.linspace(times[0], times[-1], len(times) * 10)
fig, ax = plt.subplots(figsize=(15, 10), nrows=3, sharex=True)
ax[0].scatter(times, signal, marker='.', label='signal')
ax[0].scatter(times, model_full, marker='.', label='harmonics not fixed')
ax[0].scatter(times, model_h, marker='.', label='harmonics fixed')
ax[0].scatter(times, model_h2, marker='.', label='additional harmonics fixed')
ax[0].legend()
ax[1].plot(model_t, tsf.sum_sines(model_t, const_full, f_n_full[harmonics], a_n_full[harmonics], ph_n_full[harmonics]),
              label='harmonics (not fixed)')
ax[1].plot(model_t, tsf.sum_sines(model_t, const_h, f_n_h[harmonics], a_n_h[harmonics], ph_n_h[harmonics]),
              label='harmonics (fixed)')
ax[1].plot(model_t, tsf.sum_sines(model_t, const_h2, f_n_h2[harmonics_2], a_n_h2[harmonics_2], ph_n_h2[harmonics_2]),
              label='additional harmonics (fixed)')
ax[1].legend()
ax[2].scatter(times, signal - model_full, marker='.', label='residual with harmonics not fixed')
ax[2].scatter(times, signal - model_h, marker='.', label='residual with harmonics fixed')
ax[2].scatter(times, signal - model_h2, marker='.', label='residual with additional harmonics fixed')
ax[2].legend()
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()
# bigger difference between the models and the data than between the two models
print(tsf.calc_bic(signal - model_h, 3 * len(f_n_h) + 1 - (len(harmonics) - 1)))
print(tsf.calc_bic(signal - model_h2, 3 * len(f_n_h) + 1 - (len(harmonics) - 1) + 2 * (len(f_n_h2) - len(f_n_h))))
# may need to change other freqs too though
freqs1, ampls1 = tsf.scargle(times, signal - model_h)
freqs2, ampls2 = tsf.scargle(times, signal - model_h2)
fig, ax = plt.subplots()
ax.plot(freqs_s, ampls_s, label='signal')
ax.plot(freqs1, ampls1, label='full model')
ax.plot(freqs2, ampls2, alpha=0.5, label='model plus harmonics')
for i in range(len(f_n_h), len(f_n_h2)):
    ax.plot([f_n_h2[i], f_n_h2[i]], [0, a_n_h2[i]], '--', c='grey')
    ax.annotate(f'{i+1}', (f_n_h2[i], a_n_h2[i]))
plt.xlabel('frequency (1/d)', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

# emcee for frequencies?
import os
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

import timeseries_functions as tsf

test_dir = '/lhome/lijspeert/data/test_data/prewhitening'
times1, signal1 = np.loadtxt(os.path.join(test_dir, '1_output_noisy.txt'), unpack=True)
times2, signal2 = np.loadtxt(os.path.join(test_dir, '2_output_noisy.txt'), unpack=True)
times3, signal3 = np.loadtxt(os.path.join(test_dir, '3_output_noisy.txt'), unpack=True)
n, f_n, a_n, ph_n = np.loadtxt(os.path.join(test_dir, '3_output_noisy_freqs.txt'), unpack=True)
times = times3 - times3[0]
signal = signal3
const = np.mean(signal)
slope = 0
resid = signal - tsf.sum_sines(times, const, slope, f_n, a_n, ph_n)
const_err, f_err, a_err, ph_err = tsf.formal_uncertainties(times, resid, a_n)


def calc_log_gaussian(x, mu, sigma):
    """Natural logarithm of a gaussian probability density function.
    All inputs must have the same shape (or be floats).
    
    logP(x) = log[1/(√(2π)σ)] − (x−μ)^2/2σ^2
    mu and sigma are pre-determined, fixed values here.
    """
    factor = np.log((np.sqrt(2 * np.pi) * sigma)**-1)
    power = -(x - mu)**2 / (2 * sigma**2)
    return factor + power


def calc_log_probability(theta, times, signal, mu, sigma):
    """Natural logarithm of the prior times the likelihood.
    
    Intended for use in MCMC (like the emcee package)
    The parameters (theta) have to be a flat array and are ordered in the following way:
    params = array(constant, slope, freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...)
    See sum_sines for the definition of these parameters.
    mu and sigma should be structured identically, with sigma containing the error estimates.
    """
    # calculate the log prior by using gaussians around the determined values
    log_prior = calc_log_gaussian(theta, mu, 100*sigma)
    log_prior = np.sum(log_prior)
    # 'unpack' the parameters
    n_sines = (len(theta) - 1) // 3  # each sine has freq, ampl and phase, and there is one constant
    const = theta[0]
    slope = theta[1]
    freqs = theta[2:n_sines + 2]
    ampls = np.abs(theta[n_sines + 2:2 * n_sines + 2])  # make sure non negative
    phases = np.mod(theta[2 * n_sines + 2:3 * n_sines + 2] + np.pi, 2 * np.pi) - np.pi  # make sure cyclic
    # make the model and determine the log likelihood
    model = tsf.sum_sines(times, const, slope, freqs, ampls, phases)
    log_likelihood = tsf.calc_likelihood(signal - model)
    return log_prior + log_likelihood


true_params = np.concatenate(([0], [0], freqs_ref, amps_ref, phases_ref))
params = np.concatenate(([const], [slope], f_n, a_n, ph_n))
params_err = np.concatenate(([const_err], [const_err], f_err, a_err, ph_err))
nwalkers, ndim = max(32, 3*len(params)), len(params)
# to initialise walkers, draw from a normal distribution with scale smaller than the estimated errors
rng = np.random.default_rng()
init_pos = rng.normal(loc=params, scale=params_err/2, size=(nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, calc_log_probability, args=(times, signal, params, params_err),
                                moves=emcee.moves.DEMove())
sampler.run_mcmc(init_pos, 10000, progress=True)

samples = sampler.get_chain()
fig, ax = plt.subplots(nrows=5, figsize=(10, 7), sharex=True)
labels = ['const', 'slope', 'f1', 'f2', 'f3']
for i in range(5):
    ax[i].plot(samples[:, :, i], "k", alpha=0.3)
    ax[i].set_xlim(0, len(samples))
    ax[i].set_ylabel(labels[i])
    ax[i].yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")

tau = sampler.get_autocorr_time()
print(tau)
flat_samples = sampler.get_chain(discard=100, thin=20, flat=True)
flat_samples = sampler.get_chain(discard=100, flat=True)
print(flat_samples.shape)

fig = corner.corner(flat_samples[:, :8], truths=true_params[:8])#, labels=labels)
# get the mcmc endresults
mcmc_par = np.zeros(len(params))
mcmc_err = np.zeros((len(params), 2))
for i in range(ndim):
    percentiles = np.percentile(flat_samples[:, i], [16, 50, 84])
    mcmc_par[i] = percentiles[1]
    mcmc_err[i] = np.diff(percentiles)

fig, ax = plt.subplots(nrows=3, figsize=(8, 12))
ax[0].errorbar(n, f_n, yerr=f_err, marker='_', linestyle='none', label='extraction')
ax[0].errorbar(n, mcmc_par[2:18], yerr=np.max(mcmc_err[2:18], axis=1), marker='_', linestyle='none', label='mcmc')
ax[0].scatter(n, freqs_ref, marker='.', label='reference')
ax[0].legend()
ax[0].set_ylabel('frequency (1/d)')
ax[1].errorbar(n, a_n, yerr=a_err, marker='_', linestyle='none', label='extraction')
ax[1].errorbar(n, mcmc_par[18:34], yerr=np.max(mcmc_err[18:34], axis=1), marker='_', linestyle='none', label='mcmc')
ax[1].scatter(n, amps_ref / 1000, marker='.', label='reference')
ax[1].set_ylabel('amplitude')
ax[2].errorbar(n, ph_n, yerr=ph_err, marker='_', linestyle='none', label='extraction')
ax[2].errorbar(n, mcmc_par[34:], yerr=np.max(mcmc_err[34:], axis=1), marker='_', linestyle='none', label='mcmc')
ax[2].scatter(n, phases_ref + np.pi/2, marker='.', label='reference')
ax[2].set_ylabel('phase')
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(nrows=3, figsize=(8, 12))
ax[0].errorbar(n, f_n - freqs_ref, yerr=f_err, marker='_', linestyle='none', label='extraction')
ax[0].errorbar(n, mcmc_par[2:18] - freqs_ref, yerr=np.max(mcmc_err[2:18], axis=1), marker='_', linestyle='none', label='mcmc')
ax[0].scatter(n, freqs_ref - freqs_ref, marker='.', label='reference')
ax[0].legend()
ax[0].set_ylabel('frequency (1/d)')
ax[1].errorbar(n, a_n - amps_ref / 1000, yerr=a_err, marker='_', linestyle='none', label='extraction')
ax[1].errorbar(n, mcmc_par[18:34] - amps_ref / 1000, yerr=np.max(mcmc_err[18:34], axis=1), marker='_', linestyle='none', label='mcmc')
ax[1].scatter(n, amps_ref / 1000 - amps_ref / 1000, marker='.', label='reference')
ax[1].set_ylabel('amplitude')
ax[2].errorbar(n, ph_n - phases_ref + np.pi/2, yerr=ph_err, marker='_', linestyle='none', label='extraction')
ax[2].errorbar(n, mcmc_par[34:] - phases_ref + np.pi/2, yerr=np.max(mcmc_err[34:], axis=1), marker='_', linestyle='none', label='mcmc')
ax[2].scatter(n, phases_ref + np.pi/2 - phases_ref + np.pi/2, marker='.', label='reference')
ax[2].set_ylabel('phase')
plt.tight_layout()
plt.show()

n_sines = (len(mcmc_par) - 1) // 3  # each sine has freq, ampl and phase, and there is one constant
mcmc_const = mcmc_par[0]
mcmc_slope = mcmc_par[1]
freqs = mcmc_par[2:n_sines + 2]
ampls = mcmc_par[n_sines + 2:2 * n_sines + 2]
phases = mcmc_par[2 * n_sines + 2:3 * n_sines + 2]
ampls = np.abs(mcmc_par[n_sines + 2:2 * n_sines + 2])  # make sure non negative
phases = np.mod(mcmc_par[2 * n_sines + 2:3 * n_sines + 2] + np.pi, 2 * np.pi) - np.pi  # make sure cyclic
# make the model and determine the log likelihood
mcmc_model = tsf.sum_sines(times, mcmc_const, mcmc_slope, freqs, ampls, phases)
model = tsf.sum_sines(times, const, slope, f_n, a_n, ph_n)
freqs_s, ampls_s = tsf.scargle(times, signal - const)
freqs_r, ampls_r = tsf.scargle(times, signal - model)
freqs_mcmc, ampls_mcmc = tsf.scargle(times, signal - mcmc_model)
fig, ax = plt.subplots(nrows=3, ncols=2, gridspec_kw={'width_ratios': (3, 1)}, figsize=(15, 10))
ax[0, 0].scatter(times, signal, marker='.', label='lc')
ax[0, 0].plot(times, model, c='tab:orange', label='model')
ax[0, 0].plot(times, mcmc_model, c='tab:green', label='mcmc model')
ax[0, 0].set_xlabel('time (d)')
ax[0, 0].legend()
ax[1, 0].scatter(times, signal - model, marker='.', label='residual')
ax[1, 0].scatter(times, signal - mcmc_model, marker='.', label='mcmc residual')
ax[1, 0].set_ylim((np.min(signal - model), np.max(signal - model)))
ax[1, 0].set_xlabel('time (d)')
ax[1, 0].legend()
ax[2, 0].plot(freqs_s, ampls_s, label='original')
ax[2, 0].plot(freqs_r, ampls_r, c='tab:green', label='residual')
ax[2, 0].plot(freqs_mcmc, ampls_mcmc, c='tab:purple', label='mcmc residual')
for i in range(len(f_n)):
    ax[2, 0].plot([f_n[i], f_n[i]], [0, a_n[i]], '--', c='tab:orange')
    # ax[2, 0].annotate(f'{i+1}', (f_n[i], a_n[i]))
    ax[2, 0].plot([freqs[i], freqs[i]], [0, ampls[i]], '--', c='tab:green')
ax[2, 0].set_xlabel('frequency (1/d)')
ax[2, 0].legend()
ax[0, 1].scatter(times[:200], signal[:200], marker='.', label='lc')
ax[0, 1].plot(times[:200], model[:200], c='tab:orange', label='model')
ax[0, 1].plot(times[:200], mcmc_model[:200], c='tab:green', label='mcmc model')
ax[0, 1].set_xlabel('time (d)')
ax[0, 1].legend()
ax[1, 1].scatter(times[:200], signal[:200] - model[:200], marker='.', label='residual')
ax[1, 1].scatter(times[:200], signal[:200] - mcmc_model[:200], marker='.', label='mcmc residual')
ax[1, 1].set_ylim((np.min(signal - model), np.max(signal - model)))
ax[1, 1].set_xlabel('time (d)')
ax[1, 1].legend()
ax[2, 1].plot(freqs_s, ampls_s, label='original')
ax[2, 1].plot(freqs_r, ampls_r, c='tab:green', label='residual')
ax[2, 1].plot(freqs_mcmc, ampls_mcmc, c='tab:purple', label='mcmc residual')
for i in range(len(f_n)):
    ax[2, 1].plot([f_n[i], f_n[i]], [0, a_n[i]], '--', c='tab:orange')
    ax[2, 0].plot([freqs[i], freqs[i]], [0, ampls[i]], '--', c='tab:green')
ax[2, 1].set_xlabel('frequency (1/d)')
ax[2, 1].legend()
plt.tight_layout()
plt.show()


# making more difficult lc
# need to load in the above TESS lc for the times and also the white noise
const, slope, f_n, a_n, ph_n = tsf.extract_all(times, signal)
model = tsf.sum_sines(times, const, slope, f_n, a_n, ph_n)

dx = 0.00282  # 1/d away from peak
f_input, a_input, ph_input = [2.5, 2.5+.5*dx, 2.5-.5*dx], [0.002, 0.0018, 0.0013], [0.2, 0.56, 1.42]
model_in = tsf.sum_sines(times, 0, 0, f_input, a_input, ph_input)
new_lc = signal - model + model_in

const1, slope1, f_n1, a_n1, ph_n1 = tsf.extract_all(times, new_lc)
model1 = tsf.sum_sines(times, const1, slope1, f_n1, a_n1, ph_n1)

fig, ax = plt.subplots(figsize=(15, 10), nrows=2, sharex=True)
ax[0].scatter(times, new_lc, marker='.', label='signal')
ax[0].scatter(times, model1, marker='.', label='model')
ax[0].legend()
ax[1].scatter(times, new_lc - model1, marker='.', label='residual')
ax[1].legend()
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()

freqs, ampls = tsf.scargle(times, new_lc)
freqs1, ampls1 = tsf.scargle(times, new_lc - model1)
c_err, sl_err, f_err, a_err, ph_err = tsf.formal_uncertainties(times, new_lc - model1, a_n1)
fig, ax = plt.subplots()
ax.plot(freqs, ampls, label='signal')
ax.plot(freqs1, ampls1, label='signal')
for i in range(len(f_input)):
    ax.plot([f_input[i], f_input[i]], [0, a_input[i]], '--', c='green')
    ax.annotate(f'{i+1}', (f_input[i], a_input[i]))
for i in range(len(f_n1)):
    ax.errorbar([f_n1[i], f_n1[i]], [0, a_n1[i]], xerr=[0, f_err[i]], yerr=[0, a_err[i]], linestyle=':',
                capsize=2, c='grey')
    ax.annotate(f'{i+1}', (f_n1[i], a_n1[i]))
plt.xlabel('frequency (1/d)', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.tight_layout()
plt.show()
print(np.average(np.abs(f_n1[:3] - f_input) / f_err[:3], weights=1/f_err[:3]**2),
      np.average(np.abs(a_n1[:3] - a_input) / a_err[:3], weights=1/a_err[:3]**2),
      np.average(np.abs(ph_n1[:3] - ph_input) / ph_err[:3], weights=1/ph_err[:3]**2))
# load in Dominics results
i_d1, f_d1, a_d1, a_d_e1, ph_d1, ph_d_e1, snr_d1 = np.loadtxt(os.path.join(test_dir, 'unresolved_results_Dominic',
                                                           'lc_unresolved_test_noisy_LSDoutputs.txt'),
                                                        unpack=True)
i_d2, f_d2, f_d_e2, a_d2, a_d_e2, ph_d2, ph_d_e2, snr_d2 = np.loadtxt(os.path.join(test_dir, 'unresolved_results_Dominic',
                                                                   'lc_unresolved_test_noisy_NLDoutputs.txt'),
                                                                unpack=True)
i_d3, f_d3, f_d_e3, a_d3, a_d_e3, ph_d3, ph_d_e3, snr_d3 = np.loadtxt(os.path.join(test_dir, 'unresolved_results_Dominic',
                                                                   'lc_unresolved_test_noisy_NLDoutputs_multi.txt'),
                                                                unpack=True)
f_d4, f_d_e4, a_d4, a_d_e4, ph_d4, ph_d_e4 = np.loadtxt(os.path.join(test_dir, 'unresolved_results_Dominic',
                                                                   'period04_results.txt'),
                                                        unpack=True, usecols=(1, 2, 3, 4, 5, 6))
# fit using the real values as start point
params_init = np.concatenate(([const1], [slope1], f_n1[a_n1/std > 4], a_n1[a_n1/std > 4], ph_n1[a_n1/std > 4]))
opt_result = sp.optimize.minimize(tsf.objective_sines, x0=params_init, args=(times, new_lc), method='Nelder-Mead')
n_sines = (len(opt_result.x) - 1) // 3
res_const = opt_result.x[0]
res_slope = opt_result.x[1]
res_freqs = opt_result.x[2:n_sines + 2]
res_ampls = opt_result.x[n_sines + 2:2 * n_sines + 2]
res_phases = opt_result.x[2 * n_sines + 2:3 * n_sines + 2]
# plots
model1_d = tsf.sum_sines(times, np.mean(new_lc), 0, f_d4, a_d4, ph_d4*2*np.pi)
freqs1_d, ampls1_d = tsf.scargle(times, new_lc - model1_d)
model3 = tsf.sum_sines(times, res_const, res_slope, res_freqs, res_ampls, res_phases)
freqs3, ampls3 = tsf.scargle(times, new_lc - model3)
std = np.std(new_lc - model1)
fig, ax = plt.subplots()
ax.plot(freqs, ampls, label='signal')
ax.plot(freqs1, ampls1, label='residual Luc')
ax.plot(freqs3, ampls3,  c='tab:purple', label='residual Luc after multi-NL-LS')
ax.plot(freqs1_d, ampls1_d, c='tab:red', label='residual Dominic')
for i in range(len(f_input)):
    ax.plot([f_input[i], f_input[i]], [0, a_input[i]], '--', c='green')
for i in range(len(f_n1)):
    if (a_n1[i]/std > 4):
        ax.errorbar([f_n1[i], f_n1[i]], [0, a_n1[i]], xerr=[0, f_err[i]], yerr=[0, a_err[i]], linestyle='-',
                    capsize=2, c='tab:orange')
    else:
        ax.errorbar([f_n1[i], f_n1[i]], [0, a_n1[i]], xerr=[0, f_err[i]], yerr=[0, a_err[i]], linestyle=':',
                    capsize=2, c='tab:orange')
for i in range(len(f_d4)):
    ax.errorbar([f_d4[i], f_d4[i]], [0, a_d4[i]], xerr=[0, f_d_e4[i]], yerr=[0, a_d_e4[i]], linestyle='-',
                capsize=2, c='tab:red')
for i in range(len(res_freqs)):
    ax.errorbar([res_freqs[i], res_freqs[i]], [0, res_ampls[i]], xerr=[0, 0], yerr=[0, 0], linestyle=':',
                capsize=2, c='tab:purple')
plt.xlabel('frequency (1/d)', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
model2_d = tsf.sum_sines(times, np.mean(new_lc), 0, f_d1, a_d1 / 1000, ph_d1 + np.pi/2)
freqs2_d, ampls2_d = tsf.scargle(times, new_lc - model2_d)
model2 = tsf.sum_sines(times, const2, slope2, f_n2, a_n2, ph_n2)  # calculate without refinement step
freqs2, ampls2 = tsf.scargle(times, new_lc - model2)
c_err2, sl_err2, f_err2, a_err2, ph_err2 = tsf.formal_uncertainties(times, new_lc - model2, a_n2)
fig, ax = plt.subplots()
ax.plot(freqs, ampls, label='signal')
ax.plot(freqs2, ampls2, label='residual Luc')
ax.plot(freqs2_d, ampls2_d, c='tab:red', label='residual Dominic')
for i in range(len(f_input)):
    ax.plot([f_input[i], f_input[i]], [0, a_input[i]], '--', c='green')
for i in range(len(f_n2)):
    if (a_n2[i]/std > 4):
        ax.errorbar([f_n2[i], f_n2[i]], [0, a_n2[i]], xerr=[0, f_err2[i]], yerr=[0, a_err2[i]], linestyle='-',
                    capsize=2, c='tab:orange')
    else:
        ax.errorbar([f_n2[i], f_n2[i]], [0, a_n2[i]], xerr=[0, f_err2[i]], yerr=[0, a_err2[i]], linestyle=':',
                    capsize=2, c='tab:orange')
for i in range(len(f_d1)):
    if (i_d1[i] > 0):
        ax.errorbar([f_d1[i], f_d1[i]], [0, a_d1[i] / 1000], xerr=[0, 0], yerr=[0, a_d_e1[i] / 1000], linestyle='-',
                    capsize=2, c='tab:red')
    else:
        ax.errorbar([f_d1[i], f_d1[i]], [0, a_d1[i] / 1000], xerr=[0, 0], yerr=[0, a_d_e1[i] / 1000], linestyle=':',
                    capsize=2, c='tab:red')
plt.xlabel('frequency (1/d)', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
model3_d = tsf.sum_sines(times, np.mean(new_lc), 0, f_d2, a_d2 / 1000, ph_d2 + np.pi/2)
freqs3_d, ampls3_d = tsf.scargle(times, new_lc - model3_d)
fig, ax = plt.subplots()
ax.plot(freqs, ampls, label='signal')
ax.plot(freqs2, ampls2, label='residual Luc')
ax.plot(freqs3_d, ampls3_d, c='tab:red', label='residual Dominic')
for i in range(len(f_input)):
    ax.plot([f_input[i], f_input[i]], [0, a_input[i]], '--', c='green')
for i in range(len(f_n2)):
    if (a_n2[i]/std > 4):
        ax.errorbar([f_n2[i], f_n2[i]], [0, a_n2[i]], xerr=[0, f_err2[i]], yerr=[0, a_err2[i]], linestyle='-',
                    capsize=2, c='tab:orange')
    else:
        ax.errorbar([f_n2[i], f_n2[i]], [0, a_n2[i]], xerr=[0, f_err2[i]], yerr=[0, a_err2[i]], linestyle=':',
                    capsize=2, c='tab:orange')
for i in range(len(f_d2)):
    if (i_d2[i] > 0):
        ax.errorbar([f_d2[i], f_d2[i]], [0, a_d2[i] / 1000], xerr=[0, f_d_e2[i]], yerr=[0, a_d_e2[i] / 1000], linestyle='-',
                    capsize=2, c='tab:red')
    else:
        ax.errorbar([f_d2[i], f_d2[i]], [0, a_d2[i] / 1000], xerr=[0, f_d_e2[i]], yerr=[0, a_d_e2[i] / 1000], linestyle=':',
                    capsize=2, c='tab:red')
plt.xlabel('frequency (1/d)', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
model4_d = tsf.sum_sines(times, np.mean(new_lc), 0, f_d3, a_d3 / 1000, ph_d3 + np.pi/2)
freqs4_d, ampls4_d = tsf.scargle(times, new_lc - model4_d)
fig, ax = plt.subplots()
ax.plot(freqs, ampls, label='signal')
ax.plot(freqs1, ampls1, label='residual Luc')
ax.plot(freqs4_d, ampls4_d, c='tab:red', label='residual Dominic')
for i in range(len(f_input)):
    ax.plot([f_input[i], f_input[i]], [0, a_input[i]], '--', c='green')
for i in range(len(f_n1)):
    if (a_n1[i]/std > 4):
        ax.errorbar([f_n1[i], f_n1[i]], [0, a_n1[i]], xerr=[0, f_err[i]], yerr=[0, a_err[i]], linestyle='-',
                    capsize=2, c='tab:orange')
    else:
        ax.errorbar([f_n1[i], f_n1[i]], [0, a_n1[i]], xerr=[0, f_err[i]], yerr=[0, a_err[i]], linestyle=':',
                    capsize=2, c='tab:orange')
for i in range(len(f_d3)):
    if (i_d3[i] > 0):
        ax.errorbar([f_d3[i], f_d3[i]], [0, a_d3[i] / 1000], xerr=[0, f_d_e3[i]], yerr=[0, a_d_e3[i] / 1000], linestyle='-',
                    capsize=2, c='tab:red')
    else:
        ax.errorbar([f_d3[i], f_d3[i]], [0, a_d3[i] / 1000], xerr=[0, f_d_e3[i]], yerr=[0, a_d_e3[i] / 1000], linestyle=':',
                    capsize=2, c='tab:red')
plt.xlabel('frequency (1/d)', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
"""The new sub-set-refinement-scheme works a charm for unresolved frequencies!
although less well the more closely spaced frequencies there are

multi-sine-NL-LS fit is still necessary to come as close as possible!
Do not use peaks below the SNR threshold for this fit.

Errors are inadequate for these close frequencies:
adjust by weighted average deviation/error
3 freqs-> 53.8 47.5 127.5
        330.1 261.4 866.3
2 freqs -> 7.4, 9.3, 24.9
2 + 1 -> 10.8 14.2 32.2, 5.5 0.71 10.2
perhaps errors could be weighted based on peak density
2*density**3, 3*density**3 for phases?
"""
# frequency density as multiplier for errors?
def sum_normals(x, mu, sigma):
    """"""
    x = np.ascontiguousarray(np.atleast_1d(x)).reshape(-1, 1)  # reshape to enable the vector product in the sum
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    normal_pdf = np.exp(-(x - mu)**2 / (2 * sigma**2))  # / ((2 * np.pi)**(1/2) * sigma)
    model_normals = np.sum(normal_pdf, axis=1)
    return model_normals
    
fig, ax = plt.subplots()
ax.plot(freqs, ampls, label='signal')
ax.plot(freqs1, ampls1, label='signal')
for i in range(len(f_input)):
    ax.plot([f_input[i], f_input[i]], [0, a_input[i]], '--', c='green')
    ax.annotate(f'{i+1}', (f_input[i], a_input[i]))
for i in range(len(f_n1)):
    ax.errorbar([f_n1[i], f_n1[i]], [0, a_n1[i]], xerr=[0, f_err[i]], yerr=[0, a_err[i]], linestyle=':',
                capsize=2, c='grey')
    ax.annotate(f'{i+1}', (f_n1[i], a_n1[i]))
ax.plot(freqs, (sum_normals(freqs, f_n1, 3/4/np.ptp(times)))/1000)
plt.xlabel('frequency (1/d)', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.tight_layout()
plt.show()

# signal processing
cor_signal = np.zeros(len(signal))
for i, s in enumerate(i_sectors):
    avg = np.average(signal[s[0]:s[1]])
    cor_signal[s[0]:s[1]] = (signal[s[0]:s[1]]-avg) / crowding[i] + avg

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(18, 14))
ax[0].scatter(times, signal)
ax[0].scatter(times, cor_signal)
ax[1].scatter(t_sectors[:, 0]-t_start+14, crowd)
ax[1].scatter(t_sectors[:, 0]-t_start+14, (np.min(apert)/apert)**(1/4) )
ax[2].scatter(t_sectors[:, 0]-t_start+14, apert)
plt.tight_layout()
plt.show()


# extract orbital period from harmonics

def scaled_fraction_of_variance(signal, model, n_param):
    """From Jordan et al. 2021"""
    n_times = len(signal)
    rss = np.sum((signal - model)**2)  # residual sum of squares
    sq_dev = np.sum((signal - np.mean(signal))**2)  # squared deviations from the mean
    f_sv = 1 - (rss / sq_dev * (n_times - 1) / (n_times - n_param))
    return f_sv


""" something not right in f reduction.....
Fit convergence: True. N_iter: 166421. BIC: -170927.04
BIC in fit convergence message above invalid. Actual BIC: -167637.93
Fit with fixed harmonics complete. 139 frequencies, 433 free parameters. BIC: -167637.93
Attempting to reduce the number of frequencies.
Single frequencies removed: 99. BIC= -166306.90
Frequency sets replaced by a single frequency: 1 (2 frequencies). BIC= -167601.49
Frequency sets replaced by just harmonic(s): 0 (0 frequencies). BIC= -167601.49
Reducing frequencies complete. 39 frequencies, 134 free parameters. BIC: -158644.18
"""


"""Tests of fitting
BIC initial extraction: -9866.76

10-15, 5 groups (_11)
Fit convergence: True. N_iter: 20136. BIC: -11070.36
BIC in fit convergence message above invalid. Actual BIC: -10009.54
Fit complete. 60 frequencies, 184 free parameters. BIC: -10009.54
time taken: 36.8 (dB=142.78, 3.87/s)

15-20, 4 groups (_12)
Fit convergence: True. N_iter: 7704. BIC: -11134.71
BIC in fit convergence message above invalid. Actual BIC: -10011.50
Fit complete. 60 frequencies, 184 free parameters. BIC: -10011.50
time taken: 106.5 (dB=144.74, 1.36/s - extraB=1.96)

20-25, 3 groups (_13)
Fit convergence: True. N_iter: 58576. BIC: -10949.35
BIC in fit convergence message above invalid. Actual BIC: -10013.34
Fit complete. 60 frequencies, 184 free parameters. BIC: -10013.34
time taken: 162.2 (dB=146.58, 0.904/s - extraB=1.84)

30-35, 2 groups (_15)
Fit convergence: True. N_iter: 232277. BIC: -10699.73
BIC in fit convergence message above invalid. Actual BIC: -10013.32
Fit complete. 60 frequencies, 184 free parameters. BIC: -10013.32
time taken: 764.2 (dB=146.56, 0.904/s - extraB=-0.02)

60, full fit (_14)
Fit convergence: True. N_iter: 17329661. BIC: -10068.43
Fit complete. 60 frequencies, 184 free parameters. BIC: -10068.43
time taken: 15128.7 (dB=201.67, 0.0133/s - extraB=55.11)

20-25 looks like the best compromise in terms of freq spec, and in BIC and in time progression

10-15, then 30-40
Fit convergence: True. N_iter: 20136. BIC: -11070.36
BIC in fit convergence message above invalid. Actual BIC: -10009.54
Fit convergence: True. N_iter: 426670. BIC: -10730.77
BIC in fit convergence message above invalid. Actual BIC: -10044.36
Fit complete. 60 frequencies, 184 free parameters. BIC: -10044.36
time taken: 843.79

10-15, then 20-25
Fit convergence: True. N_iter: 20136. BIC: -11070.36
BIC in fit convergence message above invalid. Actual BIC: -10009.54
Fit convergence: True. N_iter: 68269. BIC: -10958.95
BIC in fit convergence message above invalid. Actual BIC: -10043.74
Fit complete. 60 frequencies, 184 free parameters. BIC: -10043.74
time taken: 287.70
"""
out_8 = af.measure_eclipses_dt(p_orb_7, f_n_7, a_n_7, ph_n_7, )
t_zero, ecl_min, ecl_mid, depths, widths, ratios = out_8  # deepest eclipse is put first
# measure the first/last contact points
t_1, t_2 = ecl_min[0], ecl_min[1]
t_1_1 = t_1 - (widths[0] / 2)  # time of primary first contact
t_1_2 = t_1 + (widths[0] / 2)  # time of primary last contact
t_2_1 = t_2 - (widths[1] / 2)  # time of secondary first contact
t_2_2 = t_2 + (widths[1] / 2)  # time of secondary last contact
tau_1_1 = ecl_min[0] - t_1_1
tau_1_2 = t_1_2 - ecl_min[0]
tau_2_1 = ecl_min[1] - t_2_1
tau_2_2 = t_2_2 - ecl_min[1]

harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_7, p_orb_7)
t_model = np.arange(-widths[0], p_orb_7 + widths[0], 0.001)
model_h = tsf.sum_sines(t_zero + t_model, f_n_7[harmonics], a_n_7[harmonics], ph_n_7[harmonics])
non_harm = np.delete(np.arange(len(f_n_7)), harmonics)
model_linear = tsf.linear_curve(times, const_7, slope_7, i_sectors)
model_nh = tsf.sum_sines(times, f_n_7[non_harm], a_n_7[non_harm], ph_n_7[non_harm])
model_nh += model_linear
s_minmax = [np.min(signal - model_nh), np.max(signal - model_nh)]
folded = (times - t_zero) % p_orb_7
extend_l = (folded > p_orb_7 - widths[0])
extend_r = (folded < widths[0])
model_deriv_1 = tsf.sum_sines_deriv(t_zero + t_model, f_n_7[harmonics], a_n_7[harmonics], ph_n_7[harmonics], deriv=1)
model_deriv_2 = tsf.sum_sines_deriv(t_zero + t_model, f_n_7[harmonics], a_n_7[harmonics], ph_n_7[harmonics], deriv=2)
model_deriv_3 = tsf.sum_sines_deriv(t_zero + t_model, f_n_7[harmonics], a_n_7[harmonics], ph_n_7[harmonics], deriv=3)
model_deriv_4 = tsf.sum_sines_deriv(t_zero + t_model, f_n_7[harmonics], a_n_7[harmonics], ph_n_7[harmonics], deriv=4)
d_minmax = [np.min(model_deriv_1), np.max(model_deriv_1)]
model_deriv_2 = model_deriv_2 / np.max(model_deriv_2) * d_minmax[1]
model_deriv_3 = model_deriv_3 / np.max(model_deriv_3) * d_minmax[1]
model_deriv_4 = model_deriv_4 / np.max(model_deriv_4) * d_minmax[1]
# plot
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))
ax[0].scatter(folded, signal - model_nh, marker='.', c='tab:blue', label='signal minus non-harmonics')
ax[0].scatter(folded[extend_l] - p_orb_7, signal[extend_l] - model_nh[extend_l], marker='.', c='tab:blue')
ax[0].scatter(folded[extend_r] + p_orb_7, signal[extend_r] - model_nh[extend_r], marker='.', c='tab:blue')
ax[0].plot(t_model, model_h, c='tab:green', label='harmonics')
ax[0].plot(ecl_mid[[0, 0]], s_minmax, '--', c='red', label='eclipse midpoint')
ax[0].plot(ecl_mid[[1, 1]], s_minmax, '--', c='red')
ax[0].plot(ecl_min[[0, 0]], s_minmax, '--', c='purple', label='eclipse minimum')
ax[0].plot(ecl_min[[1, 1]], s_minmax, '--', c='purple')
ax[0].plot(ecl_mid[[0, 0]] - widths[[0, 0]]/2, s_minmax, '--', c='orange',
        label=r'edges $\left(\frac{d^3flux}{dt^3}=0\right)$')
ax[0].plot(ecl_mid[[0, 0]] + widths[[0, 0]]/2, s_minmax, '--', c='orange')
ax[0].plot(ecl_mid[[1, 1]] - widths[[1, 1]]/2, s_minmax, '--', c='orange')
ax[0].plot(ecl_mid[[1, 1]] + widths[[1, 1]]/2, s_minmax, '--', c='orange')
ax[0].plot(t_model[[0, -1]], [0, 0], '--', c='grey')
ax[1].set_xlabel(r'$(time - t_0) mod(P_{orb})$ (d)', fontsize=14)
ax[0].set_ylabel('normalised flux', fontsize=14)
ax[1].plot(t_model, model_deriv_1, label='deriv 1')
ax[1].plot(t_model, model_deriv_2, label='deriv 2')
ax[1].plot(t_model, model_deriv_3, label='deriv 3')
ax[1].plot(t_model, model_deriv_4, label='deriv 4')
ax[1].plot(t_model, model_deriv_1 - model_deriv_3, label='deriv 1-3')
ax[1].plot(ecl_mid[[0, 0]], d_minmax, '--', c='red', label='eclipse midpoint')
ax[1].plot(ecl_mid[[1, 1]], d_minmax, '--', c='red')
ax[1].plot(ecl_min[[0, 0]], d_minmax, '--', c='purple', label='eclipse minimum')
ax[1].plot(ecl_min[[1, 1]], d_minmax, '--', c='purple')
ax[1].plot(ecl_mid[[0, 0]] - widths[[0, 0]]/2, d_minmax, '--', c='orange',
           label=r'edges $\left(\frac{d^3flux}{dt^3}=0\right)$')
ax[1].plot(ecl_mid[[0, 0]] + widths[[0, 0]]/2, d_minmax, '--', c='orange')
ax[1].plot(ecl_mid[[1, 1]] - widths[[1, 1]]/2, d_minmax, '--', c='orange')
ax[1].plot(ecl_mid[[1, 1]] + widths[[1, 1]]/2, d_minmax, '--', c='orange')
ax[1].plot(t_model[[0, -1]], [0, 0], '--', c='grey')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()





# what does the periodogram look like for modulations
f1, a1, ph1 = 3, 1, 0.564
f2, a2, ph2 = 0.45, 0.2, 0.0
normal_s = a1 * np.sin((2 * np.pi * f1 * times) + ph1)
mod_f = a1 * np.sin((2 * np.pi * (f1 + a2 / 8 * np.sin((2 * np.pi * f2 * times) + ph2)) * times) + ph1)
mod_a = (a1 + a2 * np.sin((2 * np.pi * f2 * times) + ph2)) * np.sin((2 * np.pi * f1 * times) + ph1)
mod_ph = a1 * np.sin((2 * np.pi * f1 * times) + (ph1 + a2 * 4 * np.sin((2 * np.pi * f2 * times) + ph2)))
plt.plot(times, normal_s, label='normal')
plt.plot(times, mod_f, label='mod f')
plt.plot(times, mod_a, label='mod a')
plt.plot(times, mod_ph, label='mod ph')
plt.legend()
# periodograms
fs1, as1 = tsf.scargle(times, normal_s)
fs2, as2 = tsf.scargle(times, mod_f)
fs3, as3 = tsf.scargle(times, mod_a)
fs4, as4 = tsf.scargle(times, mod_ph)
plt.plot(fs4, as4, label='mod ph')
plt.plot(fs3, as3, label='mod a')
plt.plot(fs1, as1, label='normal')
plt.plot(fs2, as2, label='mod f')
plt.legend()
# extract... (injected some nearly white noise)
const_m1, f_n_m1, a_n_m1, ph_n_m1 = tsf.extract_all(times, normal_s + (signal - model_h2))
const_m2, f_n_m2, a_n_m2, ph_n_m2 = tsf.extract_all(times, mod_f + (signal - model_h2))
const_m3, f_n_m3, a_n_m3, ph_n_m3 = tsf.extract_all(times, mod_a + (signal - model_h2))
const_m4, f_n_m4, a_n_m4, ph_n_m4 = tsf.extract_all(times, mod_ph + (signal - model_h2))
fig, ax = plt.subplots()
ax.plot(fs4, as4, c='tab:red', label='mod ph')
# ax.plot(fs3, as3, c='tab:green', label='mod a')
# ax.plot(fs1, as1, c='tab:blue', label='normal')
# ax.plot(fs2, as2, c='tab:orange', label='mod f')
# for i in range(len(f_n_m1)):
#     ax.plot([f_n_m1[i], f_n_m1[i]], [0, a_n_m1[i]], '--', c='tab:blue')
#     ax.annotate(f'{i+1}', (f_n_m1[i], a_n_m1[i]))
# for i in range(len(f_n_m2)):
#     ax.plot([f_n_m2[i], f_n_m2[i]], [0, a_n_m2[i]], '--', c='tab:orange')
#     ax.annotate(f'{i+1}', (f_n_m2[i], a_n_m2[i]))
# for i in range(len(f_n_m3)):
#     ax.plot([f_n_m3[i], f_n_m3[i]], [0, a_n_m3[i]], '--', c='tab:green')
#     ax.annotate(f'{i+1}', (f_n_m3[i], a_n_m3[i]))
for i in range(len(f_n_m4)):
    ax.plot([f_n_m4[i], f_n_m4[i]], [0, a_n_m4[i]], '--', c='tab:red')
    ax.annotate(f'{i+1}', (f_n_m4[i], a_n_m4[i]))
plt.legend()
plt.show()
# search for a good candidate test subject
for tic in EB_catalogue.loc[EB_catalogue['flag_variability'] == 1, 'TIC_ID']:
    times, sap_signal, signal, signal_err, sectors, t_sectors, crowd = ut.load_tess_lc(tic, spoc_files, apply_flags=True)
    if (len(times) != 0):
        i_sectors = ut.convert_tess_t_sectors(times, t_sectors)
        times, signal, sector_medians, t_start, t_combined = ut.stitch_tess_sectors(times, signal, i_sectors)
    else:
        times = np.zeros(2)
        signal = np.zeros(2)
    times2, sap_signal2, signal2, signal_err2, sectors2, t_sectors2, crowd2 = ut.load_tess_lc(tic, qlp_files, apply_flags=True)
    if (len(times2) != 0):
        i_sectors2 = ut.convert_tess_t_sectors(times2, t_sectors2)
        times2, signal2, sector_medians2, t_start2, t_combined2 = ut.stitch_tess_sectors(times2, signal2, i_sectors2)
    else:
        times2 = np.zeros(2)
        signal2 = np.zeros(2)
    if (np.ptp(times) > 60) | (np.ptp(times2) > 60):
        print(tic)
        t_0 = EB_catalogue.loc[EB_catalogue['TIC_ID'] == tic, 't_supcon']
        p_orb = EB_catalogue.loc[EB_catalogue['TIC_ID'] == tic, 'eclipse_period']
        fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(14, 10),
                               gridspec_kw={'width_ratios': (3, 1)})
        ax[0, 0].scatter(times, signal)
        ax[0, 1].scatter(tsf.fold_time_series(times, float(p_orb), float(t_0)), signal)
        ax[1, 0].scatter(times2, signal2)
        ax[1, 1].scatter(tsf.fold_time_series(times2, float(p_orb), float(t_0)), signal2)
        plt.tight_layout()
        plt.show(block=True)
"""
178370122, 123887853 (short), 442918617, 427312569
178739533  tidally excited? 
[178370122, 123887853, 442918617, 427312569, 178739533]
"""
tic = 178739533
p_orb = float(EB_catalogue['eclipse_period'][EB_catalogue['TIC_ID'] == tic])
t_0 = float(EB_catalogue['t_supcon'][EB_catalogue['TIC_ID'] == tic])
# get the SPOC data
# times, sap_signal, signal, signal_err, sectors, t_sectors, crowd = ut.load_tess_lc(tic, spoc_files, apply_flags=True)
times, sap_signal, signal, signal_err, sectors, t_sectors, crowd = ut.load_tess_lc(tic, qlp_files, apply_flags=True)
i_sectors = ut.convert_tess_t_sectors(times, t_sectors)
times, signal, sector_medians, t_start, t_combined, i_half_s = ut.stitch_tess_sectors(times, signal, i_sectors)
verbose=True
save_dir='/lhome/lijspeert/data/test_data/prewhitening'
n_sectors = len(i_sectors)
# (use extraction recipy to get the parameters)
harmonics, harmonic_n = tsf.find_harmonics_from_pattern(f_n_0, p_orb)
model_h = tsf.linear_curve(times, const_0, slope_0, i_sectors)
model_h += tsf.sum_sines(times, f_n_0[harmonics], a_n_0[harmonics], ph_n_0[harmonics])
model_nh = tsf.linear_curve(times, const_0, slope_0, i_sectors)
model_nh += tsf.sum_sines(times, np.delete(f_n_0, harmonics), np.delete(a_n_0, harmonics), np.delete(ph_n_0, harmonics))
freqs, ampls = tsf.scargle(times, signal - np.mean(signal))
freqs_0, ampls_0 = tsf.scargle(times, signal - model_0)
fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(freqs, ampls, label='signal')
ax.plot(freqs_0, ampls_0, label='extraction residual')
for i in range(len(f_n_0)):
    ax.errorbar([f_n_0[i], f_n_0[i]], [0, a_n_0[i]], xerr=[0, 0], yerr=[0, 0],
                linestyle=':', capsize=2, c='tab:orange')
    ax.annotate(f'{i+1}', (f_n_0[i], a_n_0[i]))
plt.xlabel('frequency (1/d)', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.tight_layout()
plt.legend()
plt.show()
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))
ax[0].scatter(times, signal)
ax[0].scatter(times, model_0)
ax[0].scatter(times, model_h, marker='.')
ax[0].scatter(times, model_nh, marker='.')
ax[1].scatter(times, signal - model_0)
plt.tight_layout()
plt.show()
# get uncertainties in i and sb_ratio from delta-BIC
args = (p_orb, t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2, depths[0], depths[1], t_1_err, t_2_err,
        tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err, d_1_err, d_2_err)
minimum = af.objective_inclination(i, *args)
delta_bic = 10
delta_func = (lambda x: abs(af.objective_inclination(x, *args) - minimum - delta_bic))
res_1 = sp.optimize.minimize_scalar(delta_func, method='bounded', bounds=(np.pi/4, i))
res_2 = sp.optimize.minimize_scalar(delta_func, method='bounded', bounds=(i, np.pi/2))
print(res_1.fun, res_2.fun)
print(i/np.pi*180 - res_1.x/np.pi*180, i/np.pi*180, res_2.x/np.pi*180 - i/np.pi*180)
# get uncertainties by putting in gaussians
rng = np.random.default_rng()
n_gen = 10**4
normal_t_1 = rng.normal(t_1, t_1_err, n_gen)
normal_t_2 = rng.normal(t_2, t_2_err, n_gen)
normal_tau_1_1 = rng.normal(tau_1_1, tau_1_1_err, n_gen)
normal_tau_1_2 = rng.normal(tau_1_2, tau_1_2_err, n_gen)
normal_tau_2_1 = rng.normal(tau_2_1, tau_2_1_err, n_gen)
normal_tau_2_2 = rng.normal(tau_2_2, tau_2_2_err, n_gen)
normal_d_1 = rng.normal(depths[0], d_1_err, n_gen)
normal_d_2 = rng.normal(depths[1], d_2_err, n_gen)
i_vals = np.zeros(n_gen)
for k in range(n_gen):
    bounds = (np.pi/4, np.pi/2)
    args = (p_orb_7, normal_t_1[k], normal_t_2[k], normal_tau_1_1[k], normal_tau_1_2[k], normal_tau_2_1[k], normal_tau_2_2[k],
            normal_d_1[k], normal_d_2[k], t_1_err, t_2_err,
            tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err, d_1_err, d_2_err)
    res = sp.optimize.minimize_scalar(af.objective_inclination, args=args, method='bounded', bounds=bounds)
    i_vals[k] = res.x
phi0_vals = np.pi * (normal_tau_1_1 + normal_tau_1_2 + normal_tau_2_1 + normal_tau_2_2) / (2 * p_orb)
e_vals, w_vals = af.ecc_omega_approx(p_orb, normal_t_1, normal_t_2, normal_tau_1_1, normal_tau_1_2, normal_tau_2_1, normal_tau_2_2,
                                               i_vals, phi0_vals)
rsumsma_vals = af.radius_sum_from_phi0(e_vals, i_vals, phi0_vals)
sbratio_vals = np.zeros(n_gen)
for k in range(n_gen):
    sbratio_vals[k] = af.sb_ratio_from_d_ratio(normal_d_2[k] / normal_d_1[k], e_vals[k], w_vals[k], i_vals[k], rsumsma_vals[k], r_ratio=1)
# plots
# Highest density interval (HDI) for given probability.
# https://arviz-devs.github.io/arviz/api/generated/arviz.hdi.html




# looking at third light
long_spoc = tic_numbers_spoc[spoc_in_cat][n_sectors_spoc[spoc_in_cat] > 10]
long_qlp = tic_numbers_qlp[qlp_in_cat][n_sectors_qlp[qlp_in_cat] > 10]
def th_l_test(tic):
    p_orb = float(EB_catalogue['eclipse_period'][EB_catalogue['TIC_ID'] == tic])
    t_0 = float(EB_catalogue['t_supcon'][EB_catalogue['TIC_ID'] == tic])
    # get the SPOC/QLP data
    # times, sap_signal, signal, signal_err, sectors, t_sectors, crowd = ut.load_tess_lc(tic, spoc_files, apply_flags=True)
    times, sap_signal, signal, signal_err, sectors, t_sectors, crowd = ut.load_tess_lc(tic, qlp_files, apply_flags=True)
    i_sectors = ut.convert_tess_t_sectors(times, t_sectors)
    times, signal, sector_medians, t_start, t_combined, i_half_s = ut.stitch_tess_sectors(times, signal, i_sectors)
    
    wavg_p_orb, const_3, slope_3, f_n_3, a_n_3, ph_n_3 = frequency_analysis(times, signal, p_orb, i_sectors, i_half_s, verbose=False)
    corr, check = ut.check_crowdsap_correlation(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, crowdsap, verbose=False)
    
    # prewhitening
    model = tsf.linear_curve(times, const_3, slope_3, i_sectors)
    model += tsf.sum_sines(times, f_n_3, a_n_3, ph_n_3)
    resid = signal - model
    noise_level = np.std(resid)  # also: const_err * np.sqrt(len(tess_times))  # the standard deviation of the residuals
    const_err, f_n_err, a_n_err, ph_n_err = tsf.sines_uncertainties(times, resid, a_n)
    # harmonics within 3 sigma
    harmonics = tsf.find_harmonics_from_pattern(f_n_3, wavg_p_orb)
    # save individual analysis file
    data = pd.DataFrame()
    data['f_n'] = np.append([''], [f'f_{i + 1}' for i in range(len(f_n))])
    data['frequency'] = np.append([0], f_n)
    data['f_error'] = np.append([0], f_n_err)
    data['f_unit'] = np.append([''], ['1/d' for i in range(len(f_n))])
    data['a_n'] = np.append(['const'], [f'a_{i + 1}' for i in range(len(f_n))])
    data['amplitude'] = np.append([const], a_n)
    data['a_error'] = np.append([const_err], a_n_err)
    data['a_unit'] = np.append(['light curve ordinate'], ['light curve ordinate' for i in range(len(f_n))])
    data['S/N'] = np.append([const / noise_level], a_n / noise_level)
    data['ph_n'] = np.append([''], [f'ph_{i + 1}' for i in range(len(f_n))])
    data['phase'] = np.append([0], ph_n)
    data['ph_error'] = np.append([0], ph_n_err)
    data['ph_unit'] = np.append([''], ['radians' for i in range(len(f_n))])
    data['harmonic'] = np.append([False], harmonics)
    data.to_csv(os.path.join(test_dir, f'TIC_{tic}_third_light_{corr}_{check}.csv'), index=False)
    return check
pool = mp.Pool(8)
check_tl1 = pool.map(th_l_test, long_spoc)
check_tl2 = pool.map(th_l_test, long_qlp)  # change over data load statement

# fake lc
# resulting model from the short test, times from the long lc
model_f = tsf.sum_sines(times, f_n_7[harmonics], a_n_7[harmonics], ph_n_7[harmonics])
const_f, slope_f = np.ones(len(i_sectors)) - np.median(model_f), np.zeros(len(i_sectors))
model_f += tsf.linear_curve(times, const_f, slope_f, i_sectors)
# make a model in counts and add 'third light'
model_f_tl = model_f * 3000
light_3 = [600, 1100, 400, 2200, 1300, 1400, 1700, 1600, 300, 600, 400, 1200, 1100]
for i, s in enumerate(i_sectors):
    model_f_tl[s[0]:s[1]] += light_3[i]
fig, ax = plt.subplots()
ax.scatter(times, model_f_tl)
plt.show()
# normalise the counts - it is now uneven depth
model_f_norm, medians = ut.normalise_counts(model_f_tl, i_sectors)
fig, ax = plt.subplots()
ax.scatter(times, model_f_norm)
plt.show()
# correct for third light
model_f_cor = ut.correct_for_crowdsap(model_f_norm, 1 - light_3/medians, i_sectors)
fig, ax = plt.subplots()
ax.scatter(times, model_f_cor)
plt.show()
# model for the fake lc
model_f_mod = tsf.linear_curve(times, const_f, slope_f, i_sectors)
model_f_mod += tsf.sum_sines(times, f_n_7[harmonics], a_n_7[harmonics] * 0.7, ph_n_7[harmonics])
# model_f_s = tsf.sum_sines(times, f_n_7[harmonics], a_n_7[harmonics], ph_n_7[harmonics]) * 0.7  # equivalent
# model_f_s += tsf.linear_curve(times, const_f, slope_f, i_sectors)
fig, ax = plt.subplots()
ax.scatter(times, model_f_norm)
ax.scatter(times, model_f_mod)
plt.show()
# fit for third light
min_light_3, stretch, const, slope = tsfit.fit_minimum_third_light(times, model_f_norm, p_orb_7, const_f, slope_f,
                                                                   f_n_7[harmonics], a_n_7[harmonics] * 0.7,
                                                                   ph_n_7[harmonics], i_sectors, verbose=True)
corr, check = ut.check_crowdsap_correlation(times, model_f_norm, p_orb_7, const_f, slope_f, f_n_7[harmonics],
                                            a_n_7[harmonics] * 0.7, ph_n_7[harmonics], i_sectors, 1 - light_3/medians, verbose=True)
model_f_mod_cor = tsf.sum_sines(times, f_n_7[harmonics], a_n_7[harmonics] * 0.7 * stretch, ph_n_7[harmonics])
model_f_mod_cor += tsf.linear_curve(times, const, slope, i_sectors)
model_f_mod_cor = ut.model_crowdsap(model_f_mod_cor, 1 - min_light_3, i_sectors)
fig, ax = plt.subplots()
ax.scatter(times, model_f_norm)
ax.scatter(times, model_f_mod)
ax.scatter(times, model_f_mod_cor)
plt.show()
fig, ax = plt.subplots()
ax.scatter(times, model_f_norm - model_f_mod)
ax.scatter(times, model_f_norm - model_f_mod_cor)
plt.show()

# blind synthetic lc (Cole)
times, signal = np.loadtxt(os.path.join(test_dir, 'synthetic_test_eb/sim_noisy_pulse_eb.lc'), unpack=True)
p_orb = 10.78  # roughly eyeballed
times = times - times[0]
i_sectors = np.array([[0, len(times)]])
signal, medians = ut.normalise_counts(signal, i_sectors=i_sectors)
# frequency_analysis(0, times, signal, p_orb, i_sectors, i_sectors, data_id='synthetic',
#                    save_dir=os.path.join(test_dir, 'synthetic_test_eb'), verbose=True)
results, errors, stats = ut.read_results(os.path.join(test_dir, 'synthetic_test_eb',
                                                      'tic_0_analysis', 'tic_0_analysis_8.hdf5'))
p_orb_8, const_8, slope_8, f_n_8, a_n_8, ph_n_8 = results
p_orb_8 = p_orb_8[0]
p_err_8, c_err_8, sl_err_8, f_n_err_8, a_n_err_8, ph_n_err_8 = errors
n_param_8, bic_8, noise_level = stats
table_1, table_2 = tsf.orbital_elements(0, times, signal, i_sectors, i_sectors, data_id='synthetic_test_eb',
                                        save_dir=os.path.join(test_dir, 'synthetic_test_eb'), verbose=True)
# normalise signal before plotting periodograms
harmonics, harmonic_n = tsf.find_harmonics_from_pattern(f_n_8, p_orb_8[0])
n_param_calc = 2 * len(const_8) + 1 + 2 * len(harmonics) + 3 * (len(f_n_8) - len(harmonics))
model = tsf.linear_curve(times, const_8, slope_8, i_sectors)
model += tsf.sum_sines(times, f_n_8, a_n_8, ph_n_8)
bic = tsf.calc_bic(signal - model, n_param_calc)
print(np.var(signal - model), bic)
vis.plot_pd_single_output(times, signal, const_8, slope_8, f_n_8, a_n_8, ph_n_8, n_param_calc, bic, i_sectors,
                      'Synthetic test EB output 7', zoom=None, annotate=False, save_file=None, show=True)
sn_cut = (a_n_8/noise_level > 4)
harmonics, harmonic_n = tsf.find_harmonics_from_pattern(f_n_8[sn_cut], p_orb_8[0])
param_cut = 2 * len(const_8) + 1 + 2 * len(harmonics) + 3 * (len(f_n_8[sn_cut]) - len(harmonics))
model = tsf.linear_curve(times, const_8, slope_8, i_sectors)
model += tsf.sum_sines(times, f_n_8[sn_cut], a_n_8[sn_cut], ph_n_8[sn_cut])
bic = tsf.calc_bic(signal - model, n_param_calc)
print(np.var(signal - model), bic)
vis.plot_pd_single_output(times, signal, const_8, slope_8, f_n_8[sn_cut], a_n_8[sn_cut], ph_n_8[sn_cut], param_cut, bic, i_sectors,
                          'Synthetic test EB output 7', zoom=None, annotate=False, save_file=None, show=True)
# exclude harmonics from sn_cut
harmonics, harmonic_n = tsf.find_harmonics_from_pattern(f_n_8, p_orb_8[0])
h_mask = np.zeros(len(f_n_8), dtype=bool)
h_mask[harmonics] = True
sn_cut_h = (a_n_8/noise_level > 4) | h_mask
param_cut_h = 2 * len(const_8) + 1 + 2 * len(harmonics) + 3 * (len(f_n_8[sn_cut_h]) - len(harmonics))
model = tsf.linear_curve(times, const_8, slope_8, i_sectors)
model += tsf.sum_sines(times, f_n_8[sn_cut_h], a_n_8[sn_cut_h], ph_n_8[sn_cut_h])
bic = tsf.calc_bic(signal - model, n_param_calc)
print(np.var(signal - model), bic)
vis.plot_pd_single_output(times, signal, const_8, slope_8, f_n_8[sn_cut_h], a_n_8[sn_cut_h], ph_n_8[sn_cut_h], param_cut_h, bic, i_sectors,
                          'Synthetic test EB output 7', zoom=None, annotate=False, save_file=None, show=True)
# emcee for ellc model?
# not beneficial it seems


# the synthetic tests by Cole and Andrej
import multiprocessing as mp
syn_dir = '/lhome/lijspeert/data/test_data/eccentricities'
times, signal, signal_err = np.loadtxt(os.path.join(syn_dir, 'simulations_cole/sim_000_sc.dat'), unpack=True)
times = times - times[0]
i_sectors = np.array([[0, len(times)]])
plt.scatter(times, signal)

all_files = []
for root, dirs, files in os.walk(os.path.join(syn_dir, 'simulations_cole')):
    for file in files:
        if file.startswith('sim_'):
            all_files.append(os.path.join(root, file))
files_lc = []
for root, dirs, files in os.walk(os.path.join(syn_dir, 'simulations_cole')):
    for file in files:
        if file.endswith('_lc.dat'):
            files_lc.append(os.path.join(root, file))
files_sc = []
for root, dirs, files in os.walk(os.path.join(syn_dir, 'simulations_cole')):
    for file in files:
        if file.endswith('_sc.dat'):
            files_sc.append(os.path.join(root, file))

def analyse_parallel(file):
    target_id = file[68:74]  # extract the number (Cole)
    # target_id = file[65:67]  # extract the number (Andrej)
    times, signal, signal_err = np.loadtxt(file, usecols=(0, 1, 2), unpack=True)
    times = times - times[0]
    i_half_s = np.array([[0, len(times)]])
    out_a = tsf.frequency_analysis(target_id, times, signal, i_half_s, p_orb=0, save_dir=file[:64],
                                   data_id='blind_ecc_Cole', overwrite=False, verbose=False)
    # if not full output, stop
    if (len(out_a[0]) < 8):
        out_b = tsf.eclipse_analysis(target_id, times, signal, signal_err, i_half_s, save_dir=file[:64],
                                     data_id='blind_ecc_Cole', verbose=False, overwrite=False)
        out_c = tsf.pulsation_analysis(target_id, times, signal, i_half_s, save_dir=file[:64], data_id='blind_ecc_Cole',
                                       verbose=False, overwrite=False)
    return

# plotting in series
for file in files_lc:
    target_id = file[68:74]  # extract the number (Cole)
    times, signal, signal_err = np.loadtxt(file, usecols=(0, 1, 2), unpack=True)
    times = times - times[0]
    i_half_s = np.array([[0, len(times)]])
    ut.sequential_plotting(target_id, times, signal, i_half_s, file[:64])

pool = mp.Pool(12)
pool.map(analyse_parallel, np.array(files_lc)[::-1])
# find out where they got
n_all = []
for file in all_files:
    tic = file[68:74]  # extract the number (Cole)
    data_dir = os.path.join(file[:64], f'tic_{tic}_analysis')
    if os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_13.csv')):
        n = '13'
    elif os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_12.csv')):
        n = '12'
    elif os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_11.csv')):
        n = '11'
    elif os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_10.csv')):
        n = '10'
    elif os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_9.csv')):
        n = '9'
    elif os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_8.hdf5')):
        n = '8'
    elif os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_7.hdf5')):
        n = '7'
    elif os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_6.hdf5')):
        n = '6'
    elif os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_5.hdf5')):
        n = '5'
    elif os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_4.hdf5')):
        n = '4'
    elif os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_3.hdf5')):
        n = '3'
    elif os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_2.hdf5')):
        n = '2'
    elif os.path.isfile(os.path.join(file[:64], f'tic_{tic}_analysis', f'tic_{tic}_analysis_1.hdf5')):
        n = '1'
    else:
        n = '0'
    n_all.append(n)
np.savetxt(os.path.join(file[:64], 'stage_finished'), np.column_stack(([file[64:] for file in all_files], n_all)), fmt='%s %s')

"""Not enough eclipses (not even full period): 04, 05, 06, 08, 09, 10, 13
slightly more than a period but no 2 eclipses of each (p and s): 07, 11, 12
"""
all_files = []
for root, dirs, files in os.walk(os.path.join(syn_dir, 'blind_ecc_Andrej')):
    for file in files:
        if file.endswith('.lc'):
            all_files.append(os.path.join(root, file))

times, signal = np.loadtxt(os.path.join(syn_dir, 'blind_ecc_Andrej/s01.lc'), usecols=(0, 1), unpack=True)
times = times - times[0]
signal = signal / np.median(signal)
i_half_s = np.array([[0, len(times)]])
plt.scatter(times, signal)






# FEROS data overview
import os
import numpy as np
import matplotlib.pyplot as plt
import fitshandler as fh
import timeseries_functions as tsf

dir_feros = '/lhome/lijspeert/data/FEROS_wk2_data/extracted'
all_files = []
for root, dirs, files in os.walk(dir_feros):
    for file in files:
        all_files.append(os.path.join(root, file))

obj_name = ['' for _ in range(len(all_files))]
exp_date = np.zeros(len(all_files))
for i, file in enumerate(all_files):
    obj_name[i] = fh.get_card_value(os.path.join(dir_feros, file), 'OBJECT')
    exp_date[i] = fh.get_card_value(os.path.join(dir_feros, file), 'MJD-OBS')
obj_name = np.array(obj_name)
un_obj_name = np.unique(obj_name)
print(f'Number of stars observed: {len(un_obj_name)}, out of 30 proposed.')
"""
all_targets = ['HD 46792', 'HD 79365', 'HD 52349', 'TYC 8155-1212-1', 'HD 300344', 'HD 92741', 'HD 82110', 'HD 91141',
'HD 309317', 'HD 100737', 'HD 84493', 'HD 51981', 'HD 97966', 'HD 68340', 'HD 104233', 'HD 91154', 'HD 121776',
'HD 80627', 'HD 304241', 'TYC 8149-3211-1', 'HD 62738', 'TYC 8151-937-1', 'HD 75872', 'TYC 8972-249-1', 'HD 67025',
'HD 66235', 'HD 28913', 'HD 66673', 'HD 297793', 'TYC 8514-106-1']

not observed: (TYC8149-3211-1 was observed but only twice)
HD 297793, TYC 8972-249-1, TYC 8149-3211-1, HD 80627, HD 66673
"""
all_targets = ['HD46792', 'HD79365', 'HD52349', 'TYC8155-1212-1', 'HD300344', 'HD92741', 'HD82110', 'HD91141',
               'HD309317', 'HD100737', 'HD84493', 'HD51981', 'HD97966', 'HD68340', 'HD104233', 'HD91154', 'HD121776',
               'HD80627', 'HD304241', 'TYC8149-3211-1', 'HD62738', 'TYC8151-937-1', 'HD75872', 'TYC8972-249-1',
               'HD67025', 'HD66235', 'HD28913', 'HD66673', 'HD297793', 'TYC8514-106-1']
periods = [2.97, 0.907, 2.78, 3.6, 2.79, 5.37, 1.88, 2.38, 2.26, 2.55, 6.51, 0.926, 1.27, 1.31, 1.82, 3.66,
           1.74, 2.77, 2.73, 1.81, 1.65, 2.05, 0.945, 1.23, 1.28, 1.59, 1.49, 1.66, 0.815, 0.790]
t_0 = [1326.98, 1518.05, 1470.44, 1519.20, 1545.19, 1574.55, 1544.82, 1545.03, 1570.18, 1596.63, 1550.5,
       1468.44, 1569.52, 1492.33, 1570.23, 1543.79, 1597.38, 1545.68, 1575.1, 1519.14, 1492.34, 1518.30,
       1518.16, 1570.50, 1491.68, 1492.61, 1414.6, 1492.21, 1544.04, 1325.84]
t_0 = np.array(t_0) + 2457000.0 - 2400000.5
n_epochs = np.array([np.sum(obj_name == name) for name in all_targets])
target_date = [exp_date[obj_name == name] for name in all_targets]
indices = np.arange(len(all_targets))
fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 8))
ax[0].barh(all_targets, n_epochs)
ax[0].xaxis.set_tick_params(labelsize='12')
ax[0].yaxis.set_tick_params(labelsize='12')
ax[0].grid()
ax[0].set_xlabel('epochs', fontsize='14')
for i in range(len(all_targets)):
    ax[1].plot([-0.5, 0.5], [all_targets[i]] * 2, c='grey', alpha=0.5)
    ax[1].scatter(tsf.fold_time_series(target_date[i], periods[i], zero=t_0[i]), [all_targets[i]] * n_epochs[i],
                  marker='.', c='tab:blue')
ax[1].set_xlabel('orbital phase', fontsize='14')
plt.tight_layout()
plt.show()











