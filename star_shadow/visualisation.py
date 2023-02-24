"""STAR SHADOW
Satellite Time series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This Python module contains functions for visualisation;
specifically for visualising the analysis of stellar variability and eclipses.

Code written by: Luc IJspeert
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import corner

from . import timeseries_functions as tsf
from . import timeseries_fitting as tsfit
from . import analysis_functions as af
from . import utility as ut

# mpl style sheet
script_dir = os.path.dirname(os.path.abspath(__file__))  # absolute dir the script is in
plt.style.use(script_dir.replace('star_shadow/star_shadow', 'star_shadow/data/mpl_stylesheet.dat'))


def plot_pd_single_output(times, signal, p_orb, p_err, const, slope, f_n, a_n, ph_n, i_sectors, annotate=True,
                          save_file=None, show=True):
    """Plot the periodogram with one output of the analysis recipe."""
    # separate harmonics
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    # make model
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    model = model_linear + model_sinusoid
    # make periodograms
    freqs, ampls = tsf.astropy_scargle(times, signal)
    freq_range = np.ptp(freqs)
    freqs_r, ampls_r = tsf.astropy_scargle(times, signal - model)
    # get error values
    errors = tsf.formal_uncertainties(times, signal - model, a_n, i_sectors)
    # max plot value
    y_max = max(np.max(ampls), np.max(a_n))
    # plot
    fig, ax = plt.subplots()
    if (len(harmonics) > 0):
        ax.errorbar([1 / p_orb, 1 / p_orb], [0, y_max], xerr=[0, p_err / p_orb**2],
                    linestyle='-', capsize=2, c='tab:grey', label=f'orbital frequency (p={p_orb:1.4f}d +-{p_err:1.4f})')
        for i in range(2, np.max(harmonic_n) + 1):
            ax.plot([i / p_orb, i / p_orb], [0, y_max], linestyle='-', c='tab:grey', alpha=0.3)
        ax.errorbar([], [], xerr=[], yerr=[], linestyle='-', capsize=2, c='tab:red', label='extracted harmonics')
    ax.plot(freqs, ampls, c='tab:blue', label='signal')
    ax.plot(freqs_r, ampls_r, c='tab:orange', label='residual')
    for i in range(len(f_n)):
        if i in harmonics:
            ax.errorbar([f_n[i], f_n[i]], [0, a_n[i]], xerr=[0, errors[2][i]], yerr=[0, errors[3][i]],
                        linestyle='-', capsize=2, c='tab:red')
        else:
            ax.errorbar([f_n[i], f_n[i]], [0, a_n[i]], xerr=[0, errors[2][i]], yerr=[0, errors[3][i]],
                        linestyle='-', capsize=2, c='tab:pink')
        if annotate:
            ax.annotate(f'{i + 1}', (f_n[i], a_n[i]))
    ax.errorbar([], [], xerr=[], yerr=[], linestyle='-', capsize=2, c='tab:pink', label='extracted frequencies')
    ax.set_xlim(freqs[0] - freq_range * 0.05, freqs[-1] + freq_range * 0.05)
    plt.xlabel('frequency (1/d)')
    plt.ylabel('amplitude')
    plt.legend()
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_pd_full_output(times, signal, models, p_orb_i, p_err_i, f_n_i, a_n_i, i_sectors, save_file=None, show=True):
    """Plot the periodogram with the full output of the analysis recipe."""
    # make periodograms
    freqs, ampls = tsf.astropy_scargle(times, signal - np.mean(signal))
    freq_range = np.ptp(freqs)
    freqs_1, ampls_1 = tsf.astropy_scargle(times, signal - models[0] - np.all(models[0] == 0) * np.mean(signal))
    freqs_2, ampls_2 = tsf.astropy_scargle(times, signal - models[1] - np.all(models[1] == 0) * np.mean(signal))
    freqs_3, ampls_3 = tsf.astropy_scargle(times, signal - models[2] - np.all(models[2] == 0) * np.mean(signal))
    freqs_4, ampls_4 = tsf.astropy_scargle(times, signal - models[3] - np.all(models[3] == 0) * np.mean(signal))
    freqs_5, ampls_5 = tsf.astropy_scargle(times, signal - models[4] - np.all(models[4] == 0) * np.mean(signal))
    # get error values
    err_1 = tsf.formal_uncertainties(times, signal - models[0], a_n_i[0], i_sectors)
    err_2 = tsf.formal_uncertainties(times, signal - models[1], a_n_i[1], i_sectors)
    err_3 = tsf.formal_uncertainties(times, signal - models[2], a_n_i[2], i_sectors)
    err_4 = tsf.formal_uncertainties(times, signal - models[3], a_n_i[3], i_sectors)
    err_5 = tsf.formal_uncertainties(times, signal - models[4], a_n_i[4], i_sectors)
    # max plot value
    if (len(f_n_i[4]) > 0):
        y_max = max(np.max(ampls), np.max(a_n_i[4]))
    else:
        y_max = np.max(ampls)
    # plot
    fig, ax = plt.subplots()
    ax.plot(freqs, ampls, label='signal')
    if (len(f_n_i[0]) > 0):
        ax.plot(freqs_1, ampls_1, label='extraction residual')
    if (len(f_n_i[1]) > 0):
        ax.plot(freqs_2, ampls_2, label='NL-LS optimisation residual')
    if (len(f_n_i[2]) > 0):
        ax.plot(freqs_3, ampls_3, label='coupled harmonics residual')
    if (len(f_n_i[3]) > 0):
        ax.plot(freqs_4, ampls_4, label='additional frequencies residual')
    if (len(f_n_i[4]) > 0):
        ax.plot(freqs_5, ampls_5, label='NL-LS fit residual with harmonics residual')
    # period
    if (p_orb_i[4] > 0):
        ax.errorbar([1 / p_orb_i[4], 1 / p_orb_i[4]], [0, y_max], xerr=[0, p_err_i[4] / p_orb_i[4]**2],
                    linestyle='--', capsize=2, c='k', label=f'orbital frequency (p={p_orb_i[4]:1.4f}d)')
    elif (p_orb_i[2] > 0):
        ax.errorbar([1 / p_orb_i[2], 1 / p_orb_i[2]], [0, y_max], xerr=[0, p_err_i[2] / p_orb_i[2]**2],
                    linestyle='--', capsize=2, c='k', label=f'orbital frequency (p={p_orb_i[2]:1.4f}d)')
    # frequencies
    for i in range(len(f_n_i[0])):
        ax.errorbar([f_n_i[0][i], f_n_i[0][i]], [0, a_n_i[0][i]], xerr=[0, err_1[2][i]], yerr=[0, err_1[3][i]],
                    linestyle=':', capsize=2, c='tab:orange')
    for i in range(len(f_n_i[1])):
        ax.errorbar([f_n_i[1][i], f_n_i[1][i]], [0, a_n_i[1][i]], xerr=[0, err_2[2][i]], yerr=[0, err_2[3][i]],
                    linestyle=':', capsize=2, c='tab:green')
    for i in range(len(f_n_i[2])):
        ax.errorbar([f_n_i[2][i], f_n_i[2][i]], [0, a_n_i[2][i]], xerr=[0, err_3[2][i]], yerr=[0, err_3[3][i]],
                    linestyle=':', capsize=2, c='tab:red')
    for i in range(len(f_n_i[3])):
        ax.errorbar([f_n_i[3][i], f_n_i[3][i]], [0, a_n_i[3][i]], xerr=[0, err_4[2][i]], yerr=[0, err_4[3][i]],
                    linestyle=':', capsize=2, c='tab:purple')
    for i in range(len(f_n_i[4])):
        ax.errorbar([f_n_i[4][i], f_n_i[4][i]], [0, a_n_i[4][i]], xerr=[0, err_5[2][i]], yerr=[0, err_5[3][i]],
                    linestyle=':', capsize=2, c='tab:brown')
        ax.annotate(f'{i + 1}', (f_n_i[4][i], a_n_i[4][i]))
    ax.set_xlim(freqs[0] - freq_range * 0.05, freqs[-1] + freq_range * 0.05)
    plt.xlabel('frequency (1/d)')
    plt.ylabel('amplitude')
    plt.legend()
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_sinusoids(times, signal, const, slope, f_n, a_n, ph_n, i_sectors, save_file=None, show=True):
    """Shows the separated harmonics in several ways"""
    t_mean = np.mean(times)
    # make models
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sines = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - (model_linear + model_sines)
    # plot the full model light curve
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot([t_mean, t_mean], [np.min(signal), np.max(signal)], c='grey', alpha=0.3)
    ax[0].scatter(times, signal, marker='.', label='signal')
    ax[0].plot(times, model_linear + model_sines, c='tab:orange', label='full model (linear + sinusoidal)')
    ax[1].plot([t_mean, t_mean], [np.min(resid), np.max(resid)], c='grey', alpha=0.3)
    ax[1].scatter(times, resid, marker='.')
    ax[0].set_ylabel('signal/model')
    ax[0].legend()
    ax[1].set_ylabel('residual')
    ax[1].set_xlabel('time (d)')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_harmonics(times, signal, p_orb, p_err, const, slope, f_n, a_n, ph_n, i_sectors, save_file=None,
                      show=True):
    """Shows the separated harmonics in several ways"""
    t_mean = np.mean(times)
    # make models
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    model_h = tsf.sum_sines(times, f_n[harmonics], a_n[harmonics], ph_n[harmonics])
    model_nh = tsf.sum_sines(times, np.delete(f_n, harmonics), np.delete(a_n, harmonics),
                             np.delete(ph_n, harmonics))
    resid_nh = signal - model_nh
    resid_h = signal - model_h
    # plot the harmonic model and non-harmonic model
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot([t_mean, t_mean], [np.min(resid_nh), np.max(resid_nh)], c='grey', alpha=0.3)
    ax[0].scatter(times, resid_nh, marker='.', c='tab:blue', label='signal - non-harmonics')
    ax[0].plot(times, model_line + model_h, c='tab:orange', label='linear + harmonic model, '
                                                                  f'p={p_orb:1.4f}d (+-{p_err:1.4f})')
    ax[1].plot([t_mean, t_mean], [np.min(resid_h), np.max(resid_h)], c='grey', alpha=0.3)
    ax[1].scatter(times, resid_h, marker='.', c='tab:blue', label='signal - harmonics')
    ax[1].plot(times, model_line + model_nh, c='tab:orange', label='linear + non-harmonic model')
    ax[0].set_ylabel('residual/model')
    ax[0].legend()
    ax[1].set_ylabel('residual/model')
    ax[1].set_xlabel('time (d)')
    ax[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_timings_harmonics(times, signal, p_orb, timings, depths, timings_err, depths_err, const, slope,
                              f_n, a_n, ph_n, f_h, a_h, ph_h, i_sectors, save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the first and
    last contact points as well as minima indicated.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err = timings_err
    dur_b_1_err = np.sqrt(t_1_1_err**2 + t_1_2_err**2)
    dur_b_2_err = np.sqrt(t_2_1_err**2 + t_2_2_err**2)
    t_mean = np.mean(times)
    # plotting bounds
    t_ext_1 = t_1_1 - 6 * t_1_1_err - t_1
    t_ext_2 = t_1_2 + 6 * t_1_2_err - t_1
    # make the model times array, one full period plus the primary eclipse halves
    t_extended, ext_left, ext_right = tsf.fold_time_series(times, p_orb, t_1, t_ext_1=t_ext_1, t_ext_2=t_ext_2)
    s_extended = np.concatenate((signal[ext_left], signal, signal[ext_right]))
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    t_model = np.arange(t_ext_1, p_orb + t_ext_2, 0.001)
    model_h = 1 + tsf.sum_sines(t_model + t_1, f_h, a_h, ph_h, t_shift=False)
    model_nh = tsf.sum_sines(times, f_n, a_n, ph_n) - tsf.sum_sines(times, f_h, a_h, ph_h)
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line + 1
    ecl_signal = np.concatenate((ecl_signal[ext_left], ecl_signal, ecl_signal[ext_right]))
    # determine a lc offset to match the model at the edges
    h_1, h_2 = af.height_at_contact(f_h, a_h, ph_h, t_1_1, t_1_2, t_2_1, t_2_2)
    offset = 1 - (h_1 + h_2) / 2
    # some plotting parameters
    h_top_1 = h_1 + offset
    h_top_2 = h_2 + offset
    h_bot_1 = h_1 - depths[0] + offset
    h_bot_2 = h_2 - depths[1] + offset
    s_minmax = np.array([np.min(signal), np.max(signal)])
    # plot (shift al timings of primary eclipse by t_1)
    fig, ax = plt.subplots()
    ax.scatter(t_extended, s_extended, marker='.', label='original folded signal')
    ax.scatter(t_extended, ecl_signal + offset, marker='.', c='tab:orange',
               label='(non-harmonics + linear) model residual')
    ax.plot(t_model, model_h + offset, c='tab:red', label='harmonics')
    ax.plot([0, 0], s_minmax, '--', c='tab:pink')
    ax.plot([t_2 - t_1, t_2 - t_1], s_minmax, '--', c='tab:pink')
    ax.plot([t_1_1 - t_1, t_1_1 - t_1], s_minmax, '--', c='tab:purple', label=r'eclipse edges/minima/depths')
    ax.plot([t_1_2 - t_1, t_1_2 - t_1], s_minmax, '--', c='tab:purple')
    ax.plot([t_2_1 - t_1, t_2_1 - t_1], s_minmax, '--', c='tab:purple')
    ax.plot([t_2_2 - t_1, t_2_2 - t_1], s_minmax, '--', c='tab:purple')
    ax.plot([t_1_1 - t_1, t_1_2 - t_1], [h_bot_1, h_bot_1], '--', c='tab:purple')
    ax.plot([t_2_1 - t_1, t_2_2 - t_1], [h_bot_2, h_bot_2], '--', c='tab:purple')
    ax.plot([t_1_1 - t_1, t_1_2 - t_1], [h_top_1, h_top_1], '--', c='tab:purple')
    ax.plot([t_2_1 - t_1, t_2_2 - t_1], [h_top_2, h_top_2], '--', c='tab:purple')
    # 1 sigma errors
    ax.fill_between([-t_1_err, t_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:red', alpha=0.3)
    ax.fill_between([t_2 - t_1 - t_2_err, t_2 - t_1 + t_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:red', alpha=0.3)
    ax.fill_between([t_1_1 - t_1 - t_1_1_err, t_1_1 - t_1 + t_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.3, label=r'1 and 3 $\sigma$ error')
    ax.fill_between([t_1_2 - t_1 - t_1_2_err, t_1_2 - t_1 + t_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.3)
    ax.fill_between([t_2_1 - t_1 - t_2_1_err, t_2_1 - t_1 + t_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.3)
    ax.fill_between([t_2_2 - t_1 - t_2_2_err, t_2_2 - t_1 + t_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.3)
    ax.fill_between([t_1_1 - t_1, t_1_2 - t_1], y1=[h_bot_1 + depths_err[0], h_bot_1 + depths_err[0]],
                    y2=[h_bot_1 - depths_err[0], h_bot_1 - depths_err[0]], color='tab:purple', alpha=0.3)
    ax.fill_between([t_2_1 - t_1, t_2_2 - t_1], y1=[h_bot_2 + depths_err[1], h_bot_2 + depths_err[1]],
                    y2=[h_bot_2 - depths_err[1], h_bot_2 - depths_err[1]], color='tab:purple', alpha=0.3)
    # 3 sigma errors
    ax.fill_between([-3 * t_1_err, 3 * t_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:red', alpha=0.2)
    ax.fill_between([t_2 - t_1 - 3 * t_2_err, t_2 - t_1 + 3 * t_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:red', alpha=0.2)
    ax.fill_between([t_1_1 - t_1 - 3 * t_1_1_err, t_1_1 - t_1 + 3 * t_1_1_err],
                    y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:purple', alpha=0.2)
    ax.fill_between([t_1_2 - t_1 - 3 * t_1_2_err, t_1_2 - t_1 + 3 * t_1_2_err],
                    y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:purple', alpha=0.2)
    ax.fill_between([t_2_1 - t_1 - 3 * t_2_1_err, t_2_1 - t_1 + 3 * t_2_1_err],
                    y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:purple', alpha=0.2)
    ax.fill_between([t_2_2 - t_1 - 3 * t_2_2_err, t_2_2 - t_1 + 3 * t_2_2_err],
                    y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:purple', alpha=0.2)
    ax.fill_between([t_1_1 - t_1, t_1_2 - t_1], y1=[h_bot_1 + 3 * depths_err[0], h_bot_1 + 3 * depths_err[0]],
                    y2=[h_bot_1 - 3 * depths_err[0], h_bot_1 - 3 * depths_err[0]], color='tab:purple', alpha=0.2)
    ax.fill_between([t_2_1 - t_1, t_2_2 - t_1], y1=[h_bot_2 + 3 * depths_err[1], h_bot_2 + 3 * depths_err[1]],
                    y2=[h_bot_2 - 3 * depths_err[1], h_bot_2 - 3 * depths_err[1]], color='tab:purple', alpha=0.2)
    # flat bottom
    if ((t_b_1_2 - t_b_1_1) / dur_b_1_err > 1):
        ax.plot([t_b_1_1 - t_1, t_b_1_1 - t_1], s_minmax, '--', c='tab:brown')
        ax.plot([t_b_1_2 - t_1, t_b_1_2 - t_1], s_minmax, '--', c='tab:brown')
        # 1 sigma errors
        ax.fill_between([t_b_1_1 - t_1 - t_1_1_err, t_b_1_1 - t_1 + t_1_1_err],
                        y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:brown', alpha=0.3)
        ax.fill_between([t_b_1_2 - t_1 - t_1_2_err, t_b_1_2 - t_1 + t_1_2_err],
                        y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:brown', alpha=0.3)
        # 3 sigma errors
        ax.fill_between([t_b_1_1 - t_1 - 3 * t_1_1_err, t_b_1_1 - t_1 + 3 * t_1_1_err],
                        y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:brown', alpha=0.2)
        ax.fill_between([t_b_1_2 - t_1 - 3 * t_1_2_err, t_b_1_2 - t_1 + 3 * t_1_2_err],
                        y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:brown', alpha=0.2)
    if ((t_b_2_2 - t_b_2_1) / dur_b_2_err > 1):
        ax.plot([t_b_2_1 - t_1, t_b_2_1 - t_1], s_minmax, '--', c='tab:brown')
        ax.plot([t_b_2_2 - t_1, t_b_2_2 - t_1], s_minmax, '--', c='tab:brown')
        # 1 sigma errors
        ax.fill_between([t_b_2_1 - t_1 - t_2_1_err, t_b_2_1 - t_1 + t_2_1_err],
                        y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:brown', alpha=0.3)
        ax.fill_between([t_b_2_2 - t_1 - t_2_2_err, t_b_2_2 - t_1 + t_2_2_err],
                        y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:brown', alpha=0.3)
        # 3 sigma errors
        ax.fill_between([t_b_2_1 - t_1 - 3 * t_2_1_err, t_b_2_1 - t_1 + 3 * t_2_1_err],
                        y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:brown', alpha=0.2)
        ax.fill_between([t_b_2_2 - t_1 - 3 * t_2_2_err, t_b_2_2 - t_1 + 3 * t_2_2_err],
                        y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:brown', alpha=0.2)
    if ((t_b_1_2 - t_b_1_1) / dur_b_1_err > 1) | ((t_b_2_2 - t_b_2_1) / dur_b_2_err > 1):
        ax.plot([], [], '--', c='tab:brown', label='flat bottom')  # ghost label
    ax.set_xlabel(r'$(time - t_0)\ mod\ P_{orb}$ (d)')
    ax.set_ylabel('normalised flux')
    plt.legend()
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_derivatives(p_orb, f_h, a_h, ph_h, f_he, a_he, ph_he, ecl_indices, save_file=None, show=True):
    """Shows the light curve and three time derivatives with the significant
    points on the curves used to identify the eclipses
    """
    # 0.864 second steps if we work in days and per day units, as with the measure_eclipses_dt function
    t_model = np.linspace(0, 2 * p_orb, 10**6)
    eclipses = False
    if (len(ecl_indices) > 0):
        if (np.shape(ecl_indices)[1] > 0):
            eclipses = True
    if eclipses:
        # get the right time points
        peaks_1_l = t_model[ecl_indices[:, 3]]
        zeros_1_l = t_model[ecl_indices[:, 0]]
        peaks_2_n_l = t_model[ecl_indices[:, 2]]
        minimum_1_l = t_model[ecl_indices[:, 1]]
        peaks_2_p_l = t_model[ecl_indices[:, 4]]
        zeros_1_in_l = t_model[ecl_indices[:, 5]]
        minimum_0 = t_model[ecl_indices[:, 6]]
        zeros_1_in_r = t_model[ecl_indices[:, -6]]
        peaks_2_p_r = t_model[ecl_indices[:, -5]]
        minimum_1_r = t_model[ecl_indices[:, -2]]
        peaks_2_n_r = t_model[ecl_indices[:, -3]]
        zeros_1_r = t_model[ecl_indices[:, -1]]
        peaks_1_r = t_model[ecl_indices[:, -4]]
        # get the corresponding heights
        he_peaks_1_l = tsf.sum_sines(peaks_1_l, f_he, a_he, ph_he, t_shift=False)
        he_zeros_1_l = tsf.sum_sines(zeros_1_l, f_he, a_he, ph_he, t_shift=False)
        he_peaks_2_n_l = tsf.sum_sines(peaks_2_n_l, f_he, a_he, ph_he, t_shift=False)
        he_minimum_1_l = tsf.sum_sines(minimum_1_l, f_he, a_he, ph_he, t_shift=False)
        he_peaks_2_p_l = tsf.sum_sines(peaks_2_p_l, f_he, a_he, ph_he, t_shift=False)
        he_zeros_1_in_l = tsf.sum_sines(zeros_1_in_l, f_he, a_he, ph_he, t_shift=False)
        he_minimum_0 = tsf.sum_sines(minimum_0, f_he, a_he, ph_he, t_shift=False)
        he_zeros_1_in_r = tsf.sum_sines(zeros_1_in_r, f_he, a_he, ph_he, t_shift=False)
        he_peaks_2_p_r = tsf.sum_sines(peaks_2_p_r, f_he, a_he, ph_he, t_shift=False)
        he_minimum_1_r = tsf.sum_sines(minimum_1_r, f_he, a_he, ph_he, t_shift=False)
        he_peaks_2_n_r = tsf.sum_sines(peaks_2_n_r, f_he, a_he, ph_he, t_shift=False)
        he_zeros_1_r = tsf.sum_sines(zeros_1_r, f_he, a_he, ph_he, t_shift=False)
        he_peaks_1_r = tsf.sum_sines(peaks_1_r, f_he, a_he, ph_he, t_shift=False)
        # deriv 1
        h1e_peaks_1_l = tsf.sum_sines_deriv(peaks_1_l, f_he, a_he, ph_he, deriv=1, t_shift=False)
        h1e_zeros_1_l = tsf.sum_sines_deriv(zeros_1_l, f_he, a_he, ph_he, deriv=1, t_shift=False)
        h1e_peaks_2_n_l = tsf.sum_sines_deriv(peaks_2_n_l, f_he, a_he, ph_he, deriv=1, t_shift=False)
        h1e_minimum_1_l = tsf.sum_sines_deriv(minimum_1_l, f_he, a_he, ph_he, deriv=1, t_shift=False)
        h1e_peaks_2_p_l = tsf.sum_sines_deriv(peaks_2_p_l, f_he, a_he, ph_he, deriv=1, t_shift=False)
        h1e_zeros_1_in_l = tsf.sum_sines_deriv(zeros_1_in_l, f_he, a_he, ph_he, deriv=1, t_shift=False)
        h1e_minimum_0 = tsf.sum_sines_deriv(minimum_0, f_he, a_he, ph_he, deriv=1, t_shift=False)
        h1e_zeros_1_in_r = tsf.sum_sines_deriv(zeros_1_in_r, f_he, a_he, ph_he, deriv=1, t_shift=False)
        h1e_peaks_2_p_r = tsf.sum_sines_deriv(peaks_2_p_r, f_he, a_he, ph_he, deriv=1, t_shift=False)
        h1e_minimum_1_r = tsf.sum_sines_deriv(minimum_1_r, f_he, a_he, ph_he, deriv=1, t_shift=False)
        h1e_peaks_2_n_r = tsf.sum_sines_deriv(peaks_2_n_r, f_he, a_he, ph_he, deriv=1, t_shift=False)
        h1e_zeros_1_r = tsf.sum_sines_deriv(zeros_1_r, f_he, a_he, ph_he, deriv=1, t_shift=False)
        h1e_peaks_1_r = tsf.sum_sines_deriv(peaks_1_r, f_he, a_he, ph_he, deriv=1, t_shift=False)
        # deriv 2
        h2e_peaks_1_l = tsf.sum_sines_deriv(peaks_1_l, f_he, a_he, ph_he, deriv=2, t_shift=False)
        h2e_zeros_1_l = tsf.sum_sines_deriv(zeros_1_l, f_he, a_he, ph_he, deriv=2, t_shift=False)
        h2e_peaks_2_n_l = tsf.sum_sines_deriv(peaks_2_n_l, f_he, a_he, ph_he, deriv=2, t_shift=False)
        h2e_minimum_1_l = tsf.sum_sines_deriv(minimum_1_l, f_he, a_he, ph_he, deriv=2, t_shift=False)
        h2e_peaks_2_p_l = tsf.sum_sines_deriv(peaks_2_p_l, f_he, a_he, ph_he, deriv=2, t_shift=False)
        h2e_zeros_1_in_l = tsf.sum_sines_deriv(zeros_1_in_l, f_he, a_he, ph_he, deriv=2, t_shift=False)
        h2e_minimum_0 = tsf.sum_sines_deriv(minimum_0, f_he, a_he, ph_he, deriv=2, t_shift=False)
        h2e_zeros_1_in_r = tsf.sum_sines_deriv(zeros_1_in_r, f_he, a_he, ph_he, deriv=2, t_shift=False)
        h2e_peaks_2_p_r = tsf.sum_sines_deriv(peaks_2_p_r, f_he, a_he, ph_he, deriv=2, t_shift=False)
        h2e_minimum_1_r = tsf.sum_sines_deriv(minimum_1_r, f_he, a_he, ph_he, deriv=2, t_shift=False)
        h2e_peaks_2_n_r = tsf.sum_sines_deriv(peaks_2_n_r, f_he, a_he, ph_he, deriv=2, t_shift=False)
        h2e_zeros_1_r = tsf.sum_sines_deriv(zeros_1_r, f_he, a_he, ph_he, deriv=2, t_shift=False)
        h2e_peaks_1_r = tsf.sum_sines_deriv(peaks_1_r, f_he, a_he, ph_he, deriv=2, t_shift=False)
    # make a timeframe from 0 to two P to catch both eclipses in full if present
    t_model = np.arange(0, 2 * p_orb + 0.0001, 0.0001)  # this is 10x fewer points, thus much faster
    model_h = tsf.sum_sines(t_model, f_h, a_h, ph_h, t_shift=False)
    model_he = tsf.sum_sines(t_model, f_he, a_he, ph_he, t_shift=False)
    # analytic derivatives
    deriv_1 = tsf.sum_sines_deriv(t_model, f_h, a_h, ph_h, deriv=1, t_shift=False)
    deriv_2 = tsf.sum_sines_deriv(t_model, f_h, a_h, ph_h, deriv=2, t_shift=False)
    deriv_1e = tsf.sum_sines_deriv(t_model, f_he, a_he, ph_he, deriv=1, t_shift=False)
    deriv_2e = tsf.sum_sines_deriv(t_model, f_he, a_he, ph_he, deriv=2, t_shift=False)
    fig, ax = plt.subplots(nrows=3, sharex=True)
    ax[0].plot(t_model, model_he)
    ax[0].plot(t_model, model_h, c='grey', alpha=0.4)
    if eclipses:
        ax[0].scatter(peaks_1_l, he_peaks_1_l, c='tab:blue', marker='o', label='peaks_1')
        ax[0].scatter(peaks_1_r, he_peaks_1_r, c='tab:blue', marker='o')
        ax[0].scatter(zeros_1_l, he_zeros_1_l, c='tab:orange', marker='>', label='zeros_1')
        ax[0].scatter(zeros_1_r, he_zeros_1_r, c='tab:orange', marker='<')
        ax[0].scatter(peaks_2_n_l, he_peaks_2_n_l, c='tab:green', marker='v', label='peaks_2_n')
        ax[0].scatter(peaks_2_n_r, he_peaks_2_n_r, c='tab:green', marker='v')
        ax[0].scatter(minimum_1_l, he_minimum_1_l, c='tab:red', marker='d', label='minimum_1')
        ax[0].scatter(minimum_1_r, he_minimum_1_r, c='tab:red', marker='d')
        ax[0].scatter(peaks_2_p_l, he_peaks_2_p_l, c='tab:purple', marker='^', label='peaks_2_p')
        ax[0].scatter(peaks_2_p_r, he_peaks_2_p_r, c='tab:purple', marker='^')
        ax[0].scatter(zeros_1_in_l, he_zeros_1_in_l, c='tab:pink', marker='<', label='zeros_1_in')
        ax[0].scatter(zeros_1_in_r, he_zeros_1_in_r, c='tab:pink', marker='>')
        ax[0].scatter(minimum_0, he_minimum_0, c='tab:brown', marker='|', label='minimum_0')
    ax[0].set_ylabel(r'$\mathscr{l}$')
    ax[0].legend()
    ax[1].plot(t_model, deriv_1e)
    ax[1].plot(t_model, deriv_1, c='grey', alpha=0.4)
    ax[1].plot(t_model, np.zeros(len(t_model)), '--', c='tab:grey')
    if eclipses:
        ax[1].scatter(peaks_1_l, h1e_peaks_1_l, c='tab:blue', marker='o')
        ax[1].scatter(peaks_1_r, h1e_peaks_1_r, c='tab:blue', marker='o')
        ax[1].scatter(zeros_1_l, h1e_zeros_1_l, c='tab:orange', marker='>')
        ax[1].scatter(zeros_1_r, h1e_zeros_1_r, c='tab:orange', marker='<')
        ax[1].scatter(peaks_2_n_l, h1e_peaks_2_n_l, c='tab:green', marker='v')
        ax[1].scatter(peaks_2_n_r, h1e_peaks_2_n_r, c='tab:green', marker='v')
        ax[1].scatter(minimum_1_l, h1e_minimum_1_l, c='tab:red', marker='d')
        ax[1].scatter(minimum_1_r, h1e_minimum_1_r, c='tab:red', marker='d')
        ax[1].scatter(peaks_2_p_l, h1e_peaks_2_p_l, c='tab:purple', marker='^')
        ax[1].scatter(peaks_2_p_r, h1e_peaks_2_p_r, c='tab:purple', marker='^')
        ax[1].scatter(zeros_1_in_l, h1e_zeros_1_in_l, c='tab:pink', marker='<')
        ax[1].scatter(zeros_1_in_r, h1e_zeros_1_in_r, c='tab:pink', marker='>')
        ax[1].scatter(minimum_0, h1e_minimum_0, c='tab:brown', marker='|')
    ax[1].set_ylabel(r'$\frac{d\mathscr{l}}{dt}$')
    ax[2].plot(t_model, deriv_2e)
    ax[2].plot(t_model, deriv_2, c='grey', alpha=0.4)
    ax[2].plot(t_model, np.zeros(len(t_model)), '--', c='tab:grey')
    if eclipses:
        ax[2].scatter(peaks_1_l, h2e_peaks_1_l, c='tab:blue', marker='o')
        ax[2].scatter(peaks_1_r, h2e_peaks_1_r, c='tab:blue', marker='o')
        ax[2].scatter(zeros_1_l, h2e_zeros_1_l, c='tab:orange', marker='>')
        ax[2].scatter(zeros_1_r, h2e_zeros_1_r, c='tab:orange', marker='<')
        ax[2].scatter(peaks_2_n_l, h2e_peaks_2_n_l, c='tab:green', marker='v')
        ax[2].scatter(peaks_2_n_r, h2e_peaks_2_n_r, c='tab:green', marker='v')
        ax[2].scatter(minimum_1_l, h2e_minimum_1_l, c='tab:red', marker='d')
        ax[2].scatter(minimum_1_r, h2e_minimum_1_r, c='tab:red', marker='d')
        ax[2].scatter(peaks_2_p_l, h2e_peaks_2_p_l, c='tab:purple', marker='^')
        ax[2].scatter(peaks_2_p_r, h2e_peaks_2_p_r, c='tab:purple', marker='^')
        ax[2].scatter(zeros_1_in_l, h2e_zeros_1_in_l, c='tab:pink', marker='<')
        ax[2].scatter(zeros_1_in_r, h2e_zeros_1_in_r, c='tab:pink', marker='>')
        ax[2].scatter(minimum_0, h2e_minimum_0, c='tab:brown', marker='|')
    ax[2].set_ylabel(r'$\frac{d^2\mathscr{l}}{dt^2}$')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_empirical_model(times, signal, p_orb, timings, depths, const, slope, f_n, a_n, ph_n, timings_em, depths_em,
                            timings_err, depths_err, i_sectors, save_file=None, show=True):
    """Shows the initial and final simple empirical cubic function eclipse model
    """
    # unpack/define parameters
    t_zero_init = timings[0]
    t_zero = timings_em[0]
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings_em
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err = timings_err
    depth_1, depth_2 = depths_em
    depth_1_err, depth_2_err = depths_err
    dur_b_1_err = np.sqrt(t_1_1_err**2 + t_1_2_err**2)
    dur_b_2_err = np.sqrt(t_2_1_err**2 + t_2_2_err**2)
    # plotting bounds
    t_ext_1 = min(timings[2] - t_zero_init, t_1_1 - t_zero)
    t_ext_2 = max(timings[3] - t_zero_init, t_1_2 - t_zero)
    # make the model times array, one full period plus the primary eclipse halves
    t_extended, ext_left, ext_right = tsf.fold_time_series(times, p_orb, t_zero, t_ext_1=t_ext_1, t_ext_2=t_ext_2)
    s_extended = np.concatenate((signal[ext_left], signal, signal[ext_right]))
    sorter = np.argsort(t_extended)
    # sinusoid and linear models
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    model_sinusoid = np.concatenate((model_sinusoid[ext_left], model_sinusoid, model_sinusoid[ext_right]))
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_linear = np.concatenate((model_linear[ext_left], model_linear, model_linear[ext_right]))
    model_sin_lin = model_sinusoid + model_linear
    # cubic model - get the parameters for the cubics from the fit parameters
    mid_1 = (timings[2] + timings[3]) / 2
    mid_2 = (timings[4] + timings[5]) / 2
    model_ecl_init = tsfit.eclipse_empirical_lc(times, p_orb, mid_1, mid_2, timings[2], timings[4],
                                                timings[6], timings[8], depths[0], depths[1])
    model_ecl_init = np.concatenate((model_ecl_init[ext_left], model_ecl_init, model_ecl_init[ext_right]))
    model_ecl = tsfit.eclipse_empirical_lc(times, p_orb, t_1, t_2, t_1_1, t_2_1, t_b_1_1, t_b_2_1, depth_1, depth_2)
    model_ecl = np.concatenate((model_ecl[ext_left], model_ecl, model_ecl[ext_right]))
    model_ecl_sin_lin = model_ecl + model_sinusoid + model_linear
    # residuals
    resid_ecl = s_extended - model_ecl - 1
    resid_full = s_extended - model_ecl - 1 - model_sinusoid - model_linear
    # some plotting parameters
    s_minmax = np.array([np.min(signal), np.max(signal)])
    s_minmax_r = np.array([np.min(resid_ecl), np.max(resid_ecl)])
    # translate for plotting
    timings = timings - t_zero_init
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings_em - t_zero
    # plot
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].scatter(t_extended, s_extended, marker='.', label='original signal')
    ax[0].plot(t_extended[sorter], model_ecl_sin_lin[sorter] + 1, c='tab:grey', alpha=0.8,
               label='final (linear + sinusoid + empirical eclipse) model')
    ax[0].plot(t_extended[sorter], model_ecl_init[sorter] + 1, c='tab:orange', label='initial empirical eclipse model')
    ax[0].plot(t_extended[sorter], model_ecl[sorter] + 1, c='tab:red', label='final empirical eclipse model')
    ax[0].plot([timings[2], timings[2]], s_minmax, ':', c='tab:grey', label='previous eclipse edges (harmonics)')
    ax[0].plot([timings[3], timings[3]], s_minmax, ':', c='tab:grey')
    ax[0].plot([timings[4], timings[4]], s_minmax, ':', c='tab:grey')
    ax[0].plot([timings[5], timings[5]], s_minmax, ':', c='tab:grey')
    ax[0].plot([t_1_1, t_1_1], s_minmax, '--', c='tab:purple', label='eclipse edges (cubics)')
    ax[0].plot([t_1_2, t_1_2], s_minmax, '--', c='tab:purple')
    ax[0].plot([t_2_1, t_2_1], s_minmax, '--', c='tab:purple')
    ax[0].plot([t_2_2, t_2_2], s_minmax, '--', c='tab:purple')
    ax[0].plot([t_1_1, t_1_2], [1, 1], '--', c='tab:purple')
    ax[0].plot([t_1_1, t_1_2], [1 - depth_1, 1 - depth_1], '--', c='tab:purple')
    ax[0].plot([t_2_1, t_2_2], [1, 1], '--', c='tab:purple')
    ax[0].plot([t_2_1, t_2_2], [1 - depth_2, 1 - depth_2], '--', c='tab:purple')
    # 1 sigma errors
    ax[0].fill_between([t_1_1 - t_1_1_err, t_1_1 + t_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.3, label=r'1 and 3 $\sigma$ error')
    ax[0].fill_between([t_1_2 - t_1_2_err, t_1_2 + t_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.3)
    ax[0].fill_between([t_2_1 - t_2_1_err, t_2_1 + t_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.3)
    ax[0].fill_between([t_2_2 - t_2_2_err, t_2_2 + t_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.3)
    ax[0].fill_between([t_1_1, t_1_2], y1=[1 - depth_1 + depth_1_err, 1 - depth_1 + depth_1_err],
                       y2=[1 - depth_1 - depth_1_err, 1 - depth_1 - depth_1_err], color='tab:purple', alpha=0.3)
    ax[0].fill_between([t_2_1, t_2_2], y1=[1 - depth_2 + depth_2_err, 1 - depth_2 + depth_2_err],
                       y2=[1 - depth_2 - depth_2_err, 1 - depth_2 - depth_2_err], color='tab:purple', alpha=0.3)
    # 3 sigma errors
    ax[0].fill_between([t_1_1 - 3 * t_1_1_err, t_1_1 + 3 * t_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.2)
    ax[0].fill_between([t_1_2 - 3 * t_1_2_err, t_1_2 + 3 * t_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.2)
    ax[0].fill_between([t_2_1 - 3 * t_2_1_err, t_2_1 + 3 * t_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.2)
    ax[0].fill_between([t_2_2 - 3 * t_2_2_err, t_2_2 + 3 * t_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.2)
    ax[0].fill_between([t_1_1, t_1_2], y1=[1 - depth_1 + 3 * depth_1_err, 1 - depth_1 + 3 * depth_1_err],
                       y2=[1 - depth_1 - 3 * depth_1_err, 1 - depth_1 - 3 * depth_1_err],
                       color='tab:purple', alpha=0.2)
    ax[0].fill_between([t_2_1, t_2_2], y1=[1 - depth_2 + 3 * depth_2_err, 1 - depth_2 + 3 * depth_2_err],
                       y2=[1 - depth_2 - 3 * depth_2_err, 1 - depth_2 - 3 * depth_2_err],
                       color='tab:purple', alpha=0.2)
    # flat bottom
    if ((t_b_1_2 - t_b_1_1) / dur_b_1_err > 1):
        ax[0].plot([t_b_1_1, t_b_1_1], s_minmax, '--', c='tab:brown')
        ax[0].plot([t_b_1_2, t_b_1_2], s_minmax, '--', c='tab:brown')
        # 1 sigma errors
        ax[0].fill_between([t_b_1_1 - t_1_1_err, t_b_1_1 + t_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.3)
        ax[0].fill_between([t_b_1_2 - t_1_2_err, t_b_1_2 + t_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.3)
        # 3 sigma errors
        ax[0].fill_between([t_b_1_1 - 3 * t_1_1_err, t_b_1_1 + 3 * t_1_1_err], y1=s_minmax[[0, 0]],
                           y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.2)
        ax[0].fill_between([t_b_1_2 - 3 * t_1_2_err, t_b_1_2 + 3 * t_1_2_err], y1=s_minmax[[0, 0]],
                           y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.2)
    if ((t_b_2_2 - t_b_2_1) / dur_b_2_err > 1):
        ax[0].plot([t_b_2_1, t_b_2_1], s_minmax, '--', c='tab:brown')
        ax[0].plot([t_b_2_2, t_b_2_2], s_minmax, '--', c='tab:brown')
        # 1 sigma errors
        ax[0].fill_between([t_b_2_1 - t_2_1_err, t_b_2_1 + t_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.3)
        ax[0].fill_between([t_b_2_2 - t_2_2_err, t_b_2_2 + t_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.3)
        # 3 sigma errors
        ax[0].fill_between([t_b_2_1 - 3 * t_2_1_err, t_b_2_1 + 3 * t_2_1_err], y1=s_minmax[[0, 0]],
                           y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.2)
        ax[0].fill_between([t_b_2_2 - 3 * t_2_2_err, t_b_2_2 + 3 * t_2_2_err], y1=s_minmax[[0, 0]],
                           y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.2)
    if ((t_b_1_2 - t_b_1_1) / dur_b_1_err > 1) | ((t_b_2_2 - t_b_2_1) / dur_b_2_err > 1):
        ax[0].plot([], [], '--', c='tab:brown', label='flat bottom')  # ghost label
    ax[0].set_ylabel('normalised flux')
    ax[0].legend()
    # residuals subplot
    ax[1].scatter(t_extended, resid_ecl, marker='.', label='(linear + sinusoid) model residual')
    ax[1].scatter(t_extended, resid_full, marker='.', label='(linear + sinusoid + eclipse) model residual')
    ax[1].plot(t_extended[sorter], model_sin_lin[sorter], c='tab:grey', alpha=0.8, label='(linear + sinusoid) model')
    ax[1].plot([timings[2], timings[2]], s_minmax_r, ':', c='tab:grey', label='previous eclipse edges (harmonics)')
    ax[1].plot([timings[3], timings[3]], s_minmax_r, ':', c='tab:grey')
    ax[1].plot([timings[4], timings[4]], s_minmax_r, ':', c='tab:grey')
    ax[1].plot([timings[5], timings[5]], s_minmax_r, ':', c='tab:grey')
    ax[1].plot([t_1_1, t_1_1], s_minmax_r, '--', c='tab:purple', label='eclipse edges (cubics)')
    ax[1].plot([t_1_2, t_1_2], s_minmax_r, '--', c='tab:purple')
    ax[1].plot([t_2_1, t_2_1], s_minmax_r, '--', c='tab:purple')
    ax[1].plot([t_2_2, t_2_2], s_minmax_r, '--', c='tab:purple')
    ax[1].set_xlabel(r'$(time - t_0) mod(P_{orb})$ (d)')
    ax[1].set_ylabel('normalised flux')
    ax[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_dists_eclipse_parameters(e, w, i, r_sum, r_rat, sb_rat, e_vals, w_vals, i_vals, r_sum_vals, r_rat_vals,
                                  sb_rat_vals):
    """Shows the histograms resulting from the input distributions
    and the hdi_prob=0.683 and hdi_prob=0.997 bounds resulting from the HDIs

    Note: produces several plots
    """
    cos_w = np.cos(w)
    sin_w = np.sin(w)
    # inclination
    i_interval = az.hdi(i_vals, hdi_prob=0.683)
    i_bounds = az.hdi(i_vals, hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist(i_vals / np.pi * 180, bins=50, label='vary fit input')
    ax.plot([i / np.pi * 180, i / np.pi * 180], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([i_interval[0] / np.pi * 180, i_interval[0] / np.pi * 180], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([i_interval[1] / np.pi * 180, i_interval[1] / np.pi * 180], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([i_bounds[0] / np.pi * 180, i_bounds[0] / np.pi * 180], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([i_bounds[1] / np.pi * 180, i_bounds[1] / np.pi * 180], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('inclination (deg)')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # eccentricity
    e_interval = az.hdi(e_vals, hdi_prob=0.683)
    e_bounds = az.hdi(e_vals, hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist(e_vals, bins=50, label='vary fit input')
    ax.plot([e, e], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([e_interval[0], e_interval[0]], [0, np.max(hist[0])], c='tab:orange', label='hdi_prob=0.683')
    ax.plot([e_interval[1], e_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([e_bounds[0], e_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--', label='hdi_prob=0.997')
    ax.plot([e_bounds[1], e_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('eccentricity')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # e*np.cos(w)
    ecosw_interval = az.hdi(e_vals * np.cos(w_vals), hdi_prob=0.683)
    ecosw_bounds = az.hdi(e_vals * np.cos(w_vals), hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist(e_vals * np.cos(w_vals), bins=50, label='vary fit input')
    ax.plot([e * cos_w, e * cos_w], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([ecosw_interval[0], ecosw_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([ecosw_interval[1], ecosw_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([ecosw_bounds[0], ecosw_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([ecosw_bounds[1], ecosw_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('e cos(w)')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # e*np.sin(w)
    esinw_interval = az.hdi(e_vals * np.sin(w_vals), hdi_prob=0.683)
    esinw_bounds = az.hdi(e_vals * np.sin(w_vals), hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist(e_vals * np.sin(w_vals), bins=50, label='vary fit input')
    ax.plot([e * sin_w, e * sin_w], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([esinw_interval[0], esinw_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([esinw_interval[1], esinw_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([esinw_bounds[0], esinw_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([esinw_bounds[1], esinw_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('e sin(w)')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # omega (use same logic as in error_estimates_hdi)
    if (abs(w / np.pi * 180 - 180) > 80) & (abs(w / np.pi * 180 - 180) < 100):
        w_interval = az.hdi(w_vals, hdi_prob=0.683, multimodal=True)
        w_bounds = az.hdi(w_vals, hdi_prob=0.997, multimodal=True)
        if (len(w_interval) == 1):
            w_interval = w_interval[0]
            # w_errs = np.array([w - w_interval[0], w_interval[1] - w])
        else:
            interval_size = w_interval[:, 1] - w_interval[:, 0]
            sorter = np.argsort(interval_size)
            w_interval = w_interval[sorter[-2:]]  # pick only the largest two intervals
            # sign of (w - w_interval) only changes if w is in the interval
            # w_in_interval = (np.sign((w - w_interval)[:, 0] * (w - w_interval)[:, 1]) == -1)
            # w_errs = np.array([w - w_interval[w_in_interval][0, 0], w_interval[w_in_interval][0, 1] - w])
        if (len(w_bounds) == 1):
            w_bounds = w_bounds[0]
        else:
            bounds_size = w_bounds[:, 1] - w_bounds[:, 0]
            sorter = np.argsort(bounds_size)
            w_bounds = w_bounds[sorter[-2:]]  # pick only the largest two intervals
    else:
        w_interval = az.hdi(w_vals - np.pi, hdi_prob=0.683, circular=True) + np.pi
        w_bounds = az.hdi(w_vals - np.pi, hdi_prob=0.997, circular=True) + np.pi
        # w_errs = np.array([min(abs(w - w_interval[0]), abs(2*np.pi + w - w_interval[0])),
        #                    min(abs(w_interval[1] - w), abs(2*np.pi + w_interval[1] - w))])
    # plot
    fig, ax = plt.subplots()
    hist = ax.hist(w_vals / np.pi * 180, bins=50, label='vary fit input')
    ax.plot([w / np.pi * 180, w / np.pi * 180], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    if (abs(w / np.pi * 180 - 180) > 80) & (abs(w / np.pi * 180 - 180) < 100):
        ax.plot([(2 * np.pi - w) / np.pi * 180, (2 * np.pi - w) / np.pi * 180], [0, np.max(hist[0])], c='tab:pink',
                label='mirrored best fit value')
    if (len(np.shape(w_interval)) > 1):
        w_in_interval = (np.sign((w - w_interval)[:, 0] * (w - w_interval)[:, 1]) == -1)
        w_not_in_interval = (np.sign((w - w_interval)[:, 0] * (w - w_interval)[:, 1]) == 1)
        ax.plot([w_interval[w_in_interval, 0] / np.pi * 180, w_interval[w_in_interval, 0] / np.pi * 180],
                [0, np.max(hist[0])],
                c='tab:orange', label='hdi_prob=0.683')
        ax.plot([w_interval[w_in_interval, 1] / np.pi * 180, w_interval[w_in_interval, 1] / np.pi * 180],
                [0, np.max(hist[0])],
                c='tab:orange')
        ax.plot([w_interval[w_not_in_interval, 0] / np.pi * 180, w_interval[w_not_in_interval, 0] / np.pi * 180],
                [0, np.max(hist[0])], c='tab:orange', linestyle='--', label='hdi_prob=0.683')
        ax.plot([w_interval[w_not_in_interval, 1] / np.pi * 180, w_interval[w_not_in_interval, 1] / np.pi * 180],
                [0, np.max(hist[0])], c='tab:orange', linestyle='--')
    else:
        mask_int = (np.sign((w / np.pi * 180 - 180) * (w_interval / np.pi * 180 - 180)) < 0)
        w_interval_plot = np.copy(w_interval)
        w_interval_plot[mask_int] = w_interval[mask_int] + np.sign(w / np.pi * 180 - 180) * 2 * np.pi
        ax.plot([w_interval_plot[0] / np.pi * 180, w_interval_plot[0] / np.pi * 180], [0, np.max(hist[0])],
                c='tab:orange',
                label='hdi_prob=0.683')
        ax.plot([w_interval_plot[1] / np.pi * 180, w_interval_plot[1] / np.pi * 180], [0, np.max(hist[0])],
                c='tab:orange')
    if (len(np.shape(w_bounds)) > 1):
        w_in_interval = (np.sign((w - w_bounds)[:, 0] * (w - w_bounds)[:, 1]) == -1)
        w_not_in_interval = (np.sign((w - w_bounds)[:, 0] * (w - w_bounds)[:, 1]) == 1)
        ax.plot([w_bounds[w_in_interval, 0] / np.pi * 180, w_bounds[w_in_interval, 0] / np.pi * 180],
                [0, np.max(hist[0])],
                c='tab:grey', linestyle='--', label='hdi_prob=0.683')
        ax.plot([w_bounds[w_in_interval, 1] / np.pi * 180, w_bounds[w_in_interval, 1] / np.pi * 180],
                [0, np.max(hist[0])],
                c='tab:grey', linestyle='--')
        ax.plot([w_bounds[w_not_in_interval, 0] / np.pi * 180, w_bounds[w_not_in_interval, 0] / np.pi * 180],
                [0, np.max(hist[0])], c='tab:grey', linestyle=':', label='hdi_prob=0.683')
        ax.plot([w_bounds[w_not_in_interval, 1] / np.pi * 180, w_bounds[w_not_in_interval, 1] / np.pi * 180],
                [0, np.max(hist[0])], c='tab:grey', linestyle=':')
    else:
        mask_bnd = (np.sign((w / np.pi * 180 - 180) * (w_bounds / np.pi * 180 - 180)) < 0)
        w_bounds_plot = np.copy(w_bounds)
        w_bounds_plot[mask_bnd] = w_bounds[mask_bnd] + np.sign(w / np.pi * 180 - 180) * 2 * np.pi
        ax.plot([w_bounds_plot[0] / np.pi * 180, w_bounds_plot[0] / np.pi * 180], [0, np.max(hist[0])], c='tab:grey',
                linestyle='--', label='hdi_prob=0.683')
        ax.plot([w_bounds_plot[1] / np.pi * 180, w_bounds_plot[1] / np.pi * 180], [0, np.max(hist[0])], c='tab:grey',
                linestyle='--')
    ax.set_xlabel('omega (deg)')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # r_sum
    rsumsma_interval = az.hdi(r_sum_vals, hdi_prob=0.683)
    rsumsma_bounds = az.hdi(r_sum_vals, hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist(r_sum_vals, bins=50, label='vary fit input')
    ax.plot([r_sum, r_sum], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([rsumsma_interval[0], rsumsma_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([rsumsma_interval[1], rsumsma_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([rsumsma_bounds[0], rsumsma_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([rsumsma_bounds[1], rsumsma_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('(r1+r2)/a')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # r_rat
    rratio_interval = az.hdi(r_rat_vals, hdi_prob=0.683)
    rratio_bounds = az.hdi(r_rat_vals, hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist((r_rat_vals), bins=50, label='vary fit input')
    ax.plot(([r_rat, r_rat]), [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([rratio_interval[0], rratio_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([rratio_interval[1], rratio_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([rratio_bounds[0], rratio_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([rratio_bounds[1], rratio_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('r_rat')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # log(r_rat)
    log_rratio_interval = az.hdi(np.log10(r_rat_vals), hdi_prob=0.683)
    log_rratio_bounds = az.hdi(np.log10(r_rat_vals), hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist(np.log10(r_rat_vals), bins=50, label='vary fit input')
    ax.plot(np.log10([r_rat, r_rat]), [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([log_rratio_interval[0], log_rratio_interval[0]], [0, np.max(hist[0])],
            c='tab:orange', label='hdi_prob=0.683')
    ax.plot([log_rratio_interval[1], log_rratio_interval[1]], [0, np.max(hist[0])],
            c='tab:orange')
    ax.plot([log_rratio_bounds[0], log_rratio_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([log_rratio_bounds[1], log_rratio_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('log(r_rat)')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # sb_rat
    sbratio_interval = az.hdi(sb_rat_vals, hdi_prob=0.683)
    sbratio_bounds = az.hdi(sb_rat_vals, hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist((sb_rat_vals), bins=50, label='vary fit input')
    ax.plot(([sb_rat, sb_rat]), [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([sbratio_interval[0], sbratio_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([sbratio_interval[1], sbratio_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([sbratio_bounds[0], sbratio_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([sbratio_bounds[1], sbratio_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('sb_rat')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # log(sb_rat)
    log_sbratio_interval = az.hdi(np.log10(sb_rat_vals), hdi_prob=0.683)
    log_sbratio_bounds = az.hdi(np.log10(sb_rat_vals), hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist(np.log10(sb_rat_vals), bins=50, label='vary fit input')
    ax.plot(np.log10([sb_rat, sb_rat]), [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([log_sbratio_interval[0], log_sbratio_interval[0]], [0, np.max(hist[0])],
            c='tab:orange', label='hdi_prob=0.683')
    ax.plot([log_sbratio_interval[1], log_sbratio_interval[1]], [0, np.max(hist[0])],
            c='tab:orange')
    ax.plot([log_sbratio_bounds[0], log_sbratio_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([log_sbratio_bounds[1], log_sbratio_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('log(sb_rat)')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return


def plot_corner_eclipse_elements(p_orb, timings, depths, ecl_par, dists_in, dists_out, save_file=None, show=True):
    """Shows the corner plots resulting from the input distributions
    
    Note: produces several plots
    """
    r2d = 180 / np.pi  # radians to degrees
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings
    d_1, d_2 = depths
    e, w, i, r_sum, r_rat, sb_rat = ecl_par
    p_vals, t_1_vals, t_2_vals, t_1_1_vals, t_1_2_vals, t_2_1_vals, t_2_2_vals = dists_in[:7]
    t_b_1_1_vals, t_b_1_2_vals, t_b_2_1_vals, t_b_2_2_vals, d_1_vals, d_2_vals = dists_in[7:]
    e_vals, w_vals, i_vals, r_sum_vals, r_rat_vals, sb_rat_vals = dists_out
    # for if the w-distribution crosses over at 2 pi
    if (abs(w / np.pi * 180 - 180) > 80) & (abs(w / np.pi * 180 - 180) < 100):
        w_vals = np.copy(w_vals)
    else:
        w_vals = np.copy(w_vals)
        mask = (np.sign((w / np.pi * 180 - 180) * (w_vals / np.pi * 180 - 180)) < 0)
        w_vals[mask] = w_vals[mask] + np.sign(w / np.pi * 180 - 180) * 2 * np.pi
    # transform some params
    ecosw, esinw = e * np.cos(w), e * np.sin(w)
    cosi = np.cos(i)
    phi_0 = af.phi_0_from_r_sum_sma(e, i, r_sum)
    ecl_par_a = np.array([ecosw, esinw, cosi, phi_0, r_rat, sb_rat])
    ecl_par_b = np.array([e, w * r2d, i * r2d, r_sum, r_rat, sb_rat])
    ecosw_vals = e_vals * np.cos(w_vals)
    esinw_vals = e_vals * np.sin(w_vals)
    cosi_vals = np.cos(i_vals)
    phi_0_vals = af.phi_0_from_r_sum_sma(e, i, r_sum_vals)
    # input corner plot
    value_names = np.array([r'$p_{orb}$', r'$t_1$', r'$t_2$', r'$t_{1,1}$', r'$t_{1,2}$', r'$t_{2,1}$', r'$t_{2,2}$',
                            r'$t_{b,1,1}$', r'$t_{b,1,2}$', r'$t_{b,2,1}$', r'$t_{b,2,2}$', r'$depth_1$', r'$depth_2$'])
    values = np.array([p_orb, t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, d_1, d_2])
    dist_data = np.column_stack((p_vals, t_1_vals, t_2_vals, t_1_1_vals, t_1_2_vals, t_2_1_vals, t_2_2_vals,
                                 t_b_1_1_vals, t_b_1_2_vals, t_b_2_1_vals, t_b_2_2_vals, d_1_vals, d_2_vals))
    value_range = np.max(dist_data, axis=0) - np.min(dist_data, axis=0)
    nonzero_range = (value_range != 0) & (value_range != np.inf)  # nonzero and finite
    fig = corner.corner(dist_data[:, nonzero_range], truths=values[nonzero_range],
                        labels=value_names[nonzero_range], quiet=True)
    if not np.all(nonzero_range):
        fig.suptitle(f'Input distributions ({np.sum(nonzero_range)} of {len(nonzero_range)} shown)')
    else:
        fig.suptitle('Input distributions')
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    if save_file is not None:
        if save_file.endswith('.png'):
            fig_save_file = save_file.replace('.png', '_in.png')
        else:
            fig_save_file = save_file + '_in.png'
        fig.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    # output distributions corner plot
    value_names = np.array(['e cos(w)', 'e sin(w)', 'cos(i)', 'phi_0', r'$\frac{r_2}{r_1}$', r'$\frac{sb_2}{sb_1}$'])
    dist_data = np.column_stack((ecosw_vals, esinw_vals, cosi_vals, phi_0_vals, r_rat_vals, sb_rat_vals))
    value_range = np.max(dist_data, axis=0) - np.min(dist_data, axis=0)
    nonzero_range = (value_range != 0) & (value_range != np.inf)  # nonzero and finite
    fig = corner.corner(dist_data[:, nonzero_range], labels=value_names[nonzero_range], quiet=True)
    corner.overplot_lines(fig, ecl_par_a[nonzero_range], color='tab:blue')
    corner.overplot_points(fig, [ecl_par_a[nonzero_range]], marker='s', color='tab:blue')
    if not np.all(nonzero_range):
        fig.suptitle(f'Output distributions ({np.sum(nonzero_range)} of {len(nonzero_range)} shown)')
    else:
        fig.suptitle('Output distributions')
    plt.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    # also make the other parameterisation corner plot
    value_names = np.array(['e', 'w (deg)', 'i (deg)', r'$\frac{r_1+r_2}{a}$', r'$\frac{r_2}{r_1}$',
                            r'$\frac{sb_2}{sb_1}$'])
    dist_data = np.column_stack((e_vals, w_vals * r2d, i_vals * r2d, r_sum_vals, r_rat_vals, sb_rat_vals))
    value_range = np.max(dist_data, axis=0) - np.min(dist_data, axis=0)
    nonzero_range = (value_range != 0) & (value_range != np.inf)  # nonzero and finite
    fig = corner.corner(dist_data[:, nonzero_range], labels=value_names[nonzero_range], quiet=True)
    corner.overplot_lines(fig, ecl_par_b[nonzero_range], color='tab:blue')
    corner.overplot_points(fig, [ecl_par_b[nonzero_range]], marker='s', color='tab:blue')
    if not np.all(nonzero_range):
        fig.suptitle(f'Output distributions ({np.sum(nonzero_range)} of {len(nonzero_range)} shown)')
    else:
        fig.suptitle('Output distributions')
    plt.tight_layout()
    if save_file is not None:
        if save_file.endswith('.png'):
            fig_save_file = save_file.replace('.png', '_alt.png')
        else:
            fig_save_file = save_file + '_alt.png'
        fig.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_model_sigma(times, signal, p_orb, t_zero, timings, const, slope, f_n, a_n, ph_n, i_sectors,
                        ecl_par, par_i, par_bounds, save_file=None, show=True):
    """Shows the difference one parameter makes in the eclipse model
    for three different values
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings
    # make the model times array, one full period plus the primary eclipse halves
    t_extended, ext_left, ext_right = tsf.fold_time_series(times, p_orb, t_zero,
                                                           t_ext_1=t_1_1 - t_1, t_ext_2=t_1_2 - t_1)
    sorter = np.argsort(t_extended)
    t_mean_ext = np.mean(t_extended)
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    f_h, a_h, ph_h = f_n[harmonics], a_n[harmonics], ph_n[harmonics]
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line
    ecl_signal = np.concatenate((ecl_signal[ext_left], ecl_signal, ecl_signal[ext_right]))
    s_minmax = [np.min(signal), np.max(signal)]
    # determine a lc offset to match the model at the edges
    h_1, h_2 = af.height_at_contact(f_h, a_h, ph_h, t_1_1, t_1_2, t_2_1, t_2_2)
    offset = 1 - (h_1 + h_2) / 2
    # unpack all ecl_par
    ecosw, esinw, cosi, phi_0, rho_0, sb_rat, e, w, i, r_sum, r_rat = ecl_par
    # make the default eclipse model
    model = tsfit.eclipse_physical_lc(t_extended, p_orb, -t_mean_ext, e, w, i, r_sum, r_rat, sb_rat)
    # depending on par_i, we need to do stuff
    par_p = np.copy(ecl_par)
    par_n = np.copy(ecl_par)
    par_p[par_i] = par_bounds[1]
    par_n[par_i] = par_bounds[0]
    if par_i in [0, 1, 2, 3, 4]:
        # do not use the alternative parametrisation, so compute everything again
        par_p[6] = np.sqrt(par_p[0]**2 + par_p[1]**2)
        par_p[7] = np.arctan2(par_p[1], par_p[0]) % (2 * np.pi)
        par_n[6] = np.sqrt(par_n[0]**2 + par_n[1]**2)
        par_n[7] = np.arctan2(par_n[1], par_n[0]) % (2 * np.pi)
        par_p[8] = np.arccos(par_p[2])
        par_n[8] = np.arccos(par_n[2])
        par_p[9] = np.sqrt((1 - np.sin(par_p[8])**2 * np.cos(par_p[3])**2) * (1 - par_p[6]**2))
        par_n[9] = np.sqrt((1 - np.sin(par_n[8])**2 * np.cos(par_n[3])**2) * (1 - par_n[6]**2))
        par_p[10] = af.r_ratio_from_rho_0(par_p[6], par_p[7], par_p[8], par_p[3], par_p[4])
        par_n[10] = af.r_ratio_from_rho_0(par_n[6], par_n[7], par_n[8], par_n[3], par_n[4])
    # make the other models
    ecl_par_p = np.append(par_p[6:], par_p[5])
    ecl_par_n = np.append(par_n[6:], par_n[5])
    model_p = tsfit.eclipse_physical_lc(t_extended, p_orb, -t_mean_ext, *ecl_par_p)
    model_m = tsfit.eclipse_physical_lc(t_extended, p_orb, -t_mean_ext, *ecl_par_n)
    # list of names
    par_names = ['ecosw', 'esinw', 'cosi', 'phi_0', 'r_rat', 'sb_rat', 'e', 'w', 'i', 'r_sum']
    # plot
    fig, ax = plt.subplots()
    ax.scatter(t_extended, ecl_signal + offset, marker='.', label='eclipse signal')
    ax.plot(t_extended[sorter], model[sorter], c='tab:orange', label='best parameters')
    ax.fill_between(t_extended[sorter], y1=model[sorter], y2=model_p[sorter], color='tab:orange', alpha=0.3,
                    label=f'upper bound {par_names[par_i]}')
    ax.fill_between(t_extended[sorter], y1=model[sorter], y2=model_m[sorter], color='tab:purple', alpha=0.3,
                    label=f'lower bound {par_names[par_i]}')
    ax.plot([t_1_1 - t_1, t_1_1 - t_1], s_minmax, '--', c='grey', label=r'eclipse edges')
    ax.plot([t_1_2 - t_1, t_1_2 - t_1], s_minmax, '--', c='grey')
    ax.plot([t_2_1 - t_1, t_2_1 - t_1], s_minmax, '--', c='grey')
    ax.plot([t_2_2 - t_1, t_2_2 - t_1], s_minmax, '--', c='grey')
    ax.set_xlabel(r'$(time - t_0) mod(P_{orb})$ (d)')
    ax.set_ylabel('normalised flux')
    ax.set_title(f'{par_names[par_i]} = {ecl_par[par_i]:1.4f}, bounds: ({par_bounds[0]:1.4f}, {par_bounds[1]:1.4f})')
    plt.legend()
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_physical_model(times, signal, p_orb, t_zero, const_r, slope_r, f_n_r, a_n_r, ph_n_r, ecl_par, passed_r,
                           i_sectors, save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the determination
    of orbital parameters using both the eclipse timings and the ellc light curve
    models over two consecutive fits.
    """
    t_mean = np.mean(times)
    # eclipse signal with disentangled frequencies
    model_linear_r = tsf.linear_curve(times, const_r, slope_r, i_sectors)
    model_sinusoid_r = tsf.sum_sines(times, f_n_r, a_n_r, ph_n_r)
    # model of passed frequencies
    model_r_p = tsf.linear_curve(times, const_r, slope_r, i_sectors)
    model_r_p += tsf.sum_sines(times, f_n_r[passed_r], a_n_r[passed_r], ph_n_r[passed_r])
    # make the ellc model
    ecl_model = tsfit.eclipse_physical_lc(times, p_orb, t_zero, *ecl_par)
    model_sin_lin = model_sinusoid_r + model_linear_r
    model_full = ecl_model + model_sinusoid_r + model_linear_r
    resid_ecl = signal - ecl_model
    resid_full = signal - model_full
    # plot
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot([t_mean, t_mean], [np.min(signal), np.max(signal)], c='grey', alpha=0.3)
    ax[0].scatter(times, signal, marker='.', label='original signal')
    ax[0].plot(times, model_full, c='tab:grey', label='(linear + sinusoid + eclipse) model')
    ax[0].plot(times, ecl_model, c='tab:red', label='eclipse model')
    ax[0].set_ylabel('normalised flux/model')
    ax[0].legend()
    # residuals
    ax[1].plot([t_mean, t_mean], [np.min(resid_ecl), np.max(resid_ecl)], c='grey', alpha=0.3)
    ax[1].scatter(times, resid_ecl, marker='.', label='eclipse model residuals')
    ax[1].scatter(times, resid_full, marker='.', label='(linear + sinusoid + eclipse) model residuals')
    ax[1].plot(times, model_sin_lin, c='tab:grey', label='(linear + sinusoid) model')
    ax[1].plot(times, model_r_p, c='tab:red', label='sinusoids passing criteria')
    ax[1].set_ylabel('residual/model')
    ax[1].set_xlabel('time (d)')
    ax[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_physical_model_h(times, signal, p_orb, t_zero, timings_init, timings, const, slope, f_n, a_n, ph_n,
                             ecl_par_init, ecl_par, passed_r, passed_h, i_sectors, save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the determination
    of orbital parameters using both the eclipse timings and the ellc light curve
    models over two consecutive fits.
    """
    timings_init = timings_init - timings_init[0]
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings - t_zero
    # make the model times array, one full period plus the primary eclipse halves
    t_extended, ext_left, ext_right = tsf.fold_time_series(times, p_orb, t_zero, t_ext_1=t_1_1, t_ext_2=t_1_2)
    s_extended = np.concatenate((signal[ext_left], signal, signal[ext_right]))
    sorter = np.argsort(t_extended)
    # sinusoid and linear models
    model_sinusoid_r = tsf.sum_sines(times, f_n, a_n, ph_n)
    model_linear_r = tsf.linear_curve(times, const, slope, i_sectors)
    # make the eclipse model
    model_ecl_init = tsfit.eclipse_physical_lc(times, p_orb, t_zero, *ecl_par_init)
    model_ecl_init = np.concatenate((model_ecl_init[ext_left], model_ecl_init, model_ecl_init[ext_right]))
    model_ecl = tsfit.eclipse_physical_lc(times, p_orb, t_zero, *ecl_par)
    model_ecl_2 = np.concatenate((model_ecl[ext_left], model_ecl, model_ecl[ext_right]))
    # add residual harmonic sinusoids to model
    model_sin_lin = model_sinusoid_r + model_linear_r
    model_sin_lin = np.concatenate((model_sin_lin[ext_left], model_sin_lin, model_sin_lin[ext_right]))
    model_ecl_sin_lin = model_ecl + model_sinusoid_r + model_linear_r
    model_ecl_sin_lin = np.concatenate((model_ecl_sin_lin[ext_left], model_ecl_sin_lin, model_ecl_sin_lin[ext_right]))
    # residuals
    resid_ecl = signal - model_ecl
    resid_ecl = np.concatenate((resid_ecl[ext_left], resid_ecl, resid_ecl[ext_right]))
    resid_full = signal - model_ecl - model_sinusoid_r - model_linear_r
    resid_full = np.concatenate((resid_full[ext_left], resid_full, resid_full[ext_right]))
    # candidate harmonics in the disentangled frequencies
    model_r_h = tsf.sum_sines(times, f_n[passed_h], a_n[passed_h], ph_n[passed_h])
    model_r_h = np.concatenate((model_r_h[ext_left], model_r_h, model_r_h[ext_right]))
    # model of passed frequencies
    if np.any(passed_r):
        passed_hr = passed_r & passed_h
        model_r_p_h = tsf.sum_sines(times, f_n[passed_hr], a_n[passed_hr], ph_n[passed_hr])
        model_r_p_h = np.concatenate((model_r_p_h[ext_left], model_r_p_h, model_r_p_h[ext_right]))
    else:
        model_r_p_h = np.zeros(len(t_extended))
    # some plotting parameters
    s_minmax = np.array([np.min(signal), np.max(signal)])
    s_minmax_r = np.array([np.min(resid_ecl), np.max(resid_ecl)])
    # plot
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].scatter(t_extended, s_extended, marker='.', label='original signal')
    ax[0].plot(t_extended[sorter], model_ecl_sin_lin[sorter], c='tab:grey', alpha=0.8,
               label='(linear + sinusoid + eclipse) model')
    ax[0].plot(t_extended[sorter], model_ecl_init[sorter], c='tab:orange', label='initial eclipse model')
    ax[0].plot(t_extended[sorter], model_ecl_2[sorter], c='tab:red', label='eclipse model')
    ax[0].plot([timings_init[2], timings_init[2]], s_minmax, ':', c='tab:grey', label='previous eclipse edges (cubics)')
    ax[0].plot([timings_init[3], timings_init[3]], s_minmax, ':', c='tab:grey')
    ax[0].plot([timings_init[4], timings_init[4]], s_minmax, ':', c='tab:grey')
    ax[0].plot([timings_init[5], timings_init[5]], s_minmax, ':', c='tab:grey')
    ax[0].plot([t_1_1, t_1_1], s_minmax, '--', c='grey', label='eclipse edges (physical)')
    ax[0].plot([t_1_2, t_1_2], s_minmax, '--', c='grey')
    ax[0].plot([t_2_1, t_2_1], s_minmax, '--', c='grey')
    ax[0].plot([t_2_2, t_2_2], s_minmax, '--', c='grey')
    ax[0].set_ylabel('normalised flux/model')
    ax[0].legend()
    # residuals
    ax[1].scatter(t_extended, resid_ecl, marker='.', label='(linear + sinusoid) model residual')
    ax[1].scatter(t_extended, resid_full, marker='.', label='(linear + sinusoid + eclipse) model residual')
    ax[1].plot(t_extended[sorter], model_sin_lin[sorter], c='tab:grey', alpha=0.8, label='(linear + sinusoid) model')
    ax[1].plot(t_extended[sorter], model_r_h[sorter], c='tab:brown', alpha=0.8, label='candidate harmonics')
    ax[1].plot(t_extended[sorter], model_r_p_h[sorter], c='tab:red', label='candidate harmonics passing criteria')
    ax[1].plot([timings_init[2], timings_init[2]], s_minmax_r, ':', c='tab:grey',
               label='previous eclipse edges (cubics)')
    ax[1].plot([timings_init[3], timings_init[3]], s_minmax_r, ':', c='tab:grey')
    ax[1].plot([timings_init[4], timings_init[4]], s_minmax_r, ':', c='tab:grey')
    ax[1].plot([timings_init[5], timings_init[5]], s_minmax_r, ':', c='tab:grey')
    ax[1].plot([t_1_1, t_1_1], s_minmax_r, '--', c='grey', label='eclipse edges (physical)')
    ax[1].plot([t_1_2, t_1_2], s_minmax_r, '--', c='grey')
    ax[1].plot([t_2_1, t_2_1], s_minmax_r, '--', c='grey')
    ax[1].plot([t_2_2, t_2_2], s_minmax_r, '--', c='grey')
    ax[1].set_ylabel('residual/model')
    ax[1].set_xlabel(r'$(time - t_0) mod(P_{orb})$ (d)')
    ax[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_pd_leftover_sinusoids(times, signal, p_orb, t_zero, noise_level, const_r, slope_r, f_n_r, a_n_r, ph_n_r,
                               passed_r, param_lc, i_sectors, model='simple', save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the determination
    of orbital parameters using both the eclipse timings and the ellc light curve
    models over two consecutive fits.
    """
    # convert bool mask to indices
    passed_r_i = np.arange(len(f_n_r))[passed_r]
    # eclipse signal with disentangled frequencies
    model_r = tsf.linear_curve(times, const_r, slope_r, i_sectors)
    model_r += tsf.sum_sines(times, f_n_r, a_n_r, ph_n_r)
    # unpack and define parameters
    e, w, i, r_sum, r_rat, sb_rat = param_lc
    f_c, f_s = e**0.5 * np.cos(w), e**0.5 * np.sin(w)
    # make the ellc model
    if (model == 'ellc'):
        ecl_model = tsfit.wrap_ellc_lc(times, p_orb, t_zero, f_c, f_s, i, r_sum, r_rat, sb_rat, 0)
    else:
        ecl_model = tsfit.eclipse_physical_lc(times, p_orb, t_zero, e, w, i, r_sum, r_rat, sb_rat)
    ecl_resid = signal - ecl_model
    # make periodograms
    freqs, ampls = tsf.astropy_scargle(times, ecl_resid)
    freq_range = np.ptp(freqs)
    freqs_1, ampls_1 = tsf.astropy_scargle(times, ecl_resid - model_r)
    snr_threshold = ut.signal_to_noise_threshold(len(signal))
    # plot
    fig, ax = plt.subplots()
    ax.plot(freqs, ampls, label='residual after eclipse model subtraction')
    ax.plot(freqs_1, ampls_1, label='final residual')
    ax.plot(freqs[[0, -1]], [snr_threshold * noise_level, snr_threshold * noise_level], c='tab:grey', alpha=0.7,
            label=f'S/N threshold ({snr_threshold})')
    for k in range(len(f_n_r)):
        if k in passed_r_i:
            ax.plot([f_n_r[k], f_n_r[k]], [0, a_n_r[k]],  c='tab:red')
        else:
            ax.plot([f_n_r[k], f_n_r[k]], [0, a_n_r[k]], c='tab:pink')
    ax.plot([], [], c='tab:red', label='disentangled sinusoids passing criteria')
    ax.plot([], [], c='tab:pink', label='disentangled sinusoids')
    ax.set_xlim(freqs[0] - freq_range * 0.05, freqs[-1] + freq_range * 0.05)
    plt.xlabel('frequency (1/d)')
    plt.ylabel('amplitude')
    plt.legend()
    plt.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_corner_eclipse_mcmc(inf_data, ecosw, esinw, cosi, phi_0, r_rat, sb_rat, const, slope, f_n, a_n, ph_n,
                             save_file=None, show=True):
    """Show the pymc3 physical eclipse parameter sampling results in two corner plots"""
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
    r_rat_ch = inf_data.posterior.r_rat.stack(dim=['chain', 'draw']).to_numpy()
    sb_rat_ch = inf_data.posterior.sb_rat.stack(dim=['chain', 'draw']).to_numpy()
    # parameter transforms
    e_ch = np.sqrt(ecosw_ch**2 + esinw_ch**2)
    w_ch = np.arctan2(esinw_ch, ecosw_ch) % (2 * np.pi)
    i_ch = np.arccos(cosi_ch)
    r_sum_ch = np.sqrt((1 - np.sin(i_ch)**2 * np.cos(phi_0_ch)**2) * (1 - e_ch**2))
    # more parameter transforms
    e = np.sqrt(ecosw**2 + esinw**2)
    w = np.arctan2(esinw, ecosw) % (2 * np.pi)
    i = np.arccos(cosi)
    r_sum = np.sqrt((1 - np.sin(i)**2 * np.cos(phi_0)**2) * (1 - e**2))
    # convert phases to interval [-pi, pi] from [0, 2pi]
    above_pi = (ph_n >= np.pi)
    ph_n[above_pi] = ph_n[above_pi] - 2 * np.pi
    # for if the w-distribution crosses over at 2 pi
    if (abs(w / np.pi * 180) > 80) & (abs(w / np.pi * 180) < 100):
        w_ch = np.copy(w_ch)
    else:
        w_ch = np.copy(w_ch)
        mask = (np.sign((w / np.pi * 180 - 180) * (w_ch / np.pi * 180 - 180)) < 0)
        w_ch[mask] = w_ch[mask] + np.sign(w / np.pi * 180 - 180) * 2 * np.pi
    # corner plot
    value_names = np.array(['e cos(w)', 'e sin(w)', 'cos(i)', 'phi_0', r'$\frac{r_2}{r_1}$', r'$\frac{sb_2}{sb_1}$'])
    values = np.array([ecosw, esinw, cosi, phi_0, r_rat, sb_rat])
    dist_data = np.column_stack((ecosw_ch, esinw_ch, cosi_ch, phi_0_ch, r_rat_ch, sb_rat_ch))
    value_range = np.max(dist_data, axis=0) - np.min(dist_data, axis=0)
    nonzero_range = (value_range != 0) & (value_range != np.inf)  # nonzero and finite
    fig = corner.corner(dist_data[:, nonzero_range], truths=values[nonzero_range],
                        labels=value_names[nonzero_range], quiet=True)
    if not np.all(nonzero_range):
        fig.suptitle(f'Output distributions ({np.sum(nonzero_range)} of {len(nonzero_range)} shown)')
    else:
        fig.suptitle('Output distributions')
    plt.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    # alternative corner plot
    value_names = np.array(['e', 'w (deg)', 'i (deg)', r'$\frac{r_1+r_2}{a}$', r'$\frac{r_2}{r_1}$',
                            r'$\frac{sb_2}{sb_1}$'])
    values = np.array([e, w / np.pi * 180, i / np.pi * 180, r_sum, r_rat, sb_rat])
    dist_data = np.column_stack((e_ch, w_ch / np.pi * 180, i_ch / np.pi * 180, r_sum_ch, r_rat_ch, sb_rat_ch))
    value_range = np.max(dist_data, axis=0) - np.min(dist_data, axis=0)
    nonzero_range = (value_range != 0) & (value_range != np.inf)  # nonzero and finite
    fig = corner.corner(dist_data[:, nonzero_range], truths=values[nonzero_range],
                        labels=value_names[nonzero_range], quiet=True)
    if not np.all(nonzero_range):
        fig.suptitle(f'Output distributions ({np.sum(nonzero_range)} of {len(nonzero_range)} shown)')
    else:
        fig.suptitle('Output distributions')
    plt.tight_layout()
    if save_file is not None:
        if save_file.endswith('.png'):
            fig_save_file = save_file.replace('.png', '_alt.png')
        else:
            fig_save_file = save_file + '_alt.png'
        fig.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_trace_sinusoids(inf_data, const, slope, f_n, a_n, ph_n):
    """Show the pymc3 sampling results in a trace plot"""
    # convert phases to interval [-pi, pi] from [0, 2pi]
    above_pi = (ph_n >= np.pi)
    ph_n[above_pi] = ph_n[above_pi] - 2 * np.pi
    par_lines = [('const', {}, const), ('slope', {}, slope), ('f_n', {}, f_n), ('a_n', {}, a_n), ('ph_n', {}, ph_n)]
    az.plot_trace(inf_data, combined=False, compact=True, rug=True, divergences='top', lines=par_lines)
    return


def plot_pair_harmonics(inf_data, p_orb, const, slope, f_n, a_n, ph_n, save_file=None, show=True):
    """Show the pymc3 sampling results in several pair plots"""
    # convert phases to interval [-pi, pi] from [0, 2pi]
    above_pi = (ph_n >= np.pi)
    ph_n[above_pi] = ph_n[above_pi] - 2 * np.pi
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    ref_values = {'p_orb': p_orb, 'const': const, 'slope': slope,
                  'f_n': f_n[non_harm], 'a_n': a_n[non_harm], 'ph_n': ph_n[non_harm],
                  'f_h': f_n[harmonics], 'a_h': a_n[harmonics], 'ph_h': ph_n[harmonics]}
    kwargs = {'marginals': True, 'textsize': 14, 'kind': ['scatter', 'kde'],
              'marginal_kwargs': {'quantiles': [0.158, 0.5, 0.842]}, 'point_estimate': 'mean',
              'reference_values': ref_values, 'show': show}
    az.plot_pair(inf_data, var_names=['f_n', 'a_n', 'ph_n'],
                 coords={'f_n_dim_0': [0, 1, 2], 'a_n_dim_0': [0, 1, 2], 'ph_n_dim_0': [0, 1, 2]}, **kwargs)
    az.plot_pair(inf_data, var_names=['p_orb', 'f_n'], coords={'f_n_dim_0': np.arange(9)}, **kwargs)
    ax = az.plot_pair(inf_data, var_names=['p_orb', 'const', 'slope', 'f_n', 'a_n', 'ph_n', 'a_h', 'ph_h'],
                 coords={'const_dim_0': [0], 'slope_dim_0': [0], 'f_n_dim_0': [0], 'a_n_dim_0': [0], 'ph_n_dim_0': [0],
                         'a_h_dim_0': [0], 'ph_h_dim_0': [0]},
                 **kwargs)
    # save if wanted (only last plot - most interesting one)
    if save_file is not None:
        fig = ax.ravel()[0].figure
        fig.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    return


def plot_trace_harmonics(inf_data, p_orb, const, slope, f_n, a_n, ph_n):
    """Show the pymc3 sampling results in a trace plot"""
    # convert phases to interval [-pi, pi] from [0, 2pi]
    above_pi = (ph_n >= np.pi)
    ph_n[above_pi] = ph_n[above_pi] - 2 * np.pi
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    par_lines = [('p_orb', {}, p_orb), ('const', {}, const), ('slope', {}, slope),
                 ('f_n', {}, f_n[non_harm]), ('a_n', {}, a_n[non_harm]), ('ph_n', {}, ph_n[non_harm]),
                 ('f_h', {}, f_n[harmonics]), ('a_h', {}, a_n[harmonics]), ('ph_h', {}, ph_n[harmonics])]
    az.plot_trace(inf_data, combined=False, compact=True, rug=True, divergences='top', lines=par_lines)
    return


def plot_pair_eclipse(inf_data, ecosw, esinw, cosi, phi_0, r_rat, sb_rat, const, slope, f_n, a_n, ph_n,
                      save_file=None, show=True):
    """Show the pymc3 sampling results in several pair plots"""
    # convert phases to interval [-pi, pi] from [0, 2pi]
    above_pi = (ph_n >= np.pi)
    ph_n[above_pi] = ph_n[above_pi] - 2 * np.pi
    ref_values = {'t_zero': t_zero, 'const': const, 'slope': slope, 'f_n': f_n, 'a_n': a_n, 'ph_n': ph_n,
                  'ecosw': ecosw, 'esinw': esinw, 'cosi': cosi, 'phi_0': phi_0, 'r_rat': r_rat, 'sb_rat': sb_rat}
    kwargs = {'marginals': True, 'textsize': 14, 'kind': ['scatter', 'kde'],
              'marginal_kwargs': {'quantiles': [0.158, 0.5, 0.842]}, 'point_estimate': 'mean',
              'reference_values': ref_values, 'show': show}
    az.plot_pair(inf_data, var_names=['f_n', 'a_n', 'ph_n'],
                 coords={'f_n_dim_0': [0, 1], 'a_n_dim_0': [0, 1], 'ph_n_dim_0': [0, 1]}, **kwargs)
    ax1 = az.plot_pair(inf_data, var_names=['const', 'slope', 'f_n', 'a_n', 'ph_n'],
                 coords={'const_dim_0': [0], 'slope_dim_0': [0], 'f_n_dim_0': [0], 'a_n_dim_0': [0], 'ph_n_dim_0': [0]},
                 **kwargs)
    ax2 = az.plot_pair(inf_data, var_names=['t_zero', 'ecosw', 'esinw', 'cosi', 'phi_0', 'r_rat', 'sb_rat'], **kwargs)
    # save if wanted (only last plot - most interesting one)
    if save_file is not None:
        fig_save_file = save_file.replace('.png', '_a.png')
        fig = ax1.ravel()[0].figure
        fig.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
        fig_save_file = save_file.replace('.png', '_b.png')
        fig = ax2.ravel()[0].figure
        fig.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    return


def plot_trace_eclipse(inf_data, t_zero, ecosw, esinw, cosi, phi_0, r_rat, sb_rat, const, slope, f_n, a_n, ph_n):
    """Show the pymc3 sampling results in a trace plot"""
    # convert phases to interval [-pi, pi] from [0, 2pi]
    above_pi = (ph_n >= np.pi)
    ph_n[above_pi] = ph_n[above_pi] - 2 * np.pi
    par_lines = [('t_zero', {}, t_zero), ('const', {}, const), ('slope', {}, slope),
                 ('f_n', {}, f_n), ('a_n', {}, a_n), ('ph_n', {}, ph_n),
                 ('ecosw', {}, ecosw), ('esinw', {}, esinw), ('cosi', {}, cosi),
                 ('phi_0', {}, phi_0), ('r_rat', {}, r_rat), ('sb_rat', {}, sb_rat)]
    az.plot_trace(inf_data, combined=False, compact=True, rug=True, divergences='top', lines=par_lines)
    return
