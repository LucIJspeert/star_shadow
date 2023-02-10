"""STAR SHADOW
Satellite Time series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This Python module contains functions for visualisation;
specifically for visualising the analysis of stellar variability and eclipses.

Code written by: Luc IJspeert
"""

import os
import datetime
import fnmatch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import h5py
import arviz as az
import corner

from . import timeseries_functions as tsf
from . import timeseries_fitting as tsfit
from . import analysis_functions as af
from . import utility as ut

# mpl style sheet
script_dir = os.path.dirname(os.path.abspath(__file__))  # absolute dir the script is in
plt.style.use(script_dir.replace('star_shadow/star_shadow', 'star_shadow/data/mpl_stylesheet.dat'))


def plot_pd_single_output(times, signal, model, p_orb, p_err, f_n, a_n, i_sectors, annotate=True, save_file=None,
                          show=True):
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
    ax.plot(freqs, ampls, label='signal')
    if (len(f_n) > 0):
        ax.plot(freqs_r, ampls_r, label='residual')
    if (p_orb > 0):
        ax.errorbar([1 / p_orb, 1 / p_orb], [0, y_max], xerr=[0, p_err / p_orb**2],
                    linestyle='-', capsize=2, c='tab:grey', label=f'orbital frequency (p={p_orb:1.4f}d)')
        for i in range(2, np.max(harmonic_n) + 1):
            ax.plot([i / p_orb, i / p_orb], [0, y_max], linestyle='-', c='tab:grey', alpha=0.3)
    for i in range(len(f_n)):
        if i in harmonics:
            ax.errorbar([f_n[i], f_n[i]], [0, a_n[i]], xerr=[0, errors[2][i]], yerr=[0, errors[3][i]],
                        linestyle='-', capsize=2, c='tab:red')
        else:
            ax.errorbar([f_n[i], f_n[i]], [0, a_n[i]], xerr=[0, errors[2][i]], yerr=[0, errors[3][i]],
                        linestyle='-', capsize=2, c='tab:brown')
    if annotate:
        ax.annotate(f'{i + 1}', (f_n[i], a_n[i]))
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
    freqs_6, ampls_6 = tsf.astropy_scargle(times, signal - models[5] - np.all(models[5] == 0) * np.mean(signal))
    freqs_7, ampls_7 = tsf.astropy_scargle(times, signal - models[6] - np.all(models[6] == 0) * np.mean(signal))
    freqs_8, ampls_8 = tsf.astropy_scargle(times, signal - models[7] - np.all(models[7] == 0) * np.mean(signal))
    freqs_9, ampls_9 = tsf.astropy_scargle(times, signal - models[8] - np.all(models[8] == 0) * np.mean(signal))
    # get error values
    err_1 = tsf.formal_uncertainties(times, signal - models[0], a_n_i[0], i_sectors)
    err_2 = tsf.formal_uncertainties(times, signal - models[1], a_n_i[1], i_sectors)
    err_3 = tsf.formal_uncertainties(times, signal - models[2], a_n_i[2], i_sectors)
    err_4 = tsf.formal_uncertainties(times, signal - models[3], a_n_i[3], i_sectors)
    err_5 = tsf.formal_uncertainties(times, signal - models[4], a_n_i[4], i_sectors)
    err_6 = tsf.formal_uncertainties(times, signal - models[5], a_n_i[5], i_sectors)
    err_7 = tsf.formal_uncertainties(times, signal - models[6], a_n_i[6], i_sectors)
    err_8 = tsf.formal_uncertainties(times, signal - models[7], a_n_i[7], i_sectors)
    err_9 = tsf.formal_uncertainties(times, signal - models[8], a_n_i[8], i_sectors)
    # max plot value
    if (len(f_n_i[8]) > 0):
        y_max = max(np.max(ampls), np.max(a_n_i[8]))
    else:
        y_max = np.max(ampls)
    # plot
    fig, ax = plt.subplots()
    ax.plot(freqs, ampls, label='signal')
    if (len(f_n_i[0]) > 0):
        ax.plot(freqs_1, ampls_1, label='extraction residual')
    if (len(f_n_i[1]) > 0):
        ax.plot(freqs_2, ampls_2, label='NL-LS fit residual')
    if (len(f_n_i[2]) > 0):
        ax.plot(freqs_3, ampls_3, label='fixed harmonics residual')
    if (len(f_n_i[3]) > 0):
        ax.plot(freqs_4, ampls_4, label='additional harmonics residual')
    if (len(f_n_i[4]) > 0):
        ax.plot(freqs_5, ampls_5, label='first NL-LS fit residual with harmonics')
    if (len(f_n_i[5]) > 0):
        ax.plot(freqs_6, ampls_6, label='additional non-harmonics residual')
    if (len(f_n_i[6]) > 0):
        ax.plot(freqs_7, ampls_7, label='second NL-LS fit residual with harmonics')
    if (len(f_n_i[7]) > 0):
        ax.plot(freqs_8, ampls_8, label='Reduced frequencies')
    if (len(f_n_i[8]) > 0):
        ax.plot(freqs_9, ampls_9, label='third NL-LS fit residual with harmonics')
    # period
    if (p_orb_i[8] > 0):
        ax.errorbar([1 / p_orb_i[8], 1 / p_orb_i[8]], [0, y_max], xerr=[0, p_err_i[8] / p_orb_i[8]**2],
                    linestyle='--', capsize=2, c='k', label=f'orbital frequency (p={p_orb_i[7]:1.4f}d)')
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
    for i in range(len(f_n_i[5])):
        ax.errorbar([f_n_i[5][i], f_n_i[5][i]], [0, a_n_i[5][i]], xerr=[0, err_6[2][i]], yerr=[0, err_6[3][i]],
                    linestyle=':', capsize=2, c='tab:pink')
    for i in range(len(f_n_i[6])):
        ax.errorbar([f_n_i[6][i], f_n_i[6][i]], [0, a_n_i[6][i]], xerr=[0, err_7[2][i]], yerr=[0, err_7[3][i]],
                    linestyle=':', capsize=2, c='tab:grey')
    for i in range(len(f_n_i[7])):
        ax.errorbar([f_n_i[7][i], f_n_i[7][i]], [0, a_n_i[7][i]], xerr=[0, err_8[2][i]], yerr=[0, err_8[3][i]],
                    linestyle=':', capsize=2, c='tab:olive')
    for i in range(len(f_n_i[8])):
        ax.errorbar([f_n_i[8][i], f_n_i[8][i]], [0, a_n_i[8][i]], xerr=[0, err_9[2][i]], yerr=[0, err_9[3][i]],
                    linestyle=':', capsize=2, c='tab:cyan')
        ax.annotate(f'{i + 1}', (f_n_i[8][i], a_n_i[8][i]))
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
    # make models
    model = tsf.linear_curve(times, const, slope, i_sectors)
    model += tsf.sum_sines(times, f_n, a_n, ph_n)
    # plot the full model light curve
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].scatter(times, signal, marker='.', label='signal')
    ax[0].plot(times, model, c='tab:orange', label='full model (linear + sinusoidal)')
    ax[1].scatter(times, signal - model, marker='.')
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
    # make models
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    model = model_line + tsf.sum_sines(times, f_n, a_n, ph_n)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    model_h = tsf.sum_sines(times, f_n[harmonics], a_n[harmonics], ph_n[harmonics])
    model_nh = tsf.sum_sines(times, np.delete(f_n, harmonics), np.delete(a_n, harmonics),
                             np.delete(ph_n, harmonics))
    errors = tsf.formal_uncertainties(times, signal - model, a_n, i_sectors)
    # plot the harmonic model and non-harmonic model
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].scatter(times, signal - model_nh, marker='.', c='tab:blue', label='signal - non-harmonics')
    ax[0].plot(times, model_line + model_h, c='tab:orange', label='linear + harmonic model')
    ax[1].scatter(times, signal - model_h, marker='.', c='tab:blue', label='signal - harmonics')
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


def plot_lc_eclipse_timestamps(times, signal, p_orb, t_zero, timings, depths, timings_err, depths_err, const, slope,
                               f_n, a_n, ph_n, f_h, a_h, ph_h, i_sectors, save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the first and
    last contact points as well as minima indicated.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err = timings_err
    dur_b_1_err = np.sqrt(t_1_1_err**2 + t_1_2_err**2)
    dur_b_2_err = np.sqrt(t_2_1_err**2 + t_2_2_err**2)
    # plotting bounds
    t_ext_1 = t_1_1 - 6 * t_1_1_err
    t_ext_2 = t_1_2 + 6 * t_1_2_err
    # make the model times array, one full period plus the primary eclipse halves
    t_extended, ext_left, ext_right = tsf.fold_time_series(times, p_orb, t_zero, t_ext_1=t_ext_1, t_ext_2=t_ext_2)
    s_extended = np.concatenate((signal[ext_left], signal, signal[ext_right]))
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    t_model = np.arange(t_ext_1, p_orb + t_ext_2, 0.001)
    model_h = 1 + tsf.sum_sines(t_model + t_zero, f_h, a_h, ph_h, t_shift=False)
    model_nh = tsf.sum_sines(times, f_n, a_n, ph_n) - tsf.sum_sines(times, f_h, a_h, ph_h)
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line + 1
    ecl_signal = np.concatenate((ecl_signal[ext_left], ecl_signal, ecl_signal[ext_right]))
    # determine a lc offset to match the model at the edges
    h_1, h_2 = af.height_at_contact(f_h, a_h, ph_h, t_zero, t_1_1, t_1_2, t_2_1, t_2_2)
    offset = 1 - (h_1 + h_2) / 2
    # some plotting parameters
    h_top_1 = h_1 + offset
    h_top_2 = h_2 + offset
    h_bot_1 = h_1 - depths[0] + offset
    h_bot_2 = h_2 - depths[1] + offset
    s_minmax = np.array([np.min(signal), np.max(signal)])
    # plot
    fig, ax = plt.subplots()
    ax.scatter(t_extended, s_extended, marker='.', label='original folded signal')
    ax.scatter(t_extended, ecl_signal + offset, marker='.', c='tab:orange',
               label='(non-harmonics + linear) model residual')
    ax.plot(t_model, model_h + offset, c='tab:red', label='harmonics')
    ax.plot([t_1, t_1], s_minmax, '--', c='tab:pink')
    ax.plot([t_2, t_2], s_minmax, '--', c='tab:pink')
    ax.plot([t_1_1, t_1_1], s_minmax, '--', c='tab:purple', label=r'eclipse edges/minima/depths')
    ax.plot([t_1_2, t_1_2], s_minmax, '--', c='tab:purple')
    ax.plot([t_2_1, t_2_1], s_minmax, '--', c='tab:purple')
    ax.plot([t_2_2, t_2_2], s_minmax, '--', c='tab:purple')
    ax.plot([t_1_1, t_1_2], [h_bot_1, h_bot_1], '--', c='tab:purple')
    ax.plot([t_2_1, t_2_2], [h_bot_2, h_bot_2], '--', c='tab:purple')
    ax.plot([t_1_1, t_1_2], [h_top_1, h_top_1], '--', c='tab:purple')
    ax.plot([t_2_1, t_2_2], [h_top_2, h_top_2], '--', c='tab:purple')
    # 1 sigma errors
    ax.fill_between([t_1 - t_1_err, t_1 + t_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:red', alpha=0.3)
    ax.fill_between([t_2 - t_2_err, t_2 + t_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:red', alpha=0.3)
    ax.fill_between([t_1_1 - t_1_1_err, t_1_1 + t_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.3, label=r'1 and 3 $\sigma$ error')
    ax.fill_between([t_1_2 - t_1_2_err, t_1_2 + t_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.3)
    ax.fill_between([t_2_1 - t_2_1_err, t_2_1 + t_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.3)
    ax.fill_between([t_2_2 - t_2_2_err, t_2_2 + t_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.3)
    ax.fill_between([t_1_1, t_1_2], y1=[h_bot_1 + depths_err[0], h_bot_1 + depths_err[0]],
                    y2=[h_bot_1 - depths_err[0], h_bot_1 - depths_err[0]], color='tab:purple', alpha=0.3)
    ax.fill_between([t_2_1, t_2_2], y1=[h_bot_2 + depths_err[1], h_bot_2 + depths_err[1]],
                    y2=[h_bot_2 - depths_err[1], h_bot_2 - depths_err[1]], color='tab:purple', alpha=0.3)
    # 3 sigma errors
    ax.fill_between([t_1 - 3 * t_1_err, t_1 + 3 * t_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:red', alpha=0.2)
    ax.fill_between([t_2 - 3 * t_2_err, t_2 + 3 * t_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:red', alpha=0.2)
    ax.fill_between([t_1_1 - 3 * t_1_1_err, t_1_1 + 3 * t_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.2)
    ax.fill_between([t_1_2 - 3 * t_1_2_err, t_1_2 + 3 * t_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.2)
    ax.fill_between([t_2_1 - 3 * t_2_1_err, t_2_1 + 3 * t_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.2)
    ax.fill_between([t_2_2 - 3 * t_2_2_err, t_2_2 + 3 * t_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.2)
    ax.fill_between([t_1_1, t_1_2], y1=[h_bot_1 + 3 * depths_err[0], h_bot_1 + 3 * depths_err[0]],
                    y2=[h_bot_1 - 3 * depths_err[0], h_bot_1 - 3 * depths_err[0]],
                    color='tab:purple', alpha=0.2)
    ax.fill_between([t_2_1, t_2_2], y1=[h_bot_2 + 3 * depths_err[1], h_bot_2 + 3 * depths_err[1]],
                    y2=[h_bot_2 - 3 * depths_err[1], h_bot_2 - 3 * depths_err[1]],
                    color='tab:purple', alpha=0.2)
    # flat bottom
    if ((t_b_1_2 - t_b_1_1) / dur_b_1_err > 1):
        ax.plot([t_b_1_1, t_b_1_1], s_minmax, '--', c='tab:brown')
        ax.plot([t_b_1_2, t_b_1_2], s_minmax, '--', c='tab:brown')
        # 1 sigma errors
        ax.fill_between([t_b_1_1 - t_1_1_err, t_b_1_1 + t_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                        color='tab:brown', alpha=0.3)
        ax.fill_between([t_b_1_2 - t_1_2_err, t_b_1_2 + t_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                        color='tab:brown', alpha=0.3)
        # 3 sigma errors
        ax.fill_between([t_b_1_1 - 3 * t_1_1_err, t_b_1_1 + 3 * t_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                        color='tab:brown', alpha=0.2)
        ax.fill_between([t_b_1_2 - 3 * t_1_2_err, t_b_1_2 + 3 * t_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                        color='tab:brown', alpha=0.2)
    if ((t_b_2_2 - t_b_2_1) / dur_b_2_err > 1):
        ax.plot([t_b_2_1, t_b_2_1], s_minmax, '--', c='tab:brown')
        ax.plot([t_b_2_2, t_b_2_2], s_minmax, '--', c='tab:brown')
        # 1 sigma errors
        ax.fill_between([t_b_2_1 - t_2_1_err, t_b_2_1 + t_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                        color='tab:brown', alpha=0.3)
        ax.fill_between([t_b_2_2 - t_2_2_err, t_b_2_2 + t_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                        color='tab:brown', alpha=0.3)
        # 3 sigma errors
        ax.fill_between([t_b_2_1 - 3 * t_2_1_err, t_b_2_1 + 3 * t_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                        color='tab:brown', alpha=0.2)
        ax.fill_between([t_b_2_2 - 3 * t_2_2_err, t_b_2_2 + 3 * t_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                        color='tab:brown', alpha=0.2)
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


def plot_lc_empirical_model(times, signal, p_orb, t_zero, timings, depths, const, slope, f_n, a_n, ph_n, t_zero_em,
                            timings_em, depths_em, timings_err, depths_err, i_sectors, save_file=None, show=True):
    """Shows the initial and final simple empirical cubic function eclipse model
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings
    depth_1, depth_2 = depths
    t_1_em, t_2_em, t_1_1_em, t_1_2_em, t_2_1_em, t_2_2_em, t_b_1_1_em, t_b_1_2_em, t_b_2_1_em, t_b_2_2_em = timings_em
    t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err = timings_err
    depth_em_1, depth_em_2 = depths_em
    depth_1_err, depth_2_err = depths_err
    dur_b_1_err = np.sqrt(t_1_1_err**2 + t_1_2_err**2)
    dur_b_2_err = np.sqrt(t_2_1_err**2 + t_2_2_err**2)
    # plotting bounds
    t_ext_1 = min(t_1_1, t_1_1_em)
    t_ext_2 = max(t_1_2, t_1_2_em)
    # make the model times array, one full period plus the primary eclipse halves
    t_extended, ext_left, ext_right = tsf.fold_time_series(times, p_orb, t_zero_em, t_ext_1=t_ext_1, t_ext_2=t_ext_2)
    s_extended = np.concatenate((signal[ext_left], signal, signal[ext_right]))
    sorter = np.argsort(t_extended)
    # sinusoid and linear models
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    # cubic model - get the parameters for the cubics from the fit parameters
    t_model = np.arange(t_ext_1, p_orb + t_ext_2, 0.001)
    mean_t_m = np.mean(t_model)
    mid_1 = (t_1_1 + t_1_2) / 2
    mid_2 = (t_2_1 + t_2_2) / 2
    model_ecl_init = tsfit.eclipse_empirical_lc(t_model, p_orb, -mean_t_m, mid_1, mid_2, t_1_1, t_2_1,
                                                t_b_1_1, t_b_2_1, depth_1, depth_2)
    model_ecl = tsfit.eclipse_empirical_lc(t_model, p_orb, -mean_t_m, t_1_em, t_2_em, t_1_1_em, t_2_1_em,
                                           t_b_1_1_em, t_b_2_1_em, depth_em_1, depth_em_2)
    # add residual harmonic sinusoids to model
    model_ecl_2 = tsfit.eclipse_empirical_lc(times, p_orb, t_zero_em, t_1_em, t_2_em, t_1_1_em, t_2_1_em,
                                             t_b_1_1_em, t_b_2_1_em, depth_em_1, depth_em_2)
    model_sin_lin = model_sinusoid + model_linear
    model_sin_lin = np.concatenate((model_sin_lin[ext_left], model_sin_lin, model_sin_lin[ext_right]))
    model_ecl_sin_lin = model_ecl_2 + model_sinusoid + model_linear
    model_ecl_sin_lin = np.concatenate((model_ecl_sin_lin[ext_left], model_ecl_sin_lin, model_ecl_sin_lin[ext_right]))
    # residuals
    resid_ecl = signal - model_ecl_2 - 1
    resid_ecl = np.concatenate((resid_ecl[ext_left], resid_ecl, resid_ecl[ext_right]))
    resid_full = signal - model_ecl_2 - 1 - model_sinusoid - model_linear
    resid_full = np.concatenate((resid_full[ext_left], resid_full, resid_full[ext_right]))
    # some plotting parameters
    s_minmax = np.array([np.min(signal), np.max(signal)])
    s_minmax_r = np.array([np.min(resid_ecl), np.max(resid_ecl)])
    # plot
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].scatter(t_extended, s_extended, marker='.', label='original signal')
    ax[0].plot(t_extended[sorter], model_ecl_sin_lin[sorter] + 1, c='tab:grey', alpha=0.8,
               label='final (linear + sinusoid + simple empirical eclipse) model')
    ax[0].plot(t_model, model_ecl_init + 1, c='tab:orange', label='initial simple empirical eclipse model')
    ax[0].plot(t_model, model_ecl + 1, c='tab:red', label='final simple empirical eclipse model')
    ax[0].plot([t_1_1, t_1_1], s_minmax, ':', c='tab:grey', label=r'old eclipse edges (low harmonics)')
    ax[0].plot([t_1_2, t_1_2], s_minmax, ':', c='tab:grey')
    ax[0].plot([t_2_1, t_2_1], s_minmax, ':', c='tab:grey')
    ax[0].plot([t_2_2, t_2_2], s_minmax, ':', c='tab:grey')
    ax[0].plot([t_1_1_em, t_1_1_em], s_minmax, '--', c='tab:purple', label=r'new eclipse edges (cubics)')
    ax[0].plot([t_1_2_em, t_1_2_em], s_minmax, '--', c='tab:purple')
    ax[0].plot([t_2_1_em, t_2_1_em], s_minmax, '--', c='tab:purple')
    ax[0].plot([t_2_2_em, t_2_2_em], s_minmax, '--', c='tab:purple')
    ax[0].plot([t_1_1_em, t_1_2_em], [1, 1], '--', c='tab:purple')
    ax[0].plot([t_1_1_em, t_1_2_em], [1 - depth_em_1, 1 - depth_em_1], '--', c='tab:purple')
    ax[0].plot([t_2_1_em, t_2_2_em], [1, 1], '--', c='tab:purple')
    ax[0].plot([t_2_1_em, t_2_2_em], [1 - depth_em_2, 1 - depth_em_2], '--', c='tab:purple')
    # 1 sigma errors
    ax[0].fill_between([t_1_1_em - t_1_1_err, t_1_1_em + t_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.3, label=r'1 and 3 $\sigma$ error')
    ax[0].fill_between([t_1_2_em - t_1_2_err, t_1_2_em + t_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.3)
    ax[0].fill_between([t_2_1_em - t_2_1_err, t_2_1_em + t_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.3)
    ax[0].fill_between([t_2_2_em - t_2_2_err, t_2_2_em + t_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.3)
    ax[0].fill_between([t_1_1_em, t_1_2_em], y1=[1 - depth_em_1 + depth_1_err, 1 - depth_em_1 + depth_1_err],
                       y2=[1 - depth_em_1 - depth_1_err, 1 - depth_em_1 - depth_1_err], color='tab:purple', alpha=0.3)
    ax[0].fill_between([t_2_1_em, t_2_2_em], y1=[1 - depth_em_2 + depth_2_err, 1 - depth_em_2 + depth_2_err],
                       y2=[1 - depth_em_2 - depth_2_err, 1 - depth_em_2 - depth_2_err], color='tab:purple', alpha=0.3)
    # 3 sigma errors
    ax[0].fill_between([t_1_1_em - 3 * t_1_1_err, t_1_1_em + 3 * t_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.2)
    ax[0].fill_between([t_1_2_em - 3 * t_1_2_err, t_1_2_em + 3 * t_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.2)
    ax[0].fill_between([t_2_1_em - 3 * t_2_1_err, t_2_1_em + 3 * t_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.2)
    ax[0].fill_between([t_2_2_em - 3 * t_2_2_err, t_2_2_em + 3 * t_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                       color='tab:purple', alpha=0.2)
    ax[0].fill_between([t_1_1_em, t_1_2_em], y1=[1 - depth_em_1 + 3 * depth_1_err, 1 - depth_em_1 + 3 * depth_1_err],
                       y2=[1 - depth_em_1 - 3 * depth_1_err, 1 - depth_em_1 - 3 * depth_1_err],
                       color='tab:purple', alpha=0.2)
    ax[0].fill_between([t_2_1_em, t_2_2_em], y1=[1 - depth_em_2 + 3 * depth_2_err, 1 - depth_em_2 + 3 * depth_2_err],
                       y2=[1 - depth_em_2 - 3 * depth_2_err, 1 - depth_em_2 - 3 * depth_2_err],
                       color='tab:purple', alpha=0.2)
    # flat bottom
    if ((t_b_1_2_em - t_b_1_1_em) / dur_b_1_err > 1):
        ax[0].plot([t_b_1_1_em, t_b_1_1_em], s_minmax, '--', c='tab:brown')
        ax[0].plot([t_b_1_2_em, t_b_1_2_em], s_minmax, '--', c='tab:brown')
        # 1 sigma errors
        ax[0].fill_between([t_b_1_1_em - t_1_1_err, t_b_1_1_em + t_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.3)
        ax[0].fill_between([t_b_1_2_em - t_1_2_err, t_b_1_2_em + t_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.3)
        # 3 sigma errors
        ax[0].fill_between([t_b_1_1_em - 3 * t_1_1_err, t_b_1_1_em + 3 * t_1_1_err], y1=s_minmax[[0, 0]],
                           y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.2)
        ax[0].fill_between([t_b_1_2_em - 3 * t_1_2_err, t_b_1_2_em + 3 * t_1_2_err], y1=s_minmax[[0, 0]],
                           y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.2)
    if ((t_b_2_2_em - t_b_2_1_em) / dur_b_2_err > 1):
        ax[0].plot([t_b_2_1_em, t_b_2_1_em], s_minmax, '--', c='tab:brown')
        ax[0].plot([t_b_2_2_em, t_b_2_2_em], s_minmax, '--', c='tab:brown')
        # 1 sigma errors
        ax[0].fill_between([t_b_2_1_em - t_2_1_err, t_b_2_1_em + t_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.3)
        ax[0].fill_between([t_b_2_2_em - t_2_2_err, t_b_2_2_em + t_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.3)
        # 3 sigma errors
        ax[0].fill_between([t_b_2_1_em - 3 * t_2_1_err, t_b_2_1_em + 3 * t_2_1_err], y1=s_minmax[[0, 0]],
                           y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.2)
        ax[0].fill_between([t_b_2_2_em - 3 * t_2_2_err, t_b_2_2_em + 3 * t_2_2_err], y1=s_minmax[[0, 0]],
                           y2=s_minmax[[1, 1]],
                           color='tab:brown', alpha=0.2)
    if ((t_b_1_2_em - t_b_1_1_em) / dur_b_1_err > 1) | ((t_b_2_2_em - t_b_2_1_em) / dur_b_2_err > 1):
        ax[0].plot([], [], '--', c='tab:brown', label='flat bottom')  # ghost label
    ax[0].set_ylabel('normalised flux')
    ax[0].legend()
    # residuals subplot
    ax[1].scatter(t_extended, resid_ecl, marker='.', label='(linear + sinusoid) model residual')
    ax[1].scatter(t_extended, resid_full, marker='.', label='(linear + sinusoid + eclipse) model residual')
    ax[1].plot(t_extended[sorter], model_sin_lin[sorter], c='tab:grey', alpha=0.8, label='(linear + sinusoid) model')
    ax[1].plot([t_1_1, t_1_1], s_minmax_r, ':', c='tab:grey', label=r'old eclipse edges (low harmonics)')
    ax[1].plot([t_1_2, t_1_2], s_minmax_r, ':', c='tab:grey')
    ax[1].plot([t_2_1, t_2_1], s_minmax_r, ':', c='tab:grey')
    ax[1].plot([t_2_2, t_2_2], s_minmax_r, ':', c='tab:grey')
    ax[1].plot([t_1_1_em, t_1_1_em], s_minmax_r, '--', c='tab:purple', label=r'new eclipse edges (cubics)')
    ax[1].plot([t_1_2_em, t_1_2_em], s_minmax_r, '--', c='tab:purple')
    ax[1].plot([t_2_1_em, t_2_1_em], s_minmax_r, '--', c='tab:purple')
    ax[1].plot([t_2_2_em, t_2_2_em], s_minmax_r, '--', c='tab:purple')
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


def plot_lc_eclipse_parameters_simple(times, signal, p_orb, t_zero, timings, const, slope, f_n, a_n, ph_n, ecl_params,
                                      i_sectors, save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the determination
    of orbital parameters using the eclipse timings and depths
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings
    # make the model times array, one full period plus the primary eclipse halves
    t_extended, ext_left, ext_right = tsf.fold_time_series(times, p_orb, t_zero, t_ext_1=t_1_1, t_ext_2=t_1_2)
    sorter = np.argsort(t_extended)
    mask_1 = (t_extended > t_1_1) & (t_extended < t_1_2)
    mask_2 = (t_extended > t_2_1) & (t_extended < t_2_2)
    mean_t_e = np.mean(t_extended)
    mean_t_e1 = np.mean(t_extended[mask_1])
    mean_t_e2 = np.mean(t_extended[mask_1])
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line + 1
    ecl_signal = np.concatenate((ecl_signal[ext_left], ecl_signal, ecl_signal[ext_right]))
    # determine a lc offset to match the model at the edges
    h_1, h_2 = af.height_at_contact(f_n[harmonics], a_n[harmonics], ph_n[harmonics], t_zero, t_1_1, t_1_2, t_2_1, t_2_2)
    offset = 1 - (h_1 + h_2) / 2
    s_minmax = [np.min(signal), np.max(signal)]
    # unpack and define parameters
    e, w, i, phi_0, r_sum_sma, r_ratio, sb_ratio = ecl_params
    # make the simple model
    ecl_model = tsfit.eclipse_physical_lc(t_extended, p_orb, -mean_t_e, e, w, i, r_sum_sma, r_ratio, sb_ratio)
    model_ecl_1 = tsfit.eclipse_physical_lc(t_extended[mask_1], p_orb, -mean_t_e1, e, w, i, r_sum_sma, r_ratio, sb_ratio)
    model_ecl_2 = tsfit.eclipse_physical_lc(t_extended[mask_2], p_orb, -mean_t_e2, e, w, i, r_sum_sma, r_ratio, sb_ratio)
    # plot
    fig, ax = plt.subplots()
    ax.scatter(t_extended, ecl_signal + offset, marker='.', label='eclipse signal')
    ax.plot(t_extended[sorter][mask_1], model_ecl_1, c='tab:orange', label='spheres of uniform brightness')
    ax.plot(t_extended[sorter][mask_2], model_ecl_2, c='tab:orange')
    ax.plot(t_extended[sorter], ecl_model, c='tab:purple', alpha=0.3)
    ax.plot([t_1_1, t_1_1], s_minmax, '--', c='grey', label='eclipse edges')
    ax.plot([t_1_2, t_1_2], s_minmax, '--', c='grey')
    ax.plot([t_2_1, t_2_1], s_minmax, '--', c='grey')
    ax.plot([t_2_2, t_2_2], s_minmax, '--', c='grey')
    ax.set_xlabel(r'$(time - t_0) mod(P_{orb})$ (d)')
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


def plot_dists_eclipse_parameters(e, w, i, r_sum_sma, r_ratio, sb_ratio, e_vals, w_vals, i_vals, rsumsma_vals,
                                  rratio_vals, sbratio_vals):
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
    # r_sum_sma
    rsumsma_interval = az.hdi(rsumsma_vals, hdi_prob=0.683)
    rsumsma_bounds = az.hdi(rsumsma_vals, hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist(rsumsma_vals, bins=50, label='vary fit input')
    ax.plot([r_sum_sma, r_sum_sma], [0, np.max(hist[0])], c='tab:green', label='best fit value')
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
    # r_ratio
    rratio_interval = az.hdi(rratio_vals, hdi_prob=0.683)
    rratio_bounds = az.hdi(rratio_vals, hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist((rratio_vals), bins=50, label='vary fit input')
    ax.plot(([r_ratio, r_ratio]), [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([rratio_interval[0], rratio_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([rratio_interval[1], rratio_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([rratio_bounds[0], rratio_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([rratio_bounds[1], rratio_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('r_ratio')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # log(r_ratio)
    log_rratio_interval = az.hdi(np.log10(rratio_vals), hdi_prob=0.683)
    log_rratio_bounds = az.hdi(np.log10(rratio_vals), hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist(np.log10(rratio_vals), bins=50, label='vary fit input')
    ax.plot(np.log10([r_ratio, r_ratio]), [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([log_rratio_interval[0], log_rratio_interval[0]], [0, np.max(hist[0])],
            c='tab:orange', label='hdi_prob=0.683')
    ax.plot([log_rratio_interval[1], log_rratio_interval[1]], [0, np.max(hist[0])],
            c='tab:orange')
    ax.plot([log_rratio_bounds[0], log_rratio_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([log_rratio_bounds[1], log_rratio_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('log(r_ratio)')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # sb_ratio
    sbratio_interval = az.hdi(sbratio_vals, hdi_prob=0.683)
    sbratio_bounds = az.hdi(sbratio_vals, hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist((sbratio_vals), bins=50, label='vary fit input')
    ax.plot(([sb_ratio, sb_ratio]), [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([sbratio_interval[0], sbratio_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([sbratio_interval[1], sbratio_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([sbratio_bounds[0], sbratio_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([sbratio_bounds[1], sbratio_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('sb_ratio')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # log(sb_ratio)
    log_sbratio_interval = az.hdi(np.log10(sbratio_vals), hdi_prob=0.683)
    log_sbratio_bounds = az.hdi(np.log10(sbratio_vals), hdi_prob=0.997)
    fig, ax = plt.subplots()
    hist = ax.hist(np.log10(sbratio_vals), bins=50, label='vary fit input')
    ax.plot(np.log10([sb_ratio, sb_ratio]), [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([log_sbratio_interval[0], log_sbratio_interval[0]], [0, np.max(hist[0])],
            c='tab:orange', label='hdi_prob=0.683')
    ax.plot([log_sbratio_interval[1], log_sbratio_interval[1]], [0, np.max(hist[0])],
            c='tab:orange')
    ax.plot([log_sbratio_bounds[0], log_sbratio_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([log_sbratio_bounds[1], log_sbratio_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('log(sb_ratio)')
    ax.set_ylabel('N')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return


def plot_corner_eclipse_parameters(p_orb, timings, depths, p_vals, t_1_vals, t_2_vals, t_1_1_vals, t_1_2_vals,
                                   t_2_1_vals, t_2_2_vals, t_b_1_1_vals, t_b_1_2_vals, t_b_2_1_vals, t_b_2_2_vals,
                                   d_1_vals, d_2_vals, e, w, i, r_sum_sma, r_ratio, sb_ratio, e_vals, w_vals, i_vals,
                                   rsumsma_vals, rratio_vals, sbratio_vals, save_file=None, show=True):
    """Shows the corner plots resulting from the input distributions
    
    Note: produces several plots
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings
    d_1, d_2 = depths
    # for if the w-distribution crosses over at 2 pi
    if (abs(w / np.pi * 180 - 180) > 80) & (abs(w / np.pi * 180 - 180) < 100):
        w_vals = np.copy(w_vals)
    else:
        w_vals = np.copy(w_vals)
        mask = (np.sign((w / np.pi * 180 - 180) * (w_vals / np.pi * 180 - 180)) < 0)
        w_vals[mask] = w_vals[mask] + np.sign(w / np.pi * 180 - 180) * 2 * np.pi
    # input
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
    # corner plot
    value_names = np.array(['e', 'w (deg)', 'i (deg)', r'$\frac{r_1+r_2}{a}$', r'$\frac{r_2}{r_1}$',
                            r'$\frac{sb_2}{sb_1}$'])
    values = np.array([e, w / np.pi * 180, i / np.pi * 180, r_sum_sma, r_ratio, sb_ratio])
    dist_data = np.column_stack((e_vals, w_vals / np.pi * 180, i_vals / np.pi * 180, rsumsma_vals, rratio_vals,
                                 sbratio_vals))
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
            fig_save_file = save_file.replace('.png', '_out.png')
        else:
            fig_save_file = save_file + '_out.png'
        fig.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_light_curve_fit(times, signal, p_orb, t_zero, timings, const, slope, f_n, a_n, ph_n, par_init, par_opt,
                            i_sectors, save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the determination
    of orbital parameters using both the eclipse timings, simple model and the ellc light curve models.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings
    # make the model times array, one full period plus the primary eclipse halves
    t_extended, ext_left, ext_right = tsf.fold_time_series(times, p_orb, t_zero, t_ext_1=t_1_1, t_ext_2=t_1_2)
    sorter = np.argsort(t_extended)
    mean_t_e = np.mean(t_extended)
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line + 1
    ecl_signal = np.concatenate((ecl_signal[ext_left], ecl_signal, ecl_signal[ext_right]))
    s_minmax = [np.min(signal), np.max(signal)]
    # determine a lc offset to match the model at the edges
    h_1, h_2 = af.height_at_contact(f_n[harmonics], a_n[harmonics], ph_n[harmonics], t_zero, t_1_1, t_1_2, t_2_1, t_2_2)
    offset = 1 - (h_1 + h_2) / 2
    # unpack and define parameters
    e, w, i, r_sum_sma, r_ratio, sb_ratio = par_init
    opt_e, opt_w, opt_i, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio = par_opt
    opt_f_c, opt_f_s = opt_e**0.5 * np.cos(opt_w), opt_e**0.5 * np.sin(opt_w)
    # make the ellc models
    model_simple_init = tsfit.eclipse_physical_lc(t_extended, p_orb, -mean_t_e, e, w, i, r_sum_sma, r_ratio, sb_ratio)
    model_opt1 = tsfit.eclipse_physical_lc(t_extended, p_orb, -mean_t_e, opt_e, opt_w, opt_i, opt_r_sum_sma,
                                           opt_r_ratio, opt_sb_ratio)
    # plot the physical eclipse model
    fig, ax = plt.subplots()
    ax.scatter(t_extended, ecl_signal + offset, marker='.', label='eclipse signal')
    ax.plot(t_extended[sorter], model_simple_init[sorter], c='tab:orange', label='initial simple eclipse model')
    ax.plot(t_extended[sorter], model_opt1[sorter], c='tab:red', label='final simple eclipse model')
    ax.plot([t_1_1, t_1_1], s_minmax, '--', c='grey', label='eclipse edges')
    ax.plot([t_1_2, t_1_2], s_minmax, '--', c='grey')
    ax.plot([t_2_1, t_2_1], s_minmax, '--', c='grey')
    ax.plot([t_2_2, t_2_2], s_minmax, '--', c='grey')
    ax.set_xlabel(r'$(time - t_0) mod(P_{orb})$ (d)')
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


def plot_lc_ellc_errors(times, signal, p_orb, t_zero, timings, const, slope, f_n, a_n, ph_n, i_sectors,
                        par_ellc, par_i, par_bounds, save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the determination
    of orbital parameters using both the eclipse timings and the ellc light curve
    models over three consecutive fits.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings
    # make the model times array, one full period plus the primary eclipse halves
    t_extended, ext_left, ext_right = tsf.fold_time_series(times, p_orb, t_zero, t_ext_1=t_1_1, t_ext_2=t_1_2)
    sorter = np.argsort(t_extended)
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line + 1
    ecl_signal = np.concatenate((ecl_signal[ext_left], ecl_signal, ecl_signal[ext_right]))
    s_minmax = [np.min(signal), np.max(signal)]
    # determine a lc offset to match the model at the edges
    h_1, h_2 = af.height_at_contact(f_n[harmonics], a_n[harmonics], ph_n[harmonics], t_zero, t_1_1, t_1_2, t_2_1, t_2_2)
    offset = 1 - (h_1 + h_2) / 2
    # unpack and define parameters
    opt_f_c, opt_f_s, opt_i, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio = par_ellc
    # make the ellc models
    model = tsfit.wrap_ellc_lc(t_extended, p_orb, 0, opt_f_c, opt_f_s, opt_i, opt_r_sum_sma, opt_r_ratio,
                               opt_sb_ratio, 0)
    par_p = np.copy(par_ellc)
    par_n = np.copy(par_ellc)
    par_p[par_i] = par_bounds[1]
    par_n[par_i] = par_bounds[0]
    model_p = tsfit.wrap_ellc_lc(t_extended, p_orb, 0, par_p[0], par_p[1], par_p[2], par_p[3], par_p[4], par_p[5], 0)
    model_m = tsfit.wrap_ellc_lc(t_extended, p_orb, 0, par_n[0], par_n[1], par_n[2], par_n[3], par_n[4], par_n[5], 0)
    par_names = ['f_c', 'f_s', 'i', 'r_sum', 'r_ratio', 'sb_ratio']
    # plot
    fig, ax = plt.subplots()
    ax.scatter(t_extended, ecl_signal + offset, marker='.', label='eclipse signal')
    ax.plot(t_extended[sorter], model[sorter], c='tab:orange', label='best parameters')
    ax.fill_between(t_extended[sorter], y1=model[sorter], y2=model_p[sorter], color='tab:orange', alpha=0.3,
                    label=f'upper bound {par_names[par_i]}')
    ax.fill_between(t_extended[sorter], y1=model[sorter], y2=model_m[sorter], color='tab:purple', alpha=0.3,
                    label=f'lower bound {par_names[par_i]}')
    ax.plot([t_1_1, t_1_1], s_minmax, '--', c='grey', label=r'eclipse edges')
    ax.plot([t_1_2, t_1_2], s_minmax, '--', c='grey')
    ax.plot([t_2_1, t_2_1], s_minmax, '--', c='grey')
    ax.plot([t_2_2, t_2_2], s_minmax, '--', c='grey')
    ax.set_xlabel(r'$(time - t_0) mod(P_{orb})$ (d)')
    ax.set_ylabel('normalised flux')
    ax.set_title(f'{par_names[par_i]} = {par_ellc[par_i]:1.4f}, bounds: ({par_bounds[0]:1.4f}, {par_bounds[1]:1.4f})')
    plt.legend()
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_corner_lc_fit_pars(par_init, par_opt, distributions, save_file=None, show=True):
    """Corner plot of the distributions and the given 'truths' indicated
    using the parametrisation of ellc
    """
    r2d = 180 / np.pi  # radians to degrees
    e_vals, w_vals, i_vals, rsumsma_vals, rratio_vals, sbratio_vals = distributions
    # transform some params - initial
    e, w, i_rad, r_sum_sma, r_ratio, sb_ratio = par_init
    ecosw, esinw = e * np.cos(w), e * np.sin(w)
    i = i_rad * r2d
    par_init_a = np.array([ecosw, esinw, i, r_sum_sma, r_ratio, sb_ratio])
    par_init_b = np.array([e, w * r2d, i, r_sum_sma, r_ratio, sb_ratio])
    # for if the w-distribution crosses over at 2 pi
    if (abs(w / np.pi * 180 - 180) > 80) & (abs(w / np.pi * 180 - 180) < 100):
        w_vals = np.copy(w_vals)
    else:
        w_vals = np.copy(w_vals)
        mask = (np.sign((w / np.pi * 180 - 180) * (w_vals / np.pi * 180 - 180)) < 0)
        w_vals[mask] = w_vals[mask] + np.sign(w / np.pi * 180 - 180) * 2 * np.pi
    # simple fit params
    opt_e, opt_w, opt_i_rad, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio = par_opt
    opt_ecosw, opt_esinw = opt_e * np.cos(opt_w), opt_e * np.sin(opt_w)
    opt_i = opt_i_rad * r2d
    par_opt_a = np.array([opt_ecosw, opt_esinw, opt_i, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio])
    par_opt_b = np.array([opt_e, opt_w * r2d, opt_i, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio])
    ecosw_vals = e_vals * np.cos(w_vals)
    esinw_vals = e_vals * np.sin(w_vals)
    # stack dists and plot
    value_names = np.array([r'$e\cdot cos(w)$', r'$e\cdot sin(w)$', 'i (deg)', r'$\frac{r_1+r_2}{a}$',
                            r'$\frac{r_2}{r_1}$', r'$\frac{sb_2}{sb_1}$'])
    dist_data = np.column_stack((ecosw_vals, esinw_vals, i_vals * r2d, rsumsma_vals, rratio_vals, sbratio_vals))
    value_range = np.max(dist_data, axis=0) - np.min(dist_data, axis=0)
    nonzero_range = (value_range != 0) & (value_range != np.inf)  # nonzero and finite
    fig = corner.corner(dist_data[:, nonzero_range], labels=value_names[nonzero_range], quiet=True)
    corner.overplot_lines(fig, par_init_a[nonzero_range], color='tab:blue')
    corner.overplot_points(fig, [par_init_a[nonzero_range]], marker='s', color='tab:blue')
    corner.overplot_lines(fig, par_opt_a[nonzero_range], color='tab:orange')
    corner.overplot_points(fig, [par_opt_a[nonzero_range]], marker='s', color='tab:orange')
    if not np.all(nonzero_range):
        fig.suptitle('Output distributions and lc fit outcome'
                     f' ({np.sum(nonzero_range)} of {len(nonzero_range)} shown)')
    else:
        fig.suptitle('Output distributions and lc fit outcome')
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
    dist_data = np.column_stack((e_vals, w_vals * r2d, i_vals * r2d, rsumsma_vals, rratio_vals, sbratio_vals))
    value_range = np.max(dist_data, axis=0) - np.min(dist_data, axis=0)
    nonzero_range = (value_range != 0) & (value_range != np.inf)  # nonzero and finite
    fig = corner.corner(dist_data[:, nonzero_range], labels=value_names[nonzero_range], quiet=True)
    corner.overplot_lines(fig, par_init_b[nonzero_range], color='tab:blue')
    corner.overplot_points(fig, [par_init_b[nonzero_range]], marker='s', color='tab:blue')
    corner.overplot_lines(fig, par_opt_b[nonzero_range], color='tab:orange')
    corner.overplot_points(fig, [par_opt_b[nonzero_range]], marker='s', color='tab:orange')
    if not np.all(nonzero_range):
        fig.suptitle('Output distributions and lc fit outcome'
                     f' ({np.sum(nonzero_range)} of {len(nonzero_range)} shown)')
    else:
        fig.suptitle('Output distributions and lc fit outcome')
    plt.tight_layout()
    if save_file is not None:
        if save_file.endswith('.png'):
            fig_save_file = save_file.replace('.png', '_out_b.png')
        else:
            fig_save_file = save_file + '_out.png'
        fig.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_pd_disentangled_freqs(times, signal, p_orb, t_zero, noise_level, const_r, slope_r, f_n_r, a_n_r, ph_n_r,
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
    e, w, i, r_sum_sma, r_ratio, sb_ratio = param_lc
    f_c, f_s = e**0.5 * np.cos(w), e**0.5 * np.sin(w)
    # make the ellc model
    if (model == 'ellc'):
        ecl_model = tsfit.wrap_ellc_lc(times, p_orb, t_zero, f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, 0)
    else:
        ecl_model = tsfit.eclipse_physical_lc(times, p_orb, t_zero, e, w, i, r_sum_sma, r_ratio, sb_ratio)
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
            ax.plot([f_n_r[k], f_n_r[k]], [0, a_n_r[k]], linestyle='--', c='tab:red')
        else:
            ax.plot([f_n_r[k], f_n_r[k]], [0, a_n_r[k]], linestyle='--', c='tab:brown')
    ax.plot([], [], linestyle='--', c='tab:red', label='disentangled sinusoids passing criteria')
    ax.plot([], [], linestyle='--', c='tab:brown', label='disentangled sinusoids')
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


def plot_lc_disentangled_freqs(times, signal, p_orb, t_zero, const_r, slope_r, f_n_r, a_n_r, ph_n_r,
                               i_sectors, passed_r, param_lc, model='simple', save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the determination
    of orbital parameters using both the eclipse timings and the ellc light curve
    models over two consecutive fits.
    """
    # eclipse signal with disentangled frequencies
    model_linear_r = tsf.linear_curve(times, const_r, slope_r, i_sectors)
    model_sinusoid_r = tsf.sum_sines(times, f_n_r, a_n_r, ph_n_r)
    # model of passed frequencies
    model_r_p = tsf.linear_curve(times, const_r, slope_r, i_sectors)
    model_r_p += tsf.sum_sines(times, f_n_r[passed_r], a_n_r[passed_r], ph_n_r[passed_r])
    # unpack and define parameters
    e, w, i, r_sum_sma, r_ratio, sb_ratio = param_lc
    f_c, f_s = np.sqrt(e) * np.cos(w), np.sqrt(e) * np.sin(w)
    # make the ellc model
    if (model == 'ellc'):
        ecl_model = tsfit.wrap_ellc_lc(times, p_orb, t_zero, f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, 0)
    else:
        ecl_model = tsfit.eclipse_physical_lc(times, p_orb, t_zero, e, w, i, r_sum_sma, r_ratio, sb_ratio)
    model_sin_lin = model_sinusoid_r + model_linear_r
    model_full = ecl_model + model_sinusoid_r + model_linear_r
    # plot
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].scatter(times, signal, marker='.', label='original signal')
    ax[0].plot(times, model_full, c='tab:grey', label='(linear + sinusoid + eclipse) model')
    ax[0].plot(times, ecl_model, c='tab:red', label='eclipse model')
    ax[0].set_ylabel('normalised flux/model')
    ax[0].legend()
    # residuals
    ax[1].scatter(times, signal - ecl_model, marker='.', label='eclipse model residuals')
    ax[1].scatter(times, signal - model_full, marker='.', label='(linear + sinusoid + eclipse) model residuals')
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


def plot_lc_disentangled_freqs_h(times, signal, p_orb, t_zero, timings, const_r, slope_r, f_n_r, a_n_r, ph_n_r,
                                 i_sectors, passed_r, passed_h, param_lc, model='simple', save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the determination
    of orbital parameters using both the eclipse timings and the ellc light curve
    models over two consecutive fits.
    """
    freq_res = 1.5 / np.ptp(times)  # Rayleigh criterion
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings
    # make the model times array, one full period plus the primary eclipse halves
    t_extended, ext_left, ext_right = tsf.fold_time_series(times, p_orb, t_zero, t_ext_1=t_1_1, t_ext_2=t_1_2)
    s_extended = np.concatenate((signal[ext_left], signal, signal[ext_right]))
    sorter = np.argsort(t_extended)
    # sinusoid and linear models
    model_sinusoid_r = tsf.sum_sines(times, f_n_r, a_n_r, ph_n_r)
    model_linear_r = tsf.linear_curve(times, const_r, slope_r, i_sectors)
    # unpack and define parameters
    e, w, i, r_sum_sma, r_ratio, sb_ratio = param_lc
    f_c, f_s = np.sqrt(e) * np.cos(w), np.sqrt(e) * np.sin(w)
    # make the eclipse model
    if (model == 'ellc'):
        model_ecl = tsfit.wrap_ellc_lc(times, p_orb, t_zero, f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, 0)
    else:
        model_ecl = tsfit.eclipse_physical_lc(times, p_orb, t_zero, e, w, i, r_sum_sma, r_ratio, sb_ratio)
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
    model_r_h = tsf.sum_sines(times, f_n_r[passed_h], a_n_r[passed_h], ph_n_r[passed_h])
    model_r_h = np.concatenate((model_r_h[ext_left], model_r_h, model_r_h[ext_right]))
    # model of passed frequencies
    if np.any(passed_r):
        passed_hr = passed_r & passed_h
        model_r_p_h = tsf.sum_sines(times, f_n_r[passed_hr], a_n_r[passed_hr], ph_n_r[passed_hr])
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
    ax[0].plot(t_extended[sorter], model_ecl_2[sorter], c='tab:red', label='eclipse model')
    ax[0].plot([t_1_1, t_1_1], s_minmax, '--', c='grey', label='previous eclipse edges')
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
    ax[1].plot([t_1_1, t_1_1], s_minmax_r, '--', c='grey', label='previous eclipse edges')
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


def plot_mcmc_pair(inf_data, t_zero, ecosw, esinw, i, phi_0, r_rat, sb_rat, const, slope, f_n, a_n, ph_n):
    """Show the pymc3 sampling results in several pair plots"""
    az.plot_pair(inf_data, var_names=['f_n', 'a_n', 'ph_n'],
                 coords={'f_n_dim_0': [0, 1], 'a_n_dim_0': [0, 1], 'ph_n_dim_0': [0, 1]},
                 marginals=True, kind=['scatter', 'kde'])
    az.plot_pair(inf_data, var_names=['const', 'slope', 'f_n', 'a_n', 'ph_n'],
                 coords={'const_dim_0': [0], 'slope_dim_0': [0], 'f_n_dim_0': [0], 'a_n_dim_0': [0], 'ph_n_dim_0': [0]},
                 marginals=True, kind=['scatter', 'kde'])
    az.plot_pair(inf_data, var_names=['t_zero', 'ecosw', 'esinw', 'incl', 'phi_0', 'r_rat', 'sb_rat'],
                 marginals=True, kind=['scatter', 'kde'])
    return


def plot_mcmc_trace(inf_data, t_zero, ecosw, esinw, i, phi_0, r_rat, sb_rat, const, slope, f_n, a_n, ph_n):
    """Show the pymc3 sampling results in a trace plot"""
    par_lines = [('t_zero', {}, t_zero), ('const', {}, const), ('slope', {}, slope),
                 ('f_n', {}, f_n), ('a_n', {}, a_n), ('ph_n', {}, ph_n),
                 ('e_cos_w', {}, ecosw), ('e_sin_w', {}, esinw), ('incl', {}, i),
                 ('phi_0', {}, phi_0), ('r_ratio', {}, r_rat), ('sb_ratio', {}, sb_rat)]
    az.plot_trace(inf_data, combined=False, compact=True, rug=True, divergences='top', lines=par_lines)
    return
