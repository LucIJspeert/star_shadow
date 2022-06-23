"""STAR SHADOW
Satellite Time-series Analysis Routine using
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
import arviz
import corner

from . import timeseries_functions as tsf
from . import timeseries_fitting as tsfit
from . import analysis_functions as af
from . import utility as ut


def plot_pd_single_output(times, signal, const, slope, f_n, a_n, ph_n, n_param, bic, i_sectors, title, zoom=None,
                          annotate=False, save_file=None, show=True):
    """Plot the periodogram with the output of one frequency analysis step.
    Primarily meant for the visualisation of the extraction process.
    """
    # preparations
    model = tsf.linear_curve(times, const, slope, i_sectors)  # the linear part of the model
    model += tsf.sum_sines(times, f_n, a_n, ph_n)  # the sinusoid part of the model
    freqs, ampls = tsf.astropy_scargle(times, signal - np.mean(signal))
    freqs_r, ampls_r = tsf.astropy_scargle(times, signal - model)
    err = tsf.formal_uncertainties(times, signal - model, a_n, i_sectors)
    a_height = np.max(ampls)
    a_width = np.max(freqs)
    # plotting
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(title, fontsize=14)
    fgrid = mgrid.GridSpec(nrows=2, ncols=1, height_ratios=(6, 3))
    fsubgrid = mgrid.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=fgrid[1], hspace=0, height_ratios=(2, 1))
    ax0 = fig.add_subplot(fgrid[0])
    ax1 = fig.add_subplot(fsubgrid[0])
    ax2 = fig.add_subplot(fsubgrid[1], sharex=ax1)
    ax0.text(0.5, 0.95, f'(n_param: {n_param}, BIC: {bic:1.2f})', fontsize=14, horizontalalignment='center',
             verticalalignment='center', transform=ax0.transAxes)
    ax0.plot(freqs, ampls, label='signal')
    ax0.plot(freqs_r, ampls_r, label=f'residual')
    for i in range(len(f_n)):
        ax0.errorbar([f_n[i], f_n[i]], [0, a_n[i]], xerr=[0, err[2][i]], yerr=[0, err[3][i]],
                     linestyle=':', capsize=2, c='tab:red')
        if annotate:
            ax0.annotate(f'{i+1}', (f_n[i], a_n[i] + 1.1 * err[3][i]), alpha=0.6)
        if (i == len(f_n) - 1):
            ax0.arrow(f_n[i], -0.02 * a_height, 0, 0, head_length=0.01 * a_height, head_width=0.005 * a_width,
                      width=0.005 * a_width, color='blue', head_starts_at_zero=False)
    ax1.scatter(times, signal, marker='.', label='signal')
    ax1.scatter(times, model, marker='.', label=f'model')
    ax2.scatter(times, signal - model, marker='.')
    ax0.set_xlabel('frequency (1/d)', fontsize=14)
    ax2.set_xlabel('time (d)', fontsize=14)
    ax0.set_ylabel('amplitude', fontsize=14)
    ax1.set_ylabel('signal', fontsize=14)
    ax2.set_ylabel('residual', fontsize=14)
    ax0.legend(loc='upper right', fontsize=12)
    ax1.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    # create inset
    if zoom is not None:
        a_width_ins = zoom[1] - zoom[0]
        ax0_ins = ax0.inset_axes([0.4, 0.4, 0.595, 0.59], transform=ax0.transAxes)
        ax0_ins.plot(freqs, ampls, label='signal')
        ax0_ins.plot(freqs_r, ampls_r, label='residual')
        for i in range(len(f_n)):
            if (f_n[i] > zoom[0]) & (f_n[i] < zoom[1]):
                ax0_ins.errorbar([f_n[i], f_n[i]], [0, a_n[i]], xerr=[0, err[2][i]], yerr=[0, err[3][i]],
                                 linestyle=':', capsize=2, c='tab:red')
            if (i == (len(f_n) - 1)):
                ax0_ins.arrow(f_n[i], -0.02 * a_height, 0, 0, head_length=0.01 * a_height,
                              head_width=0.005 * a_width_ins, width=0.005 * a_width_ins, color='blue',
                              head_starts_at_zero=False)
        ax0_ins.set_xlim(zoom[0], zoom[1])
        ax0_ins.text(0.5, 0.95, f'(n_param: {n_param}, BIC: {bic:1.2f})', fontsize=14, ha='center', va='center',
                     transform=ax0.transAxes)
        ax0.indicate_inset_zoom(ax0_ins, edgecolor="black")
    else:
        ax0.text(0.5, 0.95, f'(n_param: {n_param}, BIC: {bic:1.2f})', fontsize=14, ha='center', va='center',
                 transform=ax0.transAxes)
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_pd_full_output(times, signal, models, p_orb_i, f_n_i, a_n_i, i_half_s, save_file=None, show=True):
    """Plot the periodogram with the full output of the analysis recipe."""
    # make periodograms
    freqs, ampls = tsf.astropy_scargle(times, signal - np.mean(signal))
    freqs_1, ampls_1 = tsf.astropy_scargle(times, signal - models[0] - np.all(models[0] == 0) * np.mean(signal))
    freqs_2, ampls_2 = tsf.astropy_scargle(times, signal - models[1] - np.all(models[1] == 0) * np.mean(signal))
    freqs_3, ampls_3 = tsf.astropy_scargle(times, signal - models[2] - np.all(models[2] == 0) * np.mean(signal))
    freqs_4, ampls_4 = tsf.astropy_scargle(times, signal - models[3] - np.all(models[3] == 0) * np.mean(signal))
    freqs_5, ampls_5 = tsf.astropy_scargle(times, signal - models[4] - np.all(models[4] == 0) * np.mean(signal))
    freqs_6, ampls_6 = tsf.astropy_scargle(times, signal - models[5] - np.all(models[5] == 0) * np.mean(signal))
    freqs_7, ampls_7 = tsf.astropy_scargle(times, signal - models[6] - np.all(models[6] == 0) * np.mean(signal))
    freqs_8, ampls_8 = tsf.astropy_scargle(times, signal - models[7] - np.all(models[7] == 0) * np.mean(signal))
    # get error values
    err_1 = tsf.formal_uncertainties(times, signal - models[0], a_n_i[0], i_half_s)
    err_2 = tsf.formal_uncertainties(times, signal - models[1], a_n_i[1], i_half_s)
    err_3 = tsf.formal_uncertainties(times, signal - models[2], a_n_i[2], i_half_s)
    err_4 = tsf.formal_uncertainties(times, signal - models[3], a_n_i[3], i_half_s)
    err_5 = tsf.formal_uncertainties(times, signal - models[4], a_n_i[4], i_half_s)
    err_6 = tsf.formal_uncertainties(times, signal - models[5], a_n_i[5], i_half_s)
    err_7 = tsf.formal_uncertainties(times, signal - models[6], a_n_i[6], i_half_s)
    err_8 = tsf.formal_uncertainties(times, signal - models[7], a_n_i[7], i_half_s)
    # plot
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(freqs, ampls, label='signal')
    ax.plot(freqs_1, ampls_1, label='extraction residual')
    ax.plot(freqs_2, ampls_2, label='NL-LS fit residual')
    ax.plot(freqs_3, ampls_3, label='fixed harmonics residual')
    ax.plot(freqs_4, ampls_4, label='extra harmonics residual')
    ax.plot(freqs_5, ampls_5, label='extra non-harmonics residual')
    ax.plot(freqs_6, ampls_6, label='NL-LS fit residual with harmonics')
    ax.plot(freqs_7, ampls_7, label='Reduced frequencies')
    ax.plot(freqs_8, ampls_8, label='second NL-LS fit residual with harmonics')
    if (p_orb_i[7] > 0):
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_i[7], p_orb_i[7])
        p_err = tsf.formal_period_uncertainty(p_orb_i[7], err_8[2], harmonics, harmonic_n)
        ax.errorbar([1/p_orb_i[7], 1/p_orb_i[7]], [0, np.max(ampls)],
                    xerr=[0, p_err/p_orb_i[7]**2], yerr=[0, p_err/p_orb_i[7]**2],
                    linestyle='--', capsize=2, c='k', label=f'orbital frequency (p={p_orb_i[7]:1.4f}d)')
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
        ax.annotate(f'{i+1}', (f_n_i[7][i], a_n_i[7][i]))
    plt.xlabel('frequency (1/d)', fontsize=14)
    plt.ylabel('amplitude', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_harmonic_output(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_half_s, save_file=None, show=True):
    """Shows the separated harmonics in several ways"""
    # make models
    model = tsf.linear_curve(times, const, slope, i_half_s)
    model += tsf.sum_sines(times, f_n, a_n, ph_n)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    model_line = tsf.linear_curve(times, const, slope, i_half_s)
    model_h = tsf.sum_sines(times, f_n[harmonics], a_n[harmonics], ph_n[harmonics])
    model_nh = tsf.sum_sines(times, np.delete(f_n, harmonics), np.delete(a_n, harmonics),
                             np.delete(ph_n, harmonics))
    errors = tsf.formal_uncertainties(times, signal - model, a_n, i_half_s)
    p_err = tsf.formal_period_uncertainty(p_orb, errors[2], harmonics, harmonic_n)
    # plot the full model light curve
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))
    ax[0].scatter(times, signal, label='signal')
    ax[0].plot(times, model, marker='.', c='tab:orange', label='full model (linear + sinusoidal)')
    ax[1].scatter(times, signal - model, marker='.')
    ax[0].set_ylabel('signal/model', fontsize=14)
    ax[0].legend(fontsize=12)
    ax[1].set_ylabel('residual', fontsize=14)
    ax[1].set_xlabel('time (d)', fontsize=14)
    plt.tight_layout()
    if save_file is not None:
        if save_file.endswith('.png'):
            fig_save_file = save_file.replace('.png', '_1.png')
        else:
            fig_save_file = save_file + '_1.png'
        plt.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    # plot the harmonic model and non-harmonic model
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))
    ax[0].scatter(times, signal - model_nh, marker='.', c='tab:blue', label='signal - non-harmonics')
    ax[0].plot(times, model_line + model_h, marker='.', c='tab:orange', label='linear + harmonic model')
    ax[1].scatter(times, signal - model_h, marker='.', c='tab:blue', label='signal - harmonics')
    ax[1].plot(times, model_line + model_nh, marker='.', c='tab:orange', label='linear + non-harmonic model')
    ax[0].set_ylabel('residual/model', fontsize=14)
    ax[0].legend(fontsize=12)
    ax[1].set_ylabel('residual/model', fontsize=14)
    ax[1].set_xlabel('time (d)', fontsize=14)
    ax[1].legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        if save_file.endswith('.png'):
            fig_save_file = save_file.replace('.png', '_2.png')
        else:
            fig_save_file = save_file + '_2.png'
        plt.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    # periodogram non-harmonics
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    freqs_nh, ampls_nh = tsf.astropy_scargle(times, signal - model_h - model_line)
    freqs, ampls = tsf.astropy_scargle(times, signal - model)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(freqs_nh, ampls_nh, label='residuals after harmonic removal')
    ax.plot(freqs, ampls, label='final residuals')
    ax.plot([1/p_orb, 1/p_orb], [0, np.max(ampls_nh)], linestyle='--', c='grey', alpha=0.5,
            label=f'orbital frequency (p={p_orb:1.4f}d)')
    for i in range(2, np.max(harmonic_n)):
        ax.plot([i/p_orb, i/p_orb], [0, np.max(ampls_nh)], linestyle='--', c='grey', alpha=0.5)
    for i in range(len(f_n[non_harm])):
        ax.errorbar([f_n[non_harm][i], f_n[non_harm][i]], [0, a_n[non_harm][i]],
                    xerr=[0, errors[2][non_harm][i]], yerr=[0, errors[3][non_harm][i]],
                    linestyle=':', capsize=2, c='tab:orange')
        ax.annotate(f'{i+1}', (f_n[non_harm][i], a_n[non_harm][i]))
    plt.xlabel('frequency (1/d)', fontsize=14)
    plt.ylabel('amplitude', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        if save_file.endswith('.png'):
            fig_save_file = save_file.replace('.png', '_3.png')
        else:
            fig_save_file = save_file + '_3.png'
        plt.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    # periodogram harmonics
    freqs_h, ampls_h = tsf.astropy_scargle(times, signal - model_nh - model_line)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(freqs_h, ampls_h, label='residuals after non-harmonic removal')
    ax.plot(freqs, ampls, label='final residuals')
    ax.errorbar([1/p_orb, 1/p_orb], [0, np.max(ampls)], xerr=[0, p_err/p_orb**2], yerr=[0, p_err/p_orb**2],
                linestyle='--', capsize=2, c='k', label=f'orbital frequency (p={p_orb:1.4f}d)')
    for i in range(len(f_n[harmonics])):
        ax.errorbar([f_n[harmonics][i], f_n[harmonics][i]], [0, a_n[harmonics][i]],
                    xerr=[0, errors[2][harmonics][i]], yerr=[0, errors[3][harmonics][i]],
                    linestyle=':', capsize=2, c='tab:orange')
        ax.annotate(f'{i+1}', (f_n[harmonics][i], a_n[harmonics][i]))
    plt.xlabel('frequency (1/d)', fontsize=14)
    plt.ylabel('amplitude', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        if save_file.endswith('.png'):
            fig_save_file = save_file.replace('.png', '_4.png')
        else:
            fig_save_file = save_file + '_4.png'
        plt.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_eclipse_timestamps(times, signal, p_orb, t_zero, timings, depths, timings_b, timing_errs, depths_err,
                               const, slope, f_n, a_n, ph_n, i_sectors, save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the first and
    last contact points as well as minima and midpoints indicated.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2 = timings
    t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2 = timings_b
    t_1_err, t_2_err, tau_1_1_err, tau_1_2_err, tau_2_1_err, tau_2_2_err = timing_errs
    # plotting bounds
    t_start = t_1_1 - 6 * tau_1_1_err
    t_end = p_orb + t_1_2 + 6 * tau_1_2_err
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    t_model = np.arange(t_start, t_end, 0.001)
    model_h = 1 + tsf.sum_sines(t_zero + t_model, f_n[harmonics], a_n[harmonics], ph_n[harmonics])
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line + 1
    s_minmax = np.array([np.min(signal), np.max(signal)])
    folded = (times - t_zero) % p_orb
    extend_l = (folded > p_orb + t_start)
    extend_r = (folded < t_end - p_orb)
    h_adjust = 1 - np.mean(signal)
    # heights at minimum
    h_1 = np.min(model_h[(t_model > t_1_1) & (t_model < t_1_2)])
    h_2 = np.min(model_h[(t_model > t_2_1) & (t_model < t_2_2)])
    # plot
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(folded, signal + h_adjust, marker='.', label='original folded signal')
    ax.scatter(folded[extend_l] - p_orb, signal[extend_l] + h_adjust, marker='.', c='tab:blue')
    ax.scatter(folded[extend_r] + p_orb, signal[extend_r] + h_adjust, marker='.', c='tab:blue')
    ax.scatter(folded, ecl_signal, marker='.', c='tab:orange', label='signal minus non-harmonics and linear curve')
    ax.scatter(folded[extend_l] - p_orb, ecl_signal[extend_l], marker='.', c='tab:orange')
    ax.scatter(folded[extend_r] + p_orb, ecl_signal[extend_r], marker='.', c='tab:orange')
    ax.plot(t_model, model_h, c='tab:green', label='harmonics')
    ax.plot([t_1, t_1], s_minmax, '--', c='tab:red', label='eclipse minimum')
    ax.plot([t_2, t_2], s_minmax, '--', c='tab:red')
    ax.plot([(t_1_1 + t_1_2)/2, (t_1_1 + t_1_2)/2], s_minmax, ':', c='tab:grey', label='eclipse midpoint')
    ax.plot([(t_2_1 + t_2_2)/2, (t_2_1 + t_2_2)/2], s_minmax, ':', c='tab:grey')
    ax.plot([t_1_1, t_1_1], s_minmax, '--', c='tab:purple', label=r'eclipse edges')
    ax.plot([t_1_2, t_1_2], s_minmax, '--', c='tab:purple')
    ax.plot([t_2_1, t_2_1], s_minmax, '--', c='tab:purple')
    ax.plot([t_2_2, t_2_2], s_minmax, '--', c='tab:purple')
    ax.plot([t_1_1, t_1_2], [h_1, h_1], '--', c='tab:pink')
    ax.plot([t_2_1, t_2_2], [h_2, h_2], '--', c='tab:pink')
    ax.plot([t_1_1, t_1_2], [h_1 + depths[0], h_1 + depths[0]], '--', c='tab:pink')
    ax.plot([t_2_1, t_2_2], [h_2 + depths[1], h_2 + depths[1]], '--', c='tab:pink')
    # 1 sigma errors
    ax.fill_between([t_1 - t_1_err, t_1 + t_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:red', alpha=0.3)
    ax.fill_between([t_2 - t_2_err, t_2 + t_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:red', alpha=0.3)
    ax.fill_between([t_1_1 - tau_1_1_err, t_1_1 + tau_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.3, label=r'1 and 3 $\sigma$ error')
    ax.fill_between([t_1_2 - tau_1_2_err, t_1_2 + tau_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.3)
    ax.fill_between([t_2_1 - tau_2_1_err, t_2_1 + tau_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.3)
    ax.fill_between([t_2_2 - tau_2_2_err, t_2_2 + tau_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.3)
    ax.fill_between([t_1_1, t_1_2], y1=[h_1 + depths_err[0], h_1 + depths_err[0]],
                    y2=[h_1 - depths_err[0], h_1 - depths_err[0]], color='tab:pink', alpha=0.3)
    ax.fill_between([t_2_1, t_2_2], y1=[h_2 + depths_err[1], h_2 + depths_err[1]],
                    y2=[h_2 - depths_err[1], h_2 - depths_err[1]], color='tab:pink', alpha=0.3)
    # 3 sigma errors
    ax.fill_between([t_1 - 3*t_1_err, t_1 + 3*t_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:red', alpha=0.2)
    ax.fill_between([t_2 - 3*t_2_err, t_2 + 3*t_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:red', alpha=0.2)
    ax.fill_between([t_1_1 - 3*tau_1_1_err, t_1_1 + 3*tau_1_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.2)
    ax.fill_between([t_1_2 - 3*tau_1_2_err, t_1_2 + 3*tau_1_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.2)
    ax.fill_between([t_2_1 - 3*tau_2_1_err, t_2_1 + 3*tau_2_1_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.2)
    ax.fill_between([t_2_2 - 3*tau_2_2_err, t_2_2 + 3*tau_2_2_err], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]],
                    color='tab:purple', alpha=0.2)
    ax.fill_between([t_1_1, t_1_2], y1=[h_1 + 3*depths_err[0], h_1 + 3*depths_err[0]],
                    y2=[h_1 - 3*depths_err[0], h_1 - 3*depths_err[0]], color='tab:pink', alpha=0.2)
    ax.fill_between([t_2_1, t_2_2], y1=[h_2 + 3*depths_err[1], h_2 + 3*depths_err[1]],
                    y2=[h_2 - 3*depths_err[1], h_2 - 3*depths_err[1]], color='tab:pink', alpha=0.2)
    # flat bottom
    if (t_b_1_2 - t_b_1_1 != 0) | (t_b_2_1 - t_b_2_2 != 0):
        ax.fill_between([t_b_1_1, t_b_1_2], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:brown', alpha=0.3,
                        label='flat bottom')
        ax.fill_between([t_b_2_1, t_b_2_2], y1=s_minmax[[0, 0]], y2=s_minmax[[1, 1]], color='tab:brown', alpha=0.3)
    ax.set_xlabel(r'$(time - t_0)\ mod\ P_{orb}$ (d)', fontsize=14)
    ax.set_ylabel('normalised flux', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_derivatives(p_orb, f_h, a_h, ph_h, ecl_indices, save_file=None, show=True):
    """Shows the light curve and three time derivatives with the significant
    points on the curves used to identify the eclipses
    """
    # make a timeframe from 0 to two P to catch both eclipses in full if present
    t_model = np.arange(0, 2 * p_orb + 0.00001, 0.00001)  # 0.864 second steps if we work in days and per day units
    model_h = tsf.sum_sines(t_model, f_h, a_h, ph_h)
    # analytic derivatives
    deriv_1 = tsf.sum_sines_deriv(t_model, f_h, a_h, ph_h, deriv=1)
    deriv_2 = tsf.sum_sines_deriv(t_model, f_h, a_h, ph_h, deriv=2)
    deriv_3 = tsf.sum_sines_deriv(t_model, f_h, a_h, ph_h, deriv=3)
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(16, 9))
    ax[0].plot(t_model, model_h)
    ax[0].scatter(t_model[ecl_indices[:, 3]], model_h[ecl_indices[:, 3]], c='tab:blue', marker='o', label='peaks_1')
    ax[0].scatter(t_model[ecl_indices[:, -4]], model_h[ecl_indices[:, -4]], c='tab:blue', marker='o')
    ax[0].scatter(t_model[ecl_indices[:, 0]], model_h[ecl_indices[:, 0]], c='tab:orange', marker='>', label='zeros_1')
    ax[0].scatter(t_model[ecl_indices[:, -1]], model_h[ecl_indices[:, -1]], c='tab:orange', marker='<')
    ax[0].scatter(t_model[ecl_indices[:, 2]], model_h[ecl_indices[:, 2]], c='tab:green', marker='v', label='peaks_2_n')
    ax[0].scatter(t_model[ecl_indices[:, -3]], model_h[ecl_indices[:, -3]], c='tab:green', marker='v')
    ax[0].scatter(t_model[ecl_indices[:, 1]], model_h[ecl_indices[:, 1]], c='tab:red', marker='d', label='minimum_1')
    ax[0].scatter(t_model[ecl_indices[:, -2]], model_h[ecl_indices[:, -2]], c='tab:red', marker='d')
    ax[0].scatter(t_model[ecl_indices[:, 4]], model_h[ecl_indices[:, 4]], c='tab:purple', marker='^', label='peaks_2_p')
    ax[0].scatter(t_model[ecl_indices[:, -5]], model_h[ecl_indices[:, -5]], c='tab:purple', marker='^')
    ax[0].scatter(t_model[ecl_indices[:, 5]], model_h[ecl_indices[:, 5]], c='tab:brown', marker='s', label='minimum_0')
    ax[0].set_ylabel(r'$\mathscr{l}$', fontsize=14)
    ax[0].legend()
    ax[1].plot(t_model, deriv_1)
    ax[1].plot(t_model, np.zeros(len(t_model)), '--', c='tab:grey')
    ax[1].scatter(t_model[ecl_indices[:, 3]], deriv_1[ecl_indices[:, 3]], c='tab:blue', marker='o')
    ax[1].scatter(t_model[ecl_indices[:, -4]], deriv_1[ecl_indices[:, -4]], c='tab:blue', marker='o')
    ax[1].scatter(t_model[ecl_indices[:, 0]], deriv_1[ecl_indices[:, 0]], c='tab:orange', marker='>')
    ax[1].scatter(t_model[ecl_indices[:, -1]], deriv_1[ecl_indices[:, -1]], c='tab:orange', marker='<')
    ax[1].scatter(t_model[ecl_indices[:, 2]], deriv_1[ecl_indices[:, 2]], c='tab:green', marker='v')
    ax[1].scatter(t_model[ecl_indices[:, -3]], deriv_1[ecl_indices[:, -3]], c='tab:green', marker='v')
    ax[1].scatter(t_model[ecl_indices[:, 1]], deriv_1[ecl_indices[:, 1]], c='tab:red', marker='d')
    ax[1].scatter(t_model[ecl_indices[:, -2]], deriv_1[ecl_indices[:, -2]], c='tab:red', marker='d')
    ax[1].scatter(t_model[ecl_indices[:, 4]], deriv_1[ecl_indices[:, 4]], c='tab:purple', marker='^')
    ax[1].scatter(t_model[ecl_indices[:, -5]], deriv_1[ecl_indices[:, -5]], c='tab:purple', marker='^')
    ax[1].scatter(t_model[ecl_indices[:, 5]], deriv_1[ecl_indices[:, 5]], c='tab:brown', marker='s')
    ax[1].set_ylabel(r'$\frac{d\mathscr{l}}{dt}$', fontsize=14)
    ax[2].plot(t_model, deriv_2)
    ax[2].plot(t_model, np.zeros(len(t_model)), '--', c='tab:grey')
    ax[2].scatter(t_model[ecl_indices[:, 3]], deriv_2[ecl_indices[:, 3]], c='tab:blue', marker='o')
    ax[2].scatter(t_model[ecl_indices[:, -4]], deriv_2[ecl_indices[:, -4]], c='tab:blue', marker='o')
    ax[2].scatter(t_model[ecl_indices[:, 0]], deriv_2[ecl_indices[:, 0]], c='tab:orange', marker='>')
    ax[2].scatter(t_model[ecl_indices[:, -1]], deriv_2[ecl_indices[:, -1]], c='tab:orange', marker='<')
    ax[2].scatter(t_model[ecl_indices[:, 2]], deriv_2[ecl_indices[:, 2]], c='tab:green', marker='v')
    ax[2].scatter(t_model[ecl_indices[:, -3]], deriv_2[ecl_indices[:, -3]], c='tab:green', marker='v')
    ax[2].scatter(t_model[ecl_indices[:, 1]], deriv_2[ecl_indices[:, 1]], c='tab:red', marker='d')
    ax[2].scatter(t_model[ecl_indices[:, -2]], deriv_2[ecl_indices[:, -2]], c='tab:red', marker='d')
    ax[2].scatter(t_model[ecl_indices[:, 4]], deriv_2[ecl_indices[:, 4]], c='tab:purple', marker='^')
    ax[2].scatter(t_model[ecl_indices[:, -5]], deriv_2[ecl_indices[:, -5]], c='tab:purple', marker='^')
    ax[2].scatter(t_model[ecl_indices[:, 5]], deriv_2[ecl_indices[:, 5]], c='tab:brown', marker='s')
    ax[2].set_ylabel(r'$\frac{d^2\mathscr{l}}{dt^2}$', fontsize=14)
    ax[3].plot(t_model, deriv_3)
    ax[3].plot(t_model, np.zeros(len(t_model)), '--', c='tab:grey')
    ax[3].scatter(t_model[ecl_indices[:, 3]], deriv_3[ecl_indices[:, 3]], c='tab:blue', marker='o')
    ax[3].scatter(t_model[ecl_indices[:, -4]], deriv_3[ecl_indices[:, -4]], c='tab:blue', marker='o')
    ax[3].scatter(t_model[ecl_indices[:, 0]], deriv_3[ecl_indices[:, 0]], c='tab:orange', marker='>')
    ax[3].scatter(t_model[ecl_indices[:, -1]], deriv_3[ecl_indices[:, -1]], c='tab:orange', marker='<')
    ax[3].scatter(t_model[ecl_indices[:, 2]], deriv_3[ecl_indices[:, 2]], c='tab:green', marker='v')
    ax[3].scatter(t_model[ecl_indices[:, -3]], deriv_3[ecl_indices[:, -3]], c='tab:green', marker='v')
    ax[3].scatter(t_model[ecl_indices[:, 1]], deriv_3[ecl_indices[:, 1]], c='tab:red', marker='d')
    ax[3].scatter(t_model[ecl_indices[:, -2]], deriv_3[ecl_indices[:, -2]], c='tab:red', marker='d')
    ax[3].scatter(t_model[ecl_indices[:, 4]], deriv_3[ecl_indices[:, 4]], c='tab:purple', marker='^')
    ax[3].scatter(t_model[ecl_indices[:, -5]], deriv_3[ecl_indices[:, -5]], c='tab:purple', marker='^')
    ax[3].scatter(t_model[ecl_indices[:, 5]], deriv_3[ecl_indices[:, 5]], c='tab:brown', marker='s')
    ax[3].set_xlabel('time (d)', fontsize=14)
    ax[3].set_ylabel(r'$\frac{d^3\mathscr{l}}{dt^3}$', fontsize=14)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_harmonic_separation(times, signal, p_orb, t_zero, timings, const, slope, f_n, a_n, ph_n, const_ho, f_ho,
                                a_ho, ph_ho, f_he, a_he, ph_he, i_sectors, save_file=None, show=True):
    """Shows the separation of the harmonics froming the eclipses and the ones
    forming the out-of-eclipse harmonic variability.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2 = timings
    # plotting bounds
    t_start = t_1_1
    t_end = p_orb + t_1_2
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    t_model = np.arange(t_start, t_end, 0.001)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_h = 1 + tsf.sum_sines(t_model + t_zero, f_n[harmonics], a_n[harmonics], ph_n[harmonics])
    model_ho = 1 + const_ho + tsf.sum_sines(t_model + t_zero, f_ho, a_ho, ph_ho)
    model_ho_t = const_ho + tsf.sum_sines(times, f_ho, a_ho, ph_ho)
    model_he = 1 + tsf.sum_sines(t_model + t_zero, f_he, a_he, ph_he)
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal_1 = signal - model_nh - model_line + 1
    ecl_signal_2 = signal - model_nh - model_ho_t - model_line + 1
    # some plotting parameters and extension masks
    s_minmax = np.array([np.min(signal), np.max(signal)])
    folded = (times - t_zero) % p_orb
    extend_l = (folded > p_orb + t_start)
    extend_r = (folded < t_end - p_orb)
    # h_adjust = 1 - np.mean(signal)
    # plot
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(folded, ecl_signal_1, marker='.', label='original folded signal')
    ax.scatter(folded[extend_l] - p_orb, ecl_signal_1[extend_l], marker='.', c='tab:blue')
    ax.scatter(folded[extend_r] + p_orb, ecl_signal_1[extend_r], marker='.', c='tab:blue')
    ax.scatter(folded, ecl_signal_2, marker='.', c='tab:orange',
               label='signal minus non-harmonics, o.o.e.-harmonics and linear curve')
    ax.scatter(folded[extend_l] - p_orb, ecl_signal_2[extend_l], marker='.', c='tab:orange')
    ax.scatter(folded[extend_r] + p_orb, ecl_signal_2[extend_r], marker='.', c='tab:orange')
    ax.plot(t_model, model_he, c='tab:green', label='i.e.-harmonics')
    ax.plot(t_model, model_h, '--', c='tab:red', label='harmonics')
    ax.plot(t_model, model_ho, c='tab:purple', label='o.o.e.-harmonics')
    # ax.plot(t_model, model_ho + model_he - 1, '--', c='tab:brown', label='o.o.e. + i.e. harmonics')
    ax.plot([t_1_1, t_1_1], s_minmax, '--', c='tab:grey', label=r'eclipse edges')
    ax.plot([t_1_2, t_1_2], s_minmax, '--', c='tab:grey')
    ax.plot([t_2_1, t_2_1], s_minmax, '--', c='tab:grey')
    ax.plot([t_2_2, t_2_2], s_minmax, '--', c='tab:grey')
    ax.set_xlabel(r'$(time - t_0) mod(P_{orb})$ (d)', fontsize=14)
    ax.set_ylabel('normalised flux', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
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
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2 = timings
    # make the model times array, one full period plus the primary eclipse halves
    t_extended = (times - t_zero) % p_orb
    ext_left = (t_extended > p_orb + t_1_1)
    ext_right = (t_extended < t_1_2)
    t_extended = np.concatenate((t_extended[ext_left] - p_orb, t_extended, t_extended[ext_right] + p_orb))
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line + 1
    ecl_signal = np.concatenate((ecl_signal[ext_left], ecl_signal, ecl_signal[ext_right]))
    s_minmax = [np.min(ecl_signal), np.max(ecl_signal)]
    # determine a lc offset to match the model at the edges
    edge_level_1_1 = np.max(ecl_signal[(t_extended > t_1_1) & (t_extended < t_1)])
    edge_level_1_2 = np.max(ecl_signal[(t_extended > t_1) & (t_extended < t_1_2)])
    offset = 1 - min(edge_level_1_1, edge_level_1_2)
    # unpack and define parameters
    e, w, i, phi_0, r_sum_sma, r_ratio, sb_ratio = ecl_params
    # determine phase angles of crucial points
    theta_1, theta_2 = af.minima_phase_angles(e, w, i)
    nu_1 = af.true_anomaly(theta_1, w)
    phi_1_1, phi_1_2, phi_2_1, phi_2_2 = af.contact_phase_angles(e, w, i, phi_0)
    # make the simple model
    thetas_1 = np.arange(0-phi_1_1, 0+phi_1_2, 0.0001)
    thetas_2 = np.arange(np.pi-phi_2_1, np.pi+phi_2_2, 0.0001)
    model_ecl_1 = np.zeros(len(thetas_1))
    model_ecl_2 = np.zeros(len(thetas_2))
    for k in range(len(thetas_1)):
        model_ecl_1[k] = 1 - af.eclipse_depth(e, w, i, thetas_1[k], r_sum_sma, r_ratio, sb_ratio)
    for k in range(len(thetas_2)):
        model_ecl_2[k] = 1 - af.eclipse_depth(e, w, i, thetas_2[k], r_sum_sma, r_ratio, sb_ratio)
    t_model_1 = p_orb / (2 * np.pi) * af.integral_kepler_2(nu_1, af.true_anomaly(thetas_1, w), e)
    t_model_2 = p_orb / (2 * np.pi) * af.integral_kepler_2(nu_1, af.true_anomaly(thetas_2, w), e)
    # include out of eclipse
    thetas = np.arange(0-2*phi_1_1, np.pi+2*phi_2_2, 0.001)
    ecl_model = np.zeros(len(thetas))
    for k in range(len(thetas)):
        ecl_model[k] = 1 - af.eclipse_depth(e, w, i, thetas[k], r_sum_sma, r_ratio, sb_ratio)
    t_model = p_orb / (2 * np.pi) * af.integral_kepler_2(nu_1, af.true_anomaly(thetas, w), e)
    # plot
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(t_extended, ecl_signal + offset, marker='.', label='eclipse signal')
    ax.plot(t_model_1, model_ecl_1, c='tab:orange', label='spheres of uniform brightness')
    ax.plot(t_model_2, model_ecl_2, c='tab:orange')
    ax.plot(t_model, ecl_model, c='tab:purple', alpha=0.3)
    ax.plot([t_1_1, t_1_1], s_minmax, '--', c='grey', label='eclipse edges')
    ax.plot([t_1_2, t_1_2], s_minmax, '--', c='grey')
    ax.plot([t_2_1, t_2_1], s_minmax, '--', c='grey')
    ax.plot([t_2_2, t_2_2], s_minmax, '--', c='grey')
    ax.set_xlabel(r'$(time - t_0) mod(P_{orb})$ (d)', fontsize=14)
    ax.set_ylabel('normalised flux', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_dists_eclipse_parameters(e, w, i, phi_0, psi_0, r_sum_sma, r_dif_sma,  r_ratio, sb_ratio, e_vals, w_vals,
                                  i_vals, phi0_vals, psi0_vals, rsumsma_vals, rdifsma_vals, rratio_vals, sbratio_vals):
    """Shows the histograms resulting from the input distributions
    and the hdi_prob=0.683 and hdi_prob=0.997 bounds resulting from the HDI's

    Note: produces several plots
    """
    cos_w = np.cos(w)
    sin_w = np.sin(w)
    # inclination
    i_interval = arviz.hdi(i_vals, hdi_prob=0.683)
    i_bounds = arviz.hdi(i_vals, hdi_prob=0.997)
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist(i_vals/np.pi*180, bins=50, label='vary fit input')
    ax.plot([i/np.pi*180, i/np.pi*180], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([i_interval[0]/np.pi*180, i_interval[0]/np.pi*180], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([i_interval[1]/np.pi*180, i_interval[1]/np.pi*180], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([i_bounds[0]/np.pi*180, i_bounds[0]/np.pi*180], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([i_bounds[1]/np.pi*180, i_bounds[1]/np.pi*180], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('inclination (deg)', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # phi_0
    phi0_interval = arviz.hdi(phi0_vals, hdi_prob=0.683)
    phi0_bounds = arviz.hdi(phi0_vals, hdi_prob=0.997)
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist(phi0_vals, bins=50, label='vary fit input')
    ax.plot([phi_0, phi_0], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([phi0_interval[0], phi0_interval[0]], [0, np.max(hist[0])], c='tab:orange', label='hdi_prob=0.683')
    ax.plot([phi0_interval[1], phi0_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([phi0_bounds[0], phi0_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([phi0_bounds[1], phi0_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('phi_0', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # psi_0
    psi0_interval = arviz.hdi(psi0_vals, hdi_prob=0.683)
    psi0_bounds = arviz.hdi(psi0_vals, hdi_prob=0.997)
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist(psi0_vals, bins=50, label='vary fit input')
    ax.plot([psi_0, psi_0], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([psi0_interval[0], psi0_interval[0]], [0, np.max(hist[0])], c='tab:orange', label='hdi_prob=0.683')
    ax.plot([psi0_interval[1], psi0_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([psi0_bounds[0], psi0_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([psi0_bounds[1], psi0_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('psi_0', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # eccentricity
    e_interval = arviz.hdi(e_vals, hdi_prob=0.683)
    e_bounds = arviz.hdi(e_vals, hdi_prob=0.997)
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist(e_vals, bins=50, label='vary fit input')
    ax.plot([e, e], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([e_interval[0], e_interval[0]], [0, np.max(hist[0])], c='tab:orange', label='hdi_prob=0.683')
    ax.plot([e_interval[1], e_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([e_bounds[0], e_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--', label='hdi_prob=0.997')
    ax.plot([e_bounds[1], e_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('eccentricity', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # e*np.cos(w)
    ecosw_interval = arviz.hdi(e_vals*np.cos(w_vals), hdi_prob=0.683)
    ecosw_bounds = arviz.hdi(e_vals*np.cos(w_vals), hdi_prob=0.997)
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist(e_vals*np.cos(w_vals), bins=50, label='vary fit input')
    ax.plot([e*cos_w, e*cos_w], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([ecosw_interval[0], ecosw_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([ecosw_interval[1], ecosw_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([ecosw_bounds[0], ecosw_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([ecosw_bounds[1], ecosw_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('e cos(w)', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # e*np.sin(w)
    esinw_interval = arviz.hdi(e_vals*np.sin(w_vals), hdi_prob=0.683)
    esinw_bounds = arviz.hdi(e_vals*np.sin(w_vals), hdi_prob=0.997)
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist(e_vals*np.sin(w_vals), bins=50, label='vary fit input')
    ax.plot([e*sin_w, e*sin_w], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([esinw_interval[0], esinw_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([esinw_interval[1], esinw_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([esinw_bounds[0], esinw_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([esinw_bounds[1], esinw_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('e sin(w)', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # omega
    if (abs(w/np.pi*180 - 180) > 80) & (abs(w/np.pi*180 - 180) < 100):
        w_vals_hist = np.copy(w_vals)
        w_interval = arviz.hdi(w_vals, hdi_prob=0.683, multimodal=True)
        w_bounds = arviz.hdi(w_vals, hdi_prob=0.997, multimodal=True)
    else:
        w_vals_hist = np.copy(w_vals)
        mask = (np.sign((w/np.pi*180 - 180) * (w_vals/np.pi*180 - 180)) < 0)
        w_vals_hist[mask] = w_vals[mask] + np.sign(w/np.pi*180 - 180) * 2 * np.pi
        w_interval = arviz.hdi(w_vals - np.pi, hdi_prob=0.683, circular=True) + np.pi
        w_bounds = arviz.hdi(w_vals - np.pi, hdi_prob=0.997, circular=True) + np.pi
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist(w_vals_hist/np.pi*180, bins=50, label='vary fit input')
    ax.plot([w/np.pi*180, w/np.pi*180], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    if (abs(w/np.pi*180 - 180) > 80) & (abs(w/np.pi*180 - 180) < 100):
        ax.plot([(2*np.pi-w)/np.pi*180, (2*np.pi-w)/np.pi*180], [0, np.max(hist[0])], c='tab:pink',
                label='mirrored best fit value')
    if (len(np.shape(w_interval)) > 1):
        w_in_interval = (np.sign((w - w_interval)[:, 0] * (w - w_interval)[:, 1]) == -1)
        w_not_in_interval = (np.sign((w - w_interval)[:, 0] * (w - w_interval)[:, 1]) == 1)
        ax.plot([w_interval[w_in_interval, 0]/np.pi*180, w_interval[w_in_interval, 0]/np.pi*180], [0, np.max(hist[0])],
                c='tab:orange', label='hdi_prob=0.683')
        ax.plot([w_interval[w_in_interval, 1]/np.pi*180, w_interval[w_in_interval, 1]/np.pi*180], [0, np.max(hist[0])],
                c='tab:orange')
        ax.plot([w_interval[w_not_in_interval, 0]/np.pi*180, w_interval[w_not_in_interval, 0]/np.pi*180],
                [0, np.max(hist[0])], c='tab:orange', linestyle='--', label='hdi_prob=0.683')
        ax.plot([w_interval[w_not_in_interval, 1]/np.pi*180, w_interval[w_not_in_interval, 1]/np.pi*180],
                [0, np.max(hist[0])], c='tab:orange', linestyle='--')
    else:
        mask_int = (np.sign((w/np.pi*180 - 180) * (w_interval/np.pi*180 - 180)) < 0)
        w_interval_plot = np.copy(w_interval)
        w_interval_plot[mask_int] = w_interval[mask_int] + np.sign(w/np.pi*180 - 180) * 2 * np.pi
        ax.plot([w_interval_plot[0]/np.pi*180, w_interval_plot[0]/np.pi*180], [0, np.max(hist[0])], c='tab:orange',
                label='hdi_prob=0.683')
        ax.plot([w_interval_plot[1]/np.pi*180, w_interval_plot[1]/np.pi*180], [0, np.max(hist[0])], c='tab:orange')
    if (len(np.shape(w_bounds)) > 1):
        w_in_interval = (np.sign((w - w_bounds)[:, 0] * (w - w_bounds)[:, 1]) == -1)
        w_not_in_interval = (np.sign((w - w_bounds)[:, 0] * (w - w_bounds)[:, 1]) == 1)
        ax.plot([w_bounds[w_in_interval, 0]/np.pi*180, w_bounds[w_in_interval, 0]/np.pi*180], [0, np.max(hist[0])],
                c='tab:grey', linestyle='--', label='hdi_prob=0.683')
        ax.plot([w_bounds[w_in_interval, 1]/np.pi*180, w_bounds[w_in_interval, 1]/np.pi*180], [0, np.max(hist[0])],
                c='tab:grey', linestyle='--')
        ax.plot([w_bounds[w_not_in_interval, 0]/np.pi*180, w_bounds[w_not_in_interval, 0]/np.pi*180],
                [0, np.max(hist[0])], c='tab:grey', linestyle=':', label='hdi_prob=0.683')
        ax.plot([w_bounds[w_not_in_interval, 1]/np.pi*180, w_bounds[w_not_in_interval, 1]/np.pi*180],
                [0, np.max(hist[0])], c='tab:grey', linestyle=':')
    else:
        mask_bnd = (np.sign((w/np.pi*180 - 180) * (w_bounds/np.pi*180 - 180)) < 0)
        w_bounds_plot = np.copy(w_bounds)
        w_bounds_plot[mask_bnd] = w_bounds[mask_bnd] + np.sign(w/np.pi*180 - 180) * 2 * np.pi
        ax.plot([w_bounds_plot[0]/np.pi*180, w_bounds_plot[0]/np.pi*180], [0, np.max(hist[0])], c='tab:grey',
                linestyle='--', label='hdi_prob=0.683')
        ax.plot([w_bounds_plot[1]/np.pi*180, w_bounds_plot[1]/np.pi*180], [0, np.max(hist[0])], c='tab:grey',
                linestyle='--')
    ax.set_xlabel('omega (deg)', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # r_sum_sma
    rsumsma_interval = arviz.hdi(rsumsma_vals, hdi_prob=0.683)
    rsumsma_bounds = arviz.hdi(rsumsma_vals, hdi_prob=0.997)
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist(rsumsma_vals, bins=50, label='vary fit input')
    ax.plot([r_sum_sma, r_sum_sma], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([rsumsma_interval[0], rsumsma_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([rsumsma_interval[1], rsumsma_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([rsumsma_bounds[0], rsumsma_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([rsumsma_bounds[1], rsumsma_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('(r1+r2)/a', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # r_dif_sma
    rdifsma_interval = arviz.hdi(rdifsma_vals, hdi_prob=0.683)
    rdifsma_bounds = arviz.hdi(rdifsma_vals, hdi_prob=0.997)
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist(rdifsma_vals, bins=50, label='vary fit input')
    ax.plot([r_dif_sma, r_dif_sma], [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([rdifsma_interval[0], rdifsma_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([rdifsma_interval[1], rdifsma_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([rdifsma_bounds[0], rdifsma_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([rdifsma_bounds[1], rdifsma_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('|r1-r2|/a', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # r_ratio
    rratio_interval = arviz.hdi(rratio_vals, hdi_prob=0.683)
    rratio_bounds = arviz.hdi(rratio_vals, hdi_prob=0.997)
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist((rratio_vals), bins=50, label='vary fit input')
    ax.plot(([r_ratio, r_ratio]), [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([rratio_interval[0], rratio_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([rratio_interval[1], rratio_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([rratio_bounds[0], rratio_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([rratio_bounds[1], rratio_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('r_ratio', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # log(r_ratio)
    log_rratio_interval = arviz.hdi(np.log10(rratio_vals), hdi_prob=0.683)
    log_rratio_bounds = arviz.hdi(np.log10(rratio_vals), hdi_prob=0.997)
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist(np.log10(rratio_vals), bins=50, label='vary fit input')
    ax.plot(np.log10([r_ratio, r_ratio]), [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([log_rratio_interval[0], log_rratio_interval[0]], [0, np.max(hist[0])],
            c='tab:orange', label='hdi_prob=0.683')
    ax.plot([log_rratio_interval[1], log_rratio_interval[1]], [0, np.max(hist[0])],
            c='tab:orange')
    ax.plot([log_rratio_bounds[0], log_rratio_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([log_rratio_bounds[1], log_rratio_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('log(r_ratio)', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # sb_ratio
    sbratio_interval = arviz.hdi(sbratio_vals, hdi_prob=0.683)
    sbratio_bounds = arviz.hdi(sbratio_vals, hdi_prob=0.997)
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist((sbratio_vals), bins=50, label='vary fit input')
    ax.plot(([sb_ratio, sb_ratio]), [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([sbratio_interval[0], sbratio_interval[0]], [0, np.max(hist[0])], c='tab:orange',
            label='hdi_prob=0.683')
    ax.plot([sbratio_interval[1], sbratio_interval[1]], [0, np.max(hist[0])], c='tab:orange')
    ax.plot([sbratio_bounds[0], sbratio_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([sbratio_bounds[1], sbratio_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('sb_ratio', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # log(sb_ratio)
    log_sbratio_interval = arviz.hdi(np.log10(sbratio_vals), hdi_prob=0.683)
    log_sbratio_bounds = arviz.hdi(np.log10(sbratio_vals), hdi_prob=0.997)
    fig, ax = plt.subplots(figsize=(16, 9))
    hist = ax.hist(np.log10(sbratio_vals), bins=50, label='vary fit input')
    ax.plot(np.log10([sb_ratio, sb_ratio]), [0, np.max(hist[0])], c='tab:green', label='best fit value')
    ax.plot([log_sbratio_interval[0], log_sbratio_interval[0]], [0, np.max(hist[0])],
            c='tab:orange', label='hdi_prob=0.683')
    ax.plot([log_sbratio_interval[1], log_sbratio_interval[1]], [0, np.max(hist[0])],
            c='tab:orange')
    ax.plot([log_sbratio_bounds[0], log_sbratio_bounds[0]], [0, np.max(hist[0])], c='tab:grey', linestyle='--',
            label='hdi_prob=0.997')
    ax.plot([log_sbratio_bounds[1], log_sbratio_bounds[1]], [0, np.max(hist[0])], c='tab:grey', linestyle='--')
    ax.set_xlabel('log(sb_ratio)', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    return


def plot_corner_eclipse_parameters(timings_tau, depths, bottom_dur, t_1_vals, t_2_vals, tau_1_1_vals, tau_1_2_vals,
                                   tau_2_1_vals, tau_2_2_vals, d_1_vals, d_2_vals, bot_1_vals, bot_2_vals, e, w, i,
                                   phi_0, psi_0, r_sum_sma, r_dif_sma, r_ratio, sb_ratio, e_vals, w_vals, i_vals,
                                   phi0_vals, psi0_vals, rsumsma_vals, rdifsma_vals, rratio_vals, sbratio_vals,
                                   save_file=None, show=True):
    """Shows the corner plots resulting from the input distributions
    
    Note: produces several plots
    """
    t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2 = timings_tau
    d_1, d_2 = depths
    bot_1, bot_2 = bottom_dur[0], bottom_dur[1]
    cos_w = np.cos(w)
    sin_w = np.sin(w)
    if (abs(w/np.pi*180 - 180) > 80) & (abs(w/np.pi*180 - 180) < 100):
        w_vals_hist = np.copy(w_vals)
    else:
        w_vals_hist = np.copy(w_vals)
        mask = (np.sign((w/np.pi*180 - 180) * (w_vals/np.pi*180 - 180)) < 0)
        w_vals_hist[mask] = w_vals[mask] + np.sign(w/np.pi*180 - 180) * 2 * np.pi
    # input
    dist_data = np.column_stack((t_1_vals, t_2_vals, tau_1_1_vals, tau_1_2_vals, tau_2_1_vals, tau_2_2_vals,
                                 bot_1_vals, bot_2_vals))
    fig = corner.corner(dist_data, truths=(t_1, t_2, tau_1_1, tau_1_2, tau_2_1, tau_2_2, bot_1, bot_2),
                        labels=(r'$t_1$', r'$t_2$', r'$\tau_{1,1}$', r'$\tau_{1,2}$', r'$\tau_{2,1}$', r'$\tau_{2,2}$',
                                r'$width_{b,1}$', r'$width_{b,2}$'))
    fig.suptitle('Input distributions', fontsize=14)
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
    dist_data = np.column_stack((e_vals, w_vals_hist/np.pi*180, i_vals/np.pi*180, rsumsma_vals,
                                 rratio_vals, sbratio_vals))
    fig = corner.corner(dist_data, truths=(e, w/np.pi*180, i/np.pi*180, r_sum_sma, r_ratio, sb_ratio),
                        labels=('e', 'w (deg)', 'i (deg)', r'$\frac{r_1+r_2}{a}$', r'$\frac{r_2}{r_1}$',
                                r'$\frac{sb_2}{sb_1}$'))
    fig.suptitle('Output distributions', fontsize=14)
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
    # alternate parameterisation
    dist_data = np.column_stack((e_vals*np.cos(w_vals_hist), e_vals*np.sin(w_vals_hist), i_vals/np.pi*180,
                                 rsumsma_vals, np.log10(rratio_vals), np.log10(sbratio_vals)))
    fig = corner.corner(dist_data,
                        truths=(e*cos_w, e*sin_w, i/np.pi*180, r_sum_sma, np.log10(r_ratio), np.log10(sb_ratio)),
                        labels=(r'$e\cdot cos(w)$', r'$e\cdot sin(w)$', 'i (deg)', r'$\frac{r_1+r_2}{a}$',
                                r'$log10\left(\frac{r_2}{r_1}\right)$', r'$log10\left(\frac{sb_2}{sb_1}\right)$'))
    fig.suptitle('Output distributions (alternate parametrisation)', fontsize=14)
    if save_file is not None:
        if save_file.endswith('.png'):
            fig_save_file = save_file.replace('.png', '_out_alt.png')
        else:
            fig_save_file = save_file + '_out_alt.png'
        fig.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    # other parameters
    dist_data = np.column_stack((d_1_vals, d_2_vals, phi0_vals, psi0_vals, rdifsma_vals))
    fig = corner.corner(dist_data, truths=(d_1, d_2, phi_0, psi_0, r_dif_sma),
                        labels=(r'$depth_1$', r'$depth_2$', r'$phi_0$', r'$psi_0$', r'$\frac{|r_1-r_2|}{a}$'))
    fig.suptitle('Intermediate distributions', fontsize=14)
    if save_file is not None:
        if save_file.endswith('.png'):
            fig_save_file = save_file.replace('.png', '_other.png')
        else:
            fig_save_file = save_file + '_other.png'
        fig.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_ellc_fit(times, signal, p_orb, t_zero, timings, const, slope, f_n, a_n, ph_n, par_init, par_opt,
                     i_sectors, save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the determination
    of orbital parameters using both the eclipse timings and the ellc light curve
    models over two consecutive fits.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2 = timings
    # make the model times array, one full period plus the primary eclipse halves
    t_model = (times - t_zero) % p_orb
    ext_left = (t_model > p_orb + t_1_1)
    ext_right = (t_model < t_1_2)
    t_model = np.concatenate((t_model[ext_left] - p_orb, t_model, t_model[ext_right] + p_orb))
    sorter = np.argsort(t_model)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line + 1
    ecl_signal = np.concatenate((ecl_signal[ext_left], ecl_signal, ecl_signal[ext_right]))
    s_minmax = [np.min(ecl_signal), np.max(ecl_signal)]
    # unpack and define parameters
    e, w, i_rad, r_sum_sma, r_ratio, sb_ratio = par_init
    f_c, f_s = e**0.5 * np.cos(w), e**0.5 * np.sin(w)
    i = i_rad / np.pi * 180
    opt_e, opt_w, opt_i_rad, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio, offset = par_opt
    opt_f_c, opt_f_s = opt_e**0.5 * np.cos(opt_w), opt_e**0.5 * np.sin(opt_w)
    opt_i = opt_i_rad / np.pi * 180
    # make the ellc models
    model = tsfit.ellc_lc_simple(t_model, p_orb, 0, f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset)
    model1 = tsfit.ellc_lc_simple(t_model, p_orb, 0, opt_f_c, opt_f_s, opt_i, opt_r_sum_sma, opt_r_ratio,
                                  opt_sb_ratio, offset)
    # plot
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(t_model, ecl_signal, marker='.', label='eclipse signal')
    ax.plot(t_model[sorter], model[sorter], c='tab:orange', label='initial values from formulae')
    ax.plot(t_model[sorter], model1[sorter], c='tab:green', label='fit for f_c, f_s, i, r_sum, r_rat, sb_rat')
    ax.plot([t_1_1, t_1_1], s_minmax, '--', c='grey', label='eclipse edges')
    ax.plot([t_1_2, t_1_2], s_minmax, '--', c='grey')
    ax.plot([t_2_1, t_2_1], s_minmax, '--', c='grey')
    ax.plot([t_2_2, t_2_2], s_minmax, '--', c='grey')
    ax.set_xlabel(r'$(time - t_0) mod(P_{orb})$ (d)', fontsize=14)
    ax.set_ylabel('normalised flux', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_ellc_errors(times, signal, p_orb, t_zero, timings, const, slope, f_n, a_n, ph_n, i_sectors,
                        params, par_i, par_bounds, save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the determination
    of orbital parameters using both the eclipse timings and the ellc light curve
    models over three consecutive fits.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2 = timings
    # make the model times array, one full period plus the primary eclipse halves
    t_model = (times - t_zero) % p_orb
    ext_left = (t_model > p_orb + t_1_1)
    ext_right = (t_model < t_1_2)
    t_model = np.concatenate((t_model[ext_left] - p_orb, t_model, t_model[ext_right] + p_orb))
    sorter = np.argsort(t_model)
    # make the eclipse signal by subtracting the non-harmonics and the linear curve from the signal
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line + 1
    ecl_signal = np.concatenate((ecl_signal[ext_left], ecl_signal, ecl_signal[ext_right]))
    s_minmax = [np.min(ecl_signal), np.max(ecl_signal)]
    # unpack and define parameters
    f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset = params
    # make the ellc models
    model = tsfit.ellc_lc_simple(t_model, p_orb, 0, f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset)
    par_p = np.copy(params)
    par_n = np.copy(params)
    par_p[par_i] = par_bounds[1]
    par_n[par_i] = par_bounds[0]
    model_p = tsfit.ellc_lc_simple(t_model, p_orb, 0, par_p[0], par_p[1], par_p[2], par_p[3], par_p[4], par_p[5],
                                   offset)
    model_m = tsfit.ellc_lc_simple(t_model, p_orb, 0, par_n[0], par_n[1], par_n[2], par_n[3], par_n[4], par_n[5],
                                   offset)
    par_names = ['f_c', 'f_s', 'i', 'r_sum', 'r_ratio', 'sb_ratio']
    # plot
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(t_model, ecl_signal, marker='.', label='eclipse signal')
    ax.plot(t_model[sorter], model[sorter], c='tab:orange', label='best parameters')
    ax.fill_between(t_model[sorter], y1=model[sorter], y2=model_p[sorter], color='tab:orange', alpha=0.3,
                    label=f'upper bound {par_names[par_i]}')
    ax.fill_between(t_model[sorter], y1=model[sorter], y2=model_m[sorter], color='tab:purple', alpha=0.3,
                    label=f'lower bound {par_names[par_i]}')
    ax.plot([t_1_1, t_1_1], s_minmax, '--', c='grey', label=r'eclipse edges')
    ax.plot([t_1_2, t_1_2], s_minmax, '--', c='grey')
    ax.plot([t_2_1, t_2_1], s_minmax, '--', c='grey')
    ax.plot([t_2_2, t_2_2], s_minmax, '--', c='grey')
    ax.set_xlabel(r'$(time - t_0) mod(P_{orb})$ (d)', fontsize=14)
    ax.set_ylabel('normalised flux', fontsize=14)
    ax.set_title(f'{par_names[par_i]} = {params[par_i]:1.4f}, bounds: ({par_bounds[0]:1.4f}, {par_bounds[1]:1.4f})',
                 fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_corner_ellc_pars(parameters_1, parameters_2, distributions, save_file=None, show=True):
    """Corner plot of the distributions and the given 'truths' indicated
    using the parametrisation of ellc
    """
    e_vals, w_vals, i_vals, phi0_vals, psi0_vals, rsumsma_vals, rdifsma_vals, rratio_vals, sbratio_vals = distributions
    # transform some params
    e, w, i_rad, r_sum_sma, r_ratio, sb_ratio = parameters_1
    f_c, f_s = e**0.5 * np.cos(w), e**0.5 * np.sin(w)
    i = i_rad / np.pi * 180
    parameters_1 = [f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio]
    opt_e, opt_w, opt_i_rad, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio, offset = parameters_2
    opt_f_c, opt_f_s = opt_e**0.5 * np.cos(opt_w), opt_e**0.5 * np.sin(opt_w)
    opt_i = opt_i_rad / np.pi * 180
    parameters_2 = [opt_f_c, opt_f_s, opt_i, opt_r_sum_sma, opt_r_ratio, opt_sb_ratio]
    f_c_vals = np.sqrt(e_vals) * np.cos(w_vals)
    f_s_vals = np.sqrt(e_vals) * np.sin(w_vals)
    # stack dists and plot
    dist_data = np.column_stack((f_c_vals, f_s_vals, i_vals/np.pi*180, rsumsma_vals, rratio_vals, sbratio_vals))
    fig = corner.corner(dist_data, labels=('f_c', 'f_s', 'i (deg)', r'$\frac{r_1+r_2}{a}$', r'$\frac{r_2}{r_1}$',
                        r'$\frac{sb_2}{sb_1}$'), truth_color='tab:orange')
    corner.overplot_lines(fig, parameters_1, color='tab:blue')
    corner.overplot_points(fig, [parameters_1], marker='s', color='tab:blue')
    corner.overplot_lines(fig, parameters_2, color='tab:orange')
    corner.overplot_points(fig, [parameters_2], marker='s', color='tab:orange')
    fig.suptitle('Output distributions and ELLC fit outcome', fontsize=14)
    if save_file is not None:
        fig.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_pd_pulsation_analysis(times, signal, p_orb, f_n, a_n, noise_level, passed_nh, save_file=None, show=True):
    """Plot the periodogram with the output of the pulsation analysis."""
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    passed_nh_i = np.arange(len(f_n))[passed_nh]
    # make periodograms
    freqs, ampls = tsf.astropy_scargle(times, signal - np.mean(signal))
    snr_threshold = ut.signal_to_noise_threshold(len(signal))
    # plot
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(times[[0, -1]], [snr_threshold*noise_level, snr_threshold*noise_level], c='tab:grey', alpha=0.6,
            label=f'S/N threshold ({snr_threshold})')
    ax.plot(freqs, ampls, label='signal')
    for k in harmonics:
        ax.plot([f_n[k], f_n[k]], [0, a_n[k]], linestyle='--', c='tab:orange')
    for k in non_harm:
        if k in passed_nh_i:
            ax.plot([f_n[k], f_n[k]], [0, a_n[k]], linestyle='--', c='tab:green')
        else:
            ax.plot([f_n[k], f_n[k]], [0, a_n[k]], linestyle=':', c='tab:red')
    ax.plot([], [], linestyle='--', c='tab:orange', label='harmonics')
    ax.plot([], [], linestyle=':', c='tab:red', label='failed criteria')
    ax.plot([], [], linestyle='--', c='tab:green', label='passed criteria')
    plt.xlabel('frequency (1/d)', fontsize=14)
    plt.ylabel('amplitude', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_pulsation_analysis(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, passed_nh,
                               t_zero, const_r, f_n_r, a_n_r, ph_n_r, params_ellc, save_file=None, show=True):
    """Shows the separated harmonics in several ways"""
    f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset = params_ellc
    # make models
    model = tsf.linear_curve(times, const, slope, i_sectors)
    model += tsf.sum_sines(times, f_n, a_n, ph_n)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    model_h = tsf.sum_sines(times, f_n[harmonics], a_n[harmonics], ph_n[harmonics])
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_pnh = tsf.sum_sines(times, f_n[passed_nh], a_n[passed_nh], ph_n[passed_nh])
    # ellc models
    model_ellc_h = tsf.sum_sines(times, f_n_r, a_n_r, ph_n_r)
    model_ellc = tsfit.ellc_lc_simple(times, p_orb, t_zero, f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset)
    # plot the non-harmonics passing the criteria
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))
    ax[0].scatter(times, signal, marker='.', c='tab:blue', label='signal')
    ax[0].plot(times, model_line + model_h, marker='.', c='tab:orange', label='linear + harmonic model')
    ax[1].scatter(times, signal - model_h, marker='.', c='tab:blue', label='signal - harmonic model')
    ax[1].plot(times, model_line + model_pnh, marker='.', c='tab:orange', label='linear + passed non-harmonic model')
    ax[0].set_ylabel('residual/model', fontsize=14)
    ax[0].legend(fontsize=12)
    ax[1].set_ylabel('residual/model', fontsize=14)
    ax[1].set_xlabel('time (d)', fontsize=14)
    ax[1].legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        if save_file.endswith('.png'):
            fig_save_file = save_file.replace('.png', '_1.png')
        else:
            fig_save_file = save_file + '_1.png'
        plt.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    # plot the disentangled harmonics and ellc model
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))
    ax[0].scatter(times, signal - model_nh, marker='.', c='tab:blue', label='signal - non-harmonic model')
    ax[0].plot(times, model_line + model_ellc + const_r, marker='.', c='tab:orange',
               label='linear + ellc model + constant')
    ax[1].scatter(times, signal - model_nh - model_ellc, marker='.', c='tab:blue',
                  label='signal - non-harmonic - ellc model')
    ax[1].plot(times, model_line + model_ellc_h + const_r, marker='.', c='tab:orange',
               label='linear + disentangled harmonics + constant')
    ax[0].set_ylabel('residual/model', fontsize=14)
    ax[0].legend(fontsize=12)
    ax[1].set_ylabel('residual/model', fontsize=14)
    ax[1].set_xlabel('time (d)', fontsize=14)
    ax[1].legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        if save_file.endswith('.png'):
            fig_save_file = save_file.replace('.png', '_2.png')
        else:
            fig_save_file = save_file + '_2.png'
        plt.savefig(fig_save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_pd_ellc_harmonics(times, signal, p_orb, t_zero, const, slope, f_n, a_n, ph_n, noise_level,
                           const_r, f_n_r, a_n_r, passed_hr, params_ellc, i_sectors, save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the determination
    of orbital parameters using both the eclipse timings and the ellc light curve
    models over two consecutive fits.
    """
    # make the model times array, one full period plus the primary eclipse halves
    passed_hr_i = np.arange(len(f_n_r))[passed_hr]
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    # unpack and define parameters
    f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset = params_ellc
    # make the ellc model
    model_ellc = tsfit.ellc_lc_simple(times, p_orb, t_zero, f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset)
    resid_ellc = signal - model_nh - model_line - model_ellc - const_r
    # make periodograms
    freqs, ampls = tsf.astropy_scargle(times, resid_ellc)
    snr_threshold = ut.signal_to_noise_threshold(len(signal))
    # plot
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(freqs, ampls, label='residual')
    ax.plot(times[[0, -1]], [snr_threshold*noise_level, snr_threshold*noise_level], c='tab:grey', alpha=0.6,
            label=f'S/N threshold ({snr_threshold})')
    for k in range(len(f_n_r)):
        if k in passed_hr_i:
            ax.plot([f_n_r[k], f_n_r[k]], [0, a_n_r[k]], linestyle='--', c='tab:green')
        else:
            ax.plot([f_n_r[k], f_n_r[k]], [0, a_n_r[k]], linestyle='--', c='tab:orange')
    ax.plot([], [], linestyle='--', c='tab:orange', label='disentangled harmonics')
    plt.xlabel('frequency (1/d)', fontsize=14)
    plt.ylabel('amplitude', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_ellc_harmonics(times, signal, p_orb, t_zero, timings, const, slope, f_n, a_n, ph_n, i_sectors,
                           const_r, f_n_r, a_n_r, ph_n_r, params_ellc, save_file=None, show=True):
    """Shows an overview of the eclipses over one period with the determination
    of orbital parameters using both the eclipse timings and the ellc light curve
    models over two consecutive fits.
    """
    t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2 = timings
    # make the model times array, one full period plus the primary eclipse halves
    t_model = (times - t_zero) % p_orb
    ext_left = (t_model > p_orb + t_1_1)
    ext_right = (t_model < t_1_2)
    t_model = np.concatenate((t_model[ext_left] - p_orb, t_model, t_model[ext_right] + p_orb))
    sorter = np.argsort(t_model)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    model_nh = tsf.sum_sines(times, f_n[non_harm], a_n[non_harm], ph_n[non_harm])
    model_line = tsf.linear_curve(times, const, slope, i_sectors)
    ecl_signal = signal - model_nh - model_line
    ecl_signal = np.concatenate((ecl_signal[ext_left], ecl_signal, ecl_signal[ext_right]))
    s_minmax = [np.min(ecl_signal), np.max(ecl_signal)]
    # unpack and define parameters
    f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset = params_ellc
    # make the ellc model
    model_ellc = tsfit.ellc_lc_simple(t_model, p_orb, 0, f_c, f_s, i, r_sum_sma, r_ratio, sb_ratio, offset)
    model_ellc_h = tsf.sum_sines(t_model + t_zero, f_n_r, a_n_r, ph_n_r)
    # plot
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(t_model, ecl_signal, marker='.', label='signal - linear - non-harmonic model')
    ax.scatter(t_model[sorter], ecl_signal[sorter] - model_ellc_h[sorter], marker='.',
               label='signal - linear - non-harmonic - disentangled harmonic model')
    ax.plot(t_model[sorter], model_ellc[sorter] + const_r, c='tab:green', label='ellc model + consant')
    ax.plot(t_model[sorter], model_ellc_h[sorter], c='tab:red', label='disentangled harmonic model')
    ax.plot([t_1_1, t_1_1], s_minmax, '--', c='grey', label='eclipse edges')
    ax.plot([t_1_2, t_1_2], s_minmax, '--', c='grey')
    ax.plot([t_2_1, t_2_1], s_minmax, '--', c='grey')
    ax.plot([t_2_2, t_2_2], s_minmax, '--', c='grey')
    ax.set_xlabel(r'$(time - t_0) mod(P_{orb})$ (d)', fontsize=14)
    ax.set_ylabel('normalised flux', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def refine_subset_visual(times, signal, close_f, const, slope, f_n, a_n, ph_n, i_sectors, iteration, save_dir,
                         verbose=False):
    """Refine a subset of frequencies that are within the Rayleigh criterion of each other
    and visualise the process.
    
    Intended as a sub-loop within another extraction routine (see: extract_all_visual).
    """
    freq_res = 1.5 / np.ptp(times)  # frequency resolution
    n_sectors = len(i_sectors)
    n_f = len(f_n)
    # determine initial bic
    model = tsf.linear_curve(times, const, slope, i_sectors)  # the linear part of the model
    model += tsf.sum_sines(times, f_n, a_n, ph_n)  # the sinusoid part of the model
    resid = signal - model
    f_n_temp, a_n_temp, ph_n_temp = np.copy(f_n), np.copy(a_n), np.copy(ph_n)
    n_param = 2 * n_sectors + 3 * n_f
    bic_prev = np.inf
    bic = tsf.calc_bic(resid, n_param)
    # stop the loop when the BIC increases
    i = 0
    while (np.round(bic_prev - bic, 2) > 0):
        # last frequencies are accepted
        f_n, a_n, ph_n = np.copy(f_n_temp), np.copy(a_n_temp), np.copy(ph_n_temp)
        bic_prev = bic
        if verbose:
            print(f'Refining iteration {i}, {n_f} frequencies, BIC= {bic:1.2f}')
        # save parameters
        with h5py.File(os.path.join(save_dir, f'vis_step_1_iteration_{iteration}_refine_{i}.hdf5'), 'w-') as file:
            file.attrs['date_time'] = str(datetime.datetime.now())
            file.attrs['iteration'] = iteration
            file.attrs['subiteration'] = i
            file.attrs['n_param'] = n_param  # number of free parameters
            file.attrs['bic'] = bic  # Bayesian Information Criterion of the residuals
            file.create_dataset('const', data=const)
            file.create_dataset('slope', data=slope)
            file.create_dataset('f_n', data=f_n_temp)
            file.create_dataset('a_n', data=a_n_temp)
            file.create_dataset('ph_n', data=ph_n_temp)
        # remove each frequency one at a time to re-extract them
        for j in close_f:
            model = tsf.linear_curve(times, const, slope, i_sectors)  # the linear part of the model
            model += tsf.sum_sines(times, np.delete(f_n_temp, j), np.delete(a_n_temp, j),
                                   np.delete(ph_n_temp, j))  # the sinusoid part of the model
            resid = signal - model
            f_j, a_j, ph_j = tsf.extract_single(times, resid, f0=f_n_temp[j] - freq_res, fn=f_n_temp[j] + freq_res,
                                                verbose=verbose)
            f_n_temp[j], a_n_temp[j], ph_n_temp[j] = f_j, a_j, ph_j
        # as a last model-refining step, redetermine the constant and slope
        model = tsf.sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
        const, slope = tsf.linear_slope(times, signal - model, i_sectors)
        model += tsf.linear_curve(times, const, slope, i_sectors)  # the linear part of the model
        # now subtract all from the signal and calculate BIC before moving to the next iteration
        resid = signal - model
        bic = tsf.calc_bic(resid, n_param)
        i += 1
    if verbose:
        print(f'Refining terminated. Iteration {i} not included with BIC= {bic:1.2f}, '
              f'delta-BIC= {bic_prev - bic:1.2f}')
    # redo the constant and slope without the last iteration of changes
    resid = signal - tsf.sum_sines(times, f_n, a_n, ph_n)
    const, slope = tsf.linear_slope(times, resid, i_sectors)
    return const, slope, f_n, a_n, ph_n


def extract_all_visual(times, signal, i_sectors, save_dir, verbose=True):
    """Extract all the frequencies from a periodic signal and visualise the process.

    For a description of the algorithm, see timeseries_functions.extract_all()
    Parameters are saved at each step, so that after completion an animated
    visualisation can be made.
    """
    times -= times[0]  # shift reference time to times[0]
    freq_res = 1.5 / np.ptp(times)  # frequency resolution
    n_sectors = len(i_sectors)
    # constant term (or y-intercept) and slope
    const, slope = tsf.linear_slope(times, signal, i_sectors)
    resid = signal - tsf.linear_curve(times, const, slope, i_sectors)
    f_n_temp, a_n_temp, ph_n_temp = np.array([[], [], []])
    f_n, a_n, ph_n = np.copy(f_n_temp), np.copy(a_n_temp), np.copy(ph_n_temp)
    n_param = 2 * n_sectors
    bic_prev = np.inf  # initialise previous BIC to infinity
    bic = tsf.calc_bic(resid, n_param)  # initialise current BIC to the mean (and slope) subtracted signal
    # stop the loop when the BIC decreases by less than 2 (or increases)
    i = 0
    while (bic_prev - bic > 2):
        # last frequency is accepted
        f_n, a_n, ph_n = np.copy(f_n_temp), np.copy(a_n_temp), np.copy(ph_n_temp)
        bic_prev = bic
        if verbose:
            print(f'Iteration {i}, {len(f_n)} frequencies, BIC= {bic:1.2f}')
        # save parameters
        with h5py.File(os.path.join(save_dir, f'vis_step_1_iteration_{i}.hdf5'), 'w-') as file:
            file.attrs['date_time'] = str(datetime.datetime.now())
            file.attrs['iteration'] = i
            file.attrs['n_param'] = n_param  # number of free parameters
            file.attrs['bic'] = bic  # Bayesian Information Criterion of the residuals
            file.create_dataset('const', data=const)
            file.create_dataset('slope', data=slope)
            file.create_dataset('f_n', data=f_n_temp)
            file.create_dataset('a_n', data=a_n_temp)
            file.create_dataset('ph_n', data=ph_n_temp)
        # attempt to extract the next frequency
        f_i, a_i, ph_i = tsf.extract_single(times, resid, verbose=verbose)
        f_n_temp, a_n_temp, ph_n_temp = np.append(f_n_temp, f_i), np.append(a_n_temp, a_i), np.append(ph_n_temp, ph_i)
        # now iterate over close frequencies (around f_i) a number of times to improve them
        close_f = af.f_within_rayleigh(i, f_n_temp, freq_res)
        if (i > 0) & (len(close_f) > 1):
            const, slope, f_n_temp, a_n_temp, ph_n_temp = refine_subset_visual(times, signal, close_f, const, slope,
                                                                               f_n_temp, a_n_temp, ph_n_temp, i_sectors,
                                                                               i, save_dir, verbose=verbose)
        # as a last model-refining step, redetermine the constant and slope
        model = tsf.sum_sines(times, f_n_temp, a_n_temp, ph_n_temp)  # the sinusoid part of the model
        const, slope = tsf.linear_slope(times, signal - model, i_sectors)
        model += tsf.linear_curve(times, const, slope, i_sectors)  # the linear part of the model
        # now subtract all from the signal and calculate BIC before moving to the next iteration
        resid = signal - model
        n_param = 2 * n_sectors + 3 * len(f_n_temp)
        bic = tsf.calc_bic(resid, n_param)
        i += 1
    if verbose:
        print(f'Extraction terminated. Iteration {i} not included with BIC= {bic:1.2f}, '
              f'delta-BIC= {bic_prev - bic:1.2f}')
    # redo the constant and slope without the last iteration frequencies
    resid = signal - tsf.sum_sines(times, f_n, a_n, ph_n)
    const, slope = tsf.linear_slope(times, resid, i_sectors)
    return const, slope, f_n, a_n, ph_n


def visualise_frequency_analysis(tic, times, signal, p_orb, i_sectors, i_half_s, save_dir, verbose=False):
    """Visualise the whole frequency analysis process."""
    # make the directory
    sub_dir = f'tic_{tic}_analysis'
    file_id = f'TIC {tic}'
    if (not os.path.isdir(os.path.join(save_dir, sub_dir))):
        os.mkdir(os.path.join(save_dir, sub_dir))  # create the subdir
    save_dir = os.path.join(save_dir, sub_dir)  # this is what we will use from now on
    # start by saving the light curve itself (this is for future reproducability)
    with h5py.File(os.path.join(save_dir, f'vis_step_0_light_curve.hdf5'), 'w-') as file:
        file.attrs['identifier'] = file_id
        file.attrs['date_time'] = str(datetime.datetime.now())
        file.create_dataset('times', data=times)
        file.create_dataset('signal', data=signal)
        file.create_dataset('i_sectors', data=i_sectors)
        file.create_dataset('i_half_s', data=i_half_s)
    # now do the extraction
    out = extract_all_visual(times, signal, i_half_s, save_dir, verbose=verbose)
    
    freq_res = 1.5 / np.ptp(times)
    # make the images (step 1)
    step_1_files = []
    for file_name in os.listdir(save_dir):
        if fnmatch.fnmatch(file_name, 'vis_step_1*.hdf5'):
            step_1_files.append(file_name)
    for file_name in step_1_files:
        with h5py.File(os.path.join(save_dir, file_name), 'r') as file:
            iteration = file.attrs['iteration']
            # stats
            n_param = file.attrs['n_param']
            bic = file.attrs['bic']
            # main results and errors
            const = np.copy(file['const'])
            slope = np.copy(file['slope'])
            f_n = np.copy(file['f_n'])
            a_n = np.copy(file['a_n'])
            ph_n = np.copy(file['ph_n'])
            if 'refine' in file_name:
                sub_iter = file.attrs['subiteration']
                title = f'Iteration {iteration}, refinement {sub_iter}'
                f_name = os.path.join(save_dir, f'vis_step_1_iteration_{iteration}_refine_{sub_iter}.png')
                f_close = af.f_within_rayleigh(len(f_n) - 1, f_n, freq_res)
                xlim = [np.min(f_n[f_close]) - freq_res, np.max(f_n[f_close]) + freq_res]
            else:
                title = f'Iteration {iteration}'
                f_name = os.path.join(save_dir, f'vis_step_1_iteration_{iteration}.png')
                xlim = None
        plot_pd_single_output(times, signal, const, slope, f_n, a_n, ph_n, n_param, bic, i_half_s, title,
                              zoom=xlim, annotate=False, save_file=f_name, show=False)
    return
