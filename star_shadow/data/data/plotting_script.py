"""Script for making some of the plots in IJspeert et al. 2024"""

import os
import fnmatch
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import star_shadow as sts

## SYNTHETIC TESTS
syn_dir = '~/data'
all_files = []
for root, dirs, files in os.walk(syn_dir):
    for file_i in files:
        if fnmatch.fnmatch(file_i, 'sim_[0-9][0-9][0-9]_lc.dat'):
            all_files.append(os.path.join(root, file_i))
all_files = np.sort(all_files)

# plotting all per case diagnostic plots in series
for file in all_files:
    target_id = os.path.splitext(os.path.basename(file))[0]
    save_dir = os.path.dirname(file)
    # load the light curve (this needs to be different for TESS files)
    times, signal, signal_err = np.loadtxt(file, usecols=(0, 1, 2), unpack=True)
    i_half_s = np.array([[0, len(times)]])
    sts.ut.sequential_plotting(times, signal, signal_err, i_half_s, target_id, save_dir, save_dir=save_dir, show=False)

# collect results in a summary file (run analysis to get the individual case results)
summary_file = os.path.join(os.path.dirname(all_files[0]), 'sim_000_lc_analysis', 'sim_000_lc_analysis_summary.csv')
hdr = np.loadtxt(summary_file, usecols=(0), delimiter=',', unpack=True, dtype=str)
obs_par_dtype = np.dtype([('id', '<U20'), ('stage', '<U20')] + [(name, float) for name in hdr[2:]])
obs_par = np.ones(len(all_files), dtype=obs_par_dtype)
for k, file in enumerate(all_files):
    target_id = os.path.splitext(os.path.basename(file))[0]
    data_dir = os.path.join(os.path.dirname(file), f'{target_id}_analysis')
    summary_file = os.path.join(data_dir, f'{target_id}_analysis_summary.csv')
    if os.path.isfile(summary_file):
        obs_par[k] = tuple(np.loadtxt(summary_file, usecols=(1), delimiter=',', unpack=True, dtype=str))
    else:
        print(summary_file)

# load parameter files
obs_par = pd.read_csv(syn_dir + '/test_results.csv')
true_par = pd.read_csv(syn_dir + '/sample_parameters_v02plus2.csv', index_col=0)

# transform result data
undetectable = [1, 2, 4, 5, 17, 21, 30, 32, 36, 37, 39, 46, 54, 59, 61, 65, 69, 70, 73, 80, 81, 83, 85, 90, 93, 95]
undetectable_sec = [3, 11, 13, 15, 22, 31, 33, 35, 38, 44, 45, 50, 74, 76, 79, 91]
# number of cycles
cycles = obs_par['t_tot'] / true_par['period']
sorter_c = np.argsort(cycles)[::-1]
# periods and bad datapoints
p_measure_1 = (obs_par['period'] - true_par['period']) / true_par['period']
p_measure_2 = (obs_par['period'] - true_par['period']) / obs_par['p_err']
p_error_1 = obs_par['p_err'] / true_par['period']
p_hdi_1 = obs_par['p_err_l'] / true_par['period']
p_hdi_2 = obs_par['p_err_u'] / true_par['period']
bad_p = (obs_par['period'][sorter_c] == -1) | (obs_par['p_err'][sorter_c] == -1)
finished = ((obs_par['stage'].astype(int) == 10) | (obs_par['stage'].astype(int) == 9)
            | (obs_par['stage'].astype(int) == 8))
p_good = np.array([case not in undetectable for case in range(100)])
ps_good = np.array([case not in undetectable + undetectable_sec for case in range(100)])
fin_good = ps_good & finished & (np.abs(p_measure_1) < 0.01)
# sort by primary depth (plot depths and levels)
max_depth = np.max((obs_par['depth_1'], obs_par['depth_2']), axis=0)
max_depth = np.max((true_par['primary_depth'], true_par['secondary_depth']), axis=0)
min_depth = np.min((obs_par['depth_1'], obs_par['depth_2']), axis=0)
deeper_sec = (obs_par['depth_2'] > obs_par['depth_1'])
sorter_dmax = np.argsort(max_depth)[::-1]
sorter_dmin = np.argsort(min_depth)[::-1]
max_ratio = np.max((obs_par['ratio_3_1'], obs_par['ratio_3_2']), axis=0)
min_ratio = np.min((obs_par['ratio_3_1'], obs_par['ratio_3_2']), axis=0)
sorter_rmax = np.argsort(max_ratio)[::-1]
sorter_rmin = np.argsort(min_ratio)[::-1]
harm_resid = obs_par['std_4'] / obs_par['std_1']
# period and period uncertainty, ordered by primary depth
bad_p_2 = (obs_par['period'][sorter_dmax] == -1) | (obs_par['p_err'][sorter_dmax] == -1)
# absolute differences and errors
# ecosw
sign_flip = [7, 23, 28, 56, 60, 89, 92, 98]  # these have prim and sec reversed (not by their mistake)
ecosw_form = obs_par['ecosw_form']
ecosw_phys = obs_par['ecosw_phys']
ecosw_form.loc[sign_flip] = -1 * obs_par['ecosw_form']  # flip sign
ecosw_phys.loc[sign_flip] = -1 * obs_par['ecosw_phys']  # flip sign
ecosw_measure_1 = (ecosw_form - (true_par['ecc'] * np.cos(true_par['omega'])))
ecosw_measure_2 = (ecosw_phys - (true_par['ecc'] * np.cos(true_par['omega'])))
ecosw_err_1 = np.vstack((obs_par['ecosw_low'], obs_par['ecosw_upp']))
ecosw_err_2 = np.vstack((obs_par['ecosw_err_l'], obs_par['ecosw_err_u']))
err_side_1 = np.clip(np.sign(-ecosw_measure_1), 0, 1).astype(int)
err_side_2 = np.clip(np.sign(-ecosw_measure_2), 0, 1).astype(int)
ecosw_measure_4 = ecosw_measure_1 / ecosw_err_1[err_side_1, np.arange(100)]
ecosw_measure_5 = ecosw_measure_2 / ecosw_err_1[err_side_2, np.arange(100)]
# esinw
esinw_form = obs_par['esinw_form']
esinw_phys = obs_par['esinw_phys']
esinw_form.loc[sign_flip] = -1 * obs_par['esinw_form']  # flip sign
esinw_phys.loc[sign_flip] = -1 * obs_par['esinw_phys']  # flip sign
esinw_measure_1 = (esinw_form - (true_par['ecc'] * np.sin(true_par['omega'])))
esinw_measure_2 = (esinw_phys - (true_par['ecc'] * np.sin(true_par['omega'])))
esinw_err_1 = np.vstack((obs_par['esinw_low'], obs_par['esinw_upp']))
esinw_err_2 = np.vstack((obs_par['esinw_err_l'], obs_par['esinw_err_u']))
err_side_1 = np.clip(np.sign(-esinw_measure_1), 0, 1).astype(int)
err_side_2 = np.clip(np.sign(-esinw_measure_2), 0, 1).astype(int)
esinw_measure_4 = esinw_measure_1 / esinw_err_1[err_side_1, np.arange(100)]
esinw_measure_5 = esinw_measure_2 / esinw_err_1[err_side_2, np.arange(100)]
# cosi
cosi_measure_1 = (obs_par['cosi_form'] - np.cos(true_par['incl']/180*np.pi))
cosi_measure_2 = (obs_par['cosi_phys'] - np.cos(true_par['incl']/180*np.pi))
cosi_err_1 = np.vstack((obs_par['cosi_low'], obs_par['cosi_upp']))
cosi_err_2 = np.vstack((obs_par['cosi_err_l'], obs_par['cosi_err_u']))
err_side_1 = np.clip(np.sign(-cosi_measure_1), 0, 1).astype(int)
err_side_2 = np.clip(np.sign(-cosi_measure_2), 0, 1).astype(int)
cosi_measure_4 = cosi_measure_1 / cosi_err_1[err_side_1, np.arange(100)]
cosi_measure_5 = cosi_measure_2 / cosi_err_1[err_side_2, np.arange(100)]
# phi_0
phi_0_true = sts.af.phi_0_from_r_sum_sma(true_par['ecc'].to_numpy(), true_par['incl'].to_numpy()/180*np.pi,  true_par['r_sum'].to_numpy())
phi_0_true[np.isnan(phi_0_true)] = 0
phi_0_measure_1 = (obs_par['phi_0_form'] - phi_0_true)
phi_0_measure_2 = (obs_par['phi_0_phys'] - phi_0_true)
phi_0_err_1 = np.vstack((obs_par['phi_0_low'], obs_par['phi_0_upp']))
phi_0_err_2 = np.vstack((obs_par['phi_0_err_l'], obs_par['phi_0_err_u']))
err_side_1 = np.clip(np.sign(-phi_0_measure_1), 0, 1).astype(int)
err_side_2 = np.clip(np.sign(-phi_0_measure_2), 0, 1).astype(int)
phi_0_measure_4 = phi_0_measure_1 / phi_0_err_1[err_side_1, np.arange(100)]
phi_0_measure_5 = phi_0_measure_2 / phi_0_err_1[err_side_2, np.arange(100)]
# log r_rat
log_rr_true = np.log10(true_par['r_rat'])
log_rr_form = obs_par['log_rr_form']
log_rr_phys = obs_par['log_rr_phys']
log_rr_form.loc[sign_flip] = -1 * obs_par['log_rr_form']  # flip sign
log_rr_phys.loc[sign_flip] = -1 * obs_par['log_rr_phys']  # flip sign
log_rr_measure_1 = (log_rr_form - log_rr_true)
log_rr_measure_2 = (log_rr_phys - log_rr_true)
log_rr_err_1 = np.vstack((obs_par['log_rr_low'], obs_par['log_rr_upp']))
log_rr_err_2 = np.vstack((obs_par['log_rr_err_l'], obs_par['log_rr_err_u']))
err_side_1 = np.clip(np.sign(-log_rr_measure_1), 0, 1).astype(int)
err_side_2 = np.clip(np.sign(-log_rr_measure_2), 0, 1).astype(int)
log_rr_measure_4 = log_rr_measure_1 / log_rr_err_1[err_side_1, np.arange(100)]
log_rr_measure_5 = log_rr_measure_2 / log_rr_err_1[err_side_2, np.arange(100)]
# log sb_rat
log_sb_true = np.log10(true_par['Sb'])
log_sb_form = obs_par['log_sb_form']
log_sb_phys = obs_par['log_sb_phys']
log_sb_form.loc[sign_flip] = -1 * obs_par['log_sb_form']  # flip sign
log_sb_phys.loc[sign_flip] = -1 * obs_par['log_sb_phys']  # flip sign
log_sb_measure_1 = (log_sb_form - log_sb_true)
log_sb_measure_2 = (log_sb_phys - log_sb_true)
log_sb_err_1 = np.vstack((obs_par['log_sb_low'], obs_par['log_sb_upp']))
log_sb_err_2 = np.vstack((obs_par['log_sb_err_l'], obs_par['log_sb_err_u']))
err_side_1 = np.clip(np.sign(-log_sb_measure_1), 0, 1).astype(int)
err_side_2 = np.clip(np.sign(-log_sb_measure_2), 0, 1).astype(int)
log_sb_measure_4 = log_sb_measure_1 / log_sb_err_1[err_side_1, np.arange(100)]
log_sb_measure_5 = log_sb_measure_2 / log_sb_err_1[err_side_2, np.arange(100)]
# eccentricity
e_measure_1 = (obs_par['e_form'] - true_par['ecc'])
e_measure_2 = (obs_par['e_phys'] - true_par['ecc'])
e_err_1 = np.vstack((obs_par['e_low'], obs_par['e_upp']))
e_err_2 = np.vstack((obs_par['e_err_l'], obs_par['e_err_u']))
err_side_1 = np.clip(np.sign(-e_measure_1), 0, 1).astype(int)
err_side_2 = np.clip(np.sign(-e_measure_2), 0, 1).astype(int)
e_measure_4 = e_measure_1 / e_err_1[err_side_1, np.arange(100)]
e_measure_5 = e_measure_2 / e_err_1[err_side_2, np.arange(100)]
# omega
w_form = obs_par['w_form']
w_phys = obs_par['w_phys']
w_form.loc[sign_flip] = (obs_par['w_form'].loc[sign_flip] - np.pi) % (2 * np.pi)
w_phys.loc[sign_flip] = (obs_par['w_phys'].loc[sign_flip] - np.pi) % (2 * np.pi)
w_measure_1 = (w_form - (true_par['omega'] % (2 * np.pi))) % (2 * np.pi)
w_measure_1[w_measure_1 > np.pi] = w_measure_1 - 2 * np.pi
w_measure_2 = (w_phys - (true_par['omega'] % (2 * np.pi))) % (2 * np.pi)
w_measure_2[w_measure_2 > np.pi] = w_measure_2 - 2 * np.pi
w_err_1 = np.vstack((obs_par['w_low'], obs_par['w_upp']))
w_err_2 = np.vstack((np.max([w_err_1[0], obs_par['w_sig']], axis=0), np.max([w_err_1[1], obs_par['w_sig']], axis=0)))
err_side_1 = np.clip(np.sign(-w_measure_1), 0, 1).astype(int)
err_side_2 = np.clip(np.sign(-w_measure_2), 0, 1).astype(int)
w_measure_4 = w_measure_1 / w_err_2[err_side_1, np.arange(100)]
w_measure_5 = w_measure_2 / w_err_2[err_side_2, np.arange(100)]
# inclination
i_measure_1 = (obs_par['i_form'] - (true_par['incl']/180*np.pi))
i_measure_2 = (obs_par['i_phys'] - (true_par['incl']/180*np.pi))
i_err_1 = np.vstack((obs_par['i_low'], obs_par['i_upp']))
# i_err_2 = np.vstack((obs_par['i_err_l'], obs_par['i_err_r']))
err_side_1 = np.clip(np.sign(-i_measure_1), 0, 1).astype(int)
err_side_2 = np.clip(np.sign(-i_measure_2), 0, 1).astype(int)
i_measure_4 = i_measure_1 / i_err_1[err_side_1, np.arange(100)]
i_measure_5 = i_measure_2 / i_err_1[err_side_2, np.arange(100)]
# r_sum
obs_par['r_sum_form'][np.isnan(obs_par['r_sum_form'])] = -1
obs_par['r_sum_phys'][np.isnan(obs_par['r_sum_phys'])] = -1
r_sum_measure_1 = (obs_par['r_sum_form'] - true_par['r_sum'])
r_sum_measure_2 = (obs_par['r_sum_phys'] - true_par['r_sum'])
r_sum_err = np.vstack((obs_par['r_sum_low'], obs_par['r_sum_upp']))
err_side_1 = np.clip(np.sign(-r_sum_measure_1), 0, 1).astype(int)
err_side_2 = np.clip(np.sign(-r_sum_measure_2), 0, 1).astype(int)
r_sum_measure_4 = r_sum_measure_1 / r_sum_err[err_side_1, np.arange(100)]
r_sum_measure_5 = r_sum_measure_2 / r_sum_err[err_side_2, np.arange(100)]
# r_rat
r_rat_form = obs_par['r_rat_form']
r_rat_phys = obs_par['r_rat_phys']
r_rat_form.loc[sign_flip] = 1 / obs_par['r_rat_form'].loc[sign_flip]  # flip fraction
r_rat_phys.loc[sign_flip] = 1 / obs_par['r_rat_phys'].loc[sign_flip]  # flip fraction
r_rat_measure_1 = (r_rat_form - true_par['r_rat'])
r_rat_measure_2 = (r_rat_phys - true_par['r_rat'])
r_rat_err = np.vstack((obs_par['r_rat_low'], obs_par['r_rat_upp']))  # error bars are changed as well:
err_side_1 = np.clip(np.sign(-r_rat_measure_1), 0, 1).astype(int)
err_side_2 = np.clip(np.sign(-r_rat_measure_2), 0, 1).astype(int)
r_rat_measure_4 = r_rat_measure_1 / r_rat_err[err_side_1, np.arange(100)]
r_rat_measure_5 = r_rat_measure_2 / r_rat_err[err_side_2, np.arange(100)]
# sb_rat
sb_rat_form = obs_par['sb_rat_form']
sb_rat_phys = obs_par['sb_rat_phys']
sb_rat_form.loc[sign_flip] = 1 / obs_par['sb_rat_form'].loc[sign_flip]  # flip fraction
sb_rat_phys.loc[sign_flip] = 1 / obs_par['sb_rat_phys'].loc[sign_flip]  # flip fraction
sb_rat_measure_1 = (sb_rat_form - true_par['Sb'])
sb_rat_measure_2 = (sb_rat_phys - true_par['Sb'])
sb_rat_err = np.vstack((obs_par['sb_rat_low'], obs_par['sb_rat_upp']))  # error bars are changed as well:
err_side_1 = np.clip(np.sign(-sb_rat_measure_1), 0, 1).astype(int)
err_side_2 = np.clip(np.sign(-sb_rat_measure_2), 0, 1).astype(int)
sb_rat_measure_4 = sb_rat_measure_1 / sb_rat_err[err_side_1, np.arange(100)]
sb_rat_measure_5 = sb_rat_measure_2 / sb_rat_err[err_side_2, np.arange(100)]
# period and period uncertainty, ordered by number of cycles
fix_range = 0.01
data_range = np.array([np.min(p_measure_1), np.max(p_measure_1)])
cycles_b2 = true_par.index[cycles[sorter_c] < 2][0] - 0.5  # sorted index for cycles below 3
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(true_par.index, np.zeros(100), c='tab:grey', alpha=0.8)
ax.errorbar(true_par.index, p_measure_1[sorter_c], yerr=p_error_1[sorter_c],
            capsize=2, marker='.', c='tab:blue', linestyle='none')
ax.errorbar(true_par.index[bad_p], p_measure_1[sorter_c][bad_p], yerr=p_error_1[sorter_c][bad_p],
            capsize=2, marker='.', c='tab:red', linestyle='none')
for i in true_par.index:
    p = true_par.index[i == sorter_c][0]
    if i in undetectable:
        ax.fill_between([p - 0.5, p + 0.5], data_range[[0, 0]], color='tab:grey', alpha=0.25, linewidth=0.0, hatch='/')
        ax.fill_between([p - 0.5, p + 0.5], data_range[[1, 1]], color='tab:grey', alpha=0.25, linewidth=0.0, hatch='/')
    if i in undetectable_sec:
        ax.fill_between([p - 0.5, p + 0.5], data_range[[0, 0]], color='tab:grey', alpha=0.25, linewidth=0.0, hatch='')
        ax.fill_between([p - 0.5, p + 0.5], data_range[[1, 1]], color='tab:grey', alpha=0.25, linewidth=0.0, hatch='')
    n = obs_par['stage'][i]
    # ax.annotate(f'{i}, {n}', (p, p_measure_1[i]), alpha=0.6)
ax.fill_between([cycles_b2, 99.5], data_range[[0, 0]], color='tab:orange', alpha=0.25, linewidth=0.0)
ax.fill_between([cycles_b2, 99.5], data_range[[1, 1]], color='tab:orange', alpha=0.25, linewidth=0.0)
ax.fill_between([], [], color='tab:grey', alpha=0.25, linewidth=0.0, label='no eclipses visible', hatch='/')
ax.fill_between([], [], color='tab:grey', alpha=0.25, linewidth=0.0, label='no secondary visible')
ax.fill_between([], [], color='tab:orange', alpha=0.25, linewidth=0.0, label='n<2 cycles')
ax.set_ylim(-fix_range, fix_range)
ax.set_xlabel('test case (sorted by number of cycles)', fontsize=14)
ax.set_ylabel(r'$\frac{P_{measured} - P_{input}}{P_{input}}$', fontsize=20)
ax2 = ax.twinx()
ax2.plot(true_par.index, cycles[sorter_c], marker='.', c='tab:grey', alpha=0.6)
ax2.set_ylim(-310, 310)
ax2.set_ylabel('Number of cycles', fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
# period and period uncertainty, ordered by primary depth
fix_range = 0.01
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(true_par.index, np.zeros(100), c='tab:grey', alpha=0.8)
ax.errorbar(true_par.index, p_measure_1[sorter_dmax], yerr=p_error_1[sorter_dmax],
            capsize=2, marker='.', c='tab:blue', linestyle='none')
ax.errorbar(true_par.index[bad_p_2], p_measure_1[sorter_dmax][bad_p_2], yerr=p_error_1[sorter_dmax][bad_p_2],
            capsize=2, marker='.', c='tab:red', linestyle='none')
for i in true_par.index:
    p = true_par.index[i == sorter_dmax][0]
    if i in undetectable:
        ax.fill_between([p - 0.5, p + 0.5], data_range[[0, 0]], color='tab:grey', alpha=0.25, linewidth=0.0, hatch='/')
        ax.fill_between([p - 0.5, p + 0.5], data_range[[1, 1]], color='tab:grey', alpha=0.25, linewidth=0.0, hatch='/')
    if i in undetectable_sec:
        ax.fill_between([p - 0.5, p + 0.5], data_range[[0, 0]], color='tab:grey', alpha=0.25, linewidth=0.0, hatch='')
        ax.fill_between([p - 0.5, p + 0.5], data_range[[1, 1]], color='tab:grey', alpha=0.25, linewidth=0.0, hatch='')
    if (cycles[sorter_dmax][i] < 2):
        ax.fill_between([p - 0.5, p + 0.5], data_range[[0, 0]], color='tab:orange', alpha=0.25, linewidth=0.0)
        ax.fill_between([p - 0.5, p + 0.5], data_range[[1, 1]], color='tab:orange', alpha=0.25, linewidth=0.0)
    # ax.annotate(f'{i}', (p, p_measure_1[i]), alpha=0.6)
ax.fill_between([], [], color='tab:grey', alpha=0.25, linewidth=0.0, label='no eclipses visible', hatch='/')
ax.fill_between([], [], color='tab:grey', alpha=0.25, linewidth=0.0, label='no secondary visible')
ax.fill_between([], [], color='tab:orange', alpha=0.25, linewidth=0.0, label='n<2 cycles')
ax.set_ylim(-fix_range, fix_range)
ax.set_xlabel('test case (sorted by eclipse depth)', fontsize=14)
ax.set_ylabel(r'$\frac{P_{measured} - P_{input}}{P_{input}}$', fontsize=20)
ax2 = ax.twinx()
ax2.plot(true_par.index, max_depth[sorter_dmax], marker='.', c='tab:grey', alpha=0.6)
ax2.set_ylim(-0.5, 0.5)
ax2.set_ylabel('Primary eclipse depth', fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
# e and e_err vs secondary depth
fix_range = 0.23
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax[0].scatter(min_depth[fin_good], true_par['ecc'][fin_good], marker='.', c='tab:grey', alpha=0.8, label='input')
ax[0].scatter(min_depth[fin_good], obs_par['e_phys'][fin_good], marker='.', c='tab:blue', label='eclipse model')
for x_par, y_par, true_y in zip(min_depth[fin_good], obs_par['e_phys'][fin_good], true_par['ecc'][fin_good]):
    ax[0].plot([x_par, x_par], [y_par, true_y], ':', c='tab:gray')
ax[0].set_ylim(-0.1, 1.1)
ax[0].set_ylabel('eccentricity', fontsize=14)
ax[0].legend()
ax[1].plot([0, np.max(min_depth[fin_good])], [0, 0], c='tab:grey', alpha=0.8)
ax[1].errorbar(min_depth[fin_good], e_measure_2[fin_good],
               yerr=np.vstack((e_err_1[0][fin_good], e_err_1[1][fin_good])),
               capsize=2, marker='.', c='tab:blue', linestyle='none', label='eclipse model')
# for i in true_par.index:
#     if fin_good[i]:
#         ax[1].annotate(f'{i}', (min_depth[i], e_measure_2[i]), alpha=0.6)
ax[1].set_ylim(-fix_range, fix_range)
ax[1].set_xlabel('secondary eclipse depth', fontsize=14)
ax[1].set_ylabel('$e_{measured} - e_{input}$', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()
# ecosw
fix_range = 0.09
true_ecosw = true_par['ecc'] * np.cos(true_par['omega'])
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))
ax[0].scatter(min_depth[fin_good], true_ecosw[fin_good], marker='.', c='tab:grey', alpha=0.8, label='input')
ax[0].scatter(min_depth[fin_good], ecosw_phys[fin_good], marker='.', c='tab:blue', label='eclipse model')
for x_par, y_par, true_y in zip(min_depth[fin_good], ecosw_phys[fin_good], true_ecosw[fin_good]):
    ax[0].plot([x_par, x_par], [y_par, true_y], ':', c='tab:gray')
ax[0].set_ylim(-1.1, 1.1)
ax[0].set_ylabel('ecosw', fontsize=14)
ax[0].legend()
ax[1].plot([0, np.max(min_depth[fin_good])], [0, 0], c='tab:grey', alpha=0.8)
ax[1].errorbar(min_depth[fin_good], ecosw_measure_2[fin_good],
               yerr=np.vstack((e_err_1[0][fin_good], e_err_1[1][fin_good])),
               capsize=2, marker='.', c='tab:blue', linestyle='none', label='eclipse model')
# for i in true_par.index:
#     if fin_good[i]:
#         ax[1].annotate(f'{i}', (min_depth[i], ecosw_measure_2[i]), alpha=0.6)
ax[1].set_ylim(-fix_range, fix_range)
ax[1].set_xlabel('secondary eclipse depth', fontsize=14)
ax[1].set_ylabel('$ecosw_{measured} - ecosw_{input}$', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()
# plot absolute i difference versus third light
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax[0].plot([0, np.max(true_par['l3'])], [90, 90], '--', c='tab:grey', alpha=0.8, label='$90^\circ$')
ax[0].scatter(true_par['l3'][fin_good], true_par['incl'][fin_good], marker='.', c='tab:grey', alpha=0.8, label='input')
ax[0].scatter(true_par['l3'][fin_good], obs_par['i_phys'][fin_good]/np.pi*180, marker='.', c='tab:blue',
              label='eclipse model')
for x_par, y_par, true_y in zip(true_par['l3'][fin_good], obs_par['i_phys'][fin_good]/np.pi*180, true_par['incl'][fin_good]):
    ax[0].plot([x_par, x_par], [y_par, true_y], ':', c='tab:gray')
ax[0].set_ylim(45, 95)
ax[0].set_ylabel('inclination (degrees)', fontsize=14)
ax[0].legend()
ax[1].plot([0, np.max(true_par['l3'])], [0, 0], c='tab:grey', alpha=0.8)
ax[1].errorbar(true_par['l3'][fin_good], i_measure_2[fin_good] / np.pi * 180,
               yerr=np.vstack((i_err_1[0][fin_good] / np.pi * 180, i_err_1[1][fin_good] / np.pi * 180)),
               capsize=2, marker='.', c='tab:blue', linestyle='none', label='eclipse model')
# for i in true_par.index:
#     if fin_good[i]:
#         ax[1].annotate(f'{i}', (true_par['l3'][fin_good][i], i_measure_2[i]/np.pi*180), alpha=0.6)
ax[1].set_xlabel('third light', fontsize=14)
ax[1].set_ylabel('$i_{measured} - i_{input}$ (degrees)', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()

# KDE and hist - error in p
norm_kde_2 = sp.stats.gaussian_kde(p_measure_2[fin_good], bw_method=1/(p_measure_2[fin_good]).std())
points = np.arange(-7, 7, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(p_measure_2[fin_good], bins=np.arange(-6.875, 7, 0.5), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_2(points) * len(p_measure_2[fin_good]), color='tab:blue', linewidth=4)
ax.set_xlabel('$\chi_p$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()
# KDE and hist - absolute error in e
norm_kde_1 = sp.stats.gaussian_kde(e_measure_1[fin_good], bw_method=0.02/(e_measure_1[fin_good]).std())
norm_kde_2 = sp.stats.gaussian_kde(e_measure_2[fin_good], bw_method=0.02/(e_measure_2[fin_good]).std())
points = np.arange(-0.3, 0.3, 0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(e_measure_1[fin_good], bins=np.arange(-0.3, 0.32, 0.02), linewidth=0, alpha=0.3)
ax.hist(e_measure_2[fin_good], bins=np.arange(-0.3, 0.32, 0.02), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_1(points), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_2(points), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$e_{measured} - e_{input}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - scaled error in e
norm_kde_4 = sp.stats.gaussian_kde(e_measure_4[fin_good], bw_method=1/(e_measure_4[fin_good]).std())
norm_kde_5 = sp.stats.gaussian_kde(e_measure_5[fin_good], bw_method=1/(e_measure_5[fin_good]).std())
points = np.arange(-14, 14, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(e_measure_4[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.hist(e_measure_5[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_4(points) * len(e_measure_4[fin_good]), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_5(points) * len(e_measure_5[fin_good]), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\chi_e$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - absolute error in ecosw
norm_kde_1 = sp.stats.gaussian_kde(ecosw_measure_1[fin_good], bw_method=0.005/(ecosw_measure_1[fin_good]).std())
norm_kde_2 = sp.stats.gaussian_kde(ecosw_measure_2[fin_good], bw_method=0.005/(ecosw_measure_2[fin_good]).std())
points = np.arange(-0.06, 0.06, 0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(ecosw_measure_1[fin_good], bins=np.arange(-0.06, 0.07, 0.005), linewidth=0, alpha=0.3)
ax.hist(ecosw_measure_2[fin_good], bins=np.arange(-0.06, 0.07, 0.005), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_1(points) / 2, color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_2(points) / 2, color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$ecos(w)_{measured} - ecos(w)_{input}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - scaled error in ecosw
norm_kde_4 = sp.stats.gaussian_kde(ecosw_measure_4[fin_good], bw_method=1/(ecosw_measure_4[fin_good]).std())
norm_kde_5 = sp.stats.gaussian_kde(ecosw_measure_5[fin_good], bw_method=1/(ecosw_measure_5[fin_good]).std())
points = np.arange(-14, 14, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(ecosw_measure_4[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.hist(ecosw_measure_5[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_4(points) * len(ecosw_measure_4[fin_good]), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_5(points) * len(ecosw_measure_5[fin_good]), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\chi_{ecos(w)}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - absolute error in esinw
norm_kde_1 = sp.stats.gaussian_kde(esinw_measure_1[fin_good], bw_method=0.02/(esinw_measure_1[fin_good]).std())
norm_kde_2 = sp.stats.gaussian_kde(esinw_measure_2[fin_good], bw_method=0.02/(esinw_measure_2[fin_good]).std())
points = np.arange(-0.35, 0.35, 0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(esinw_measure_1[fin_good], bins=np.arange(-0.35, 0.36, 0.02), linewidth=0, alpha=0.3)
ax.hist(esinw_measure_2[fin_good], bins=np.arange(-0.35, 0.36, 0.02), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_1(points), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_2(points), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$esin(w)_{measured} - esin(w)_{input}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - scaled error in esinw
norm_kde_4 = sp.stats.gaussian_kde(esinw_measure_4[fin_good], bw_method=1/(esinw_measure_4[fin_good]).std())
norm_kde_5 = sp.stats.gaussian_kde(esinw_measure_5[fin_good], bw_method=1/(esinw_measure_5[fin_good]).std())
points = np.arange(-14, 14, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(esinw_measure_4[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.hist(esinw_measure_5[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_4(points) * len(esinw_measure_4[fin_good]), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_5(points) * len(esinw_measure_5[fin_good]), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\chi_{esin(w)}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - absolute error in w
norm_kde_1 = sp.stats.gaussian_kde(w_measure_1[fin_good], bw_method=0.05/(w_measure_1[fin_good]).std())
norm_kde_2 = sp.stats.gaussian_kde(w_measure_2[fin_good], bw_method=0.05/(w_measure_2[fin_good]).std())
points = np.arange(-1, 1, 0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(w_measure_1[fin_good], bins=np.arange(-1, 1, 0.05), linewidth=0, alpha=0.3)
ax.hist(w_measure_2[fin_good], bins=np.arange(-1, 1, 0.05), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_1(points) * 2, color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_2(points) * 2, color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$w_{measured} - w_{input} (radians)$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - error in w
norm_kde_4 = sp.stats.gaussian_kde(w_measure_4[fin_good], bw_method=1/(w_measure_4[fin_good]).std())
norm_kde_5 = sp.stats.gaussian_kde(w_measure_5[fin_good], bw_method=1/(w_measure_5[fin_good]).std())
points = np.arange(-14, 14, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(w_measure_4[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.hist(w_measure_5[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_4(points) * len(w_measure_4[fin_good]), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_5(points) * len(w_measure_5[fin_good]), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\chi_w$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - absolute error in i
norm_kde_1 = sp.stats.gaussian_kde(i_measure_1[fin_good], bw_method=0.02/(i_measure_1[fin_good]).std())
norm_kde_2 = sp.stats.gaussian_kde(i_measure_2[fin_good], bw_method=0.02/(i_measure_2[fin_good]).std())
points = np.arange(-0.3, 0.3, 0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(i_measure_1[fin_good], bins=np.arange(-0.3, 0.32, 0.02), linewidth=0, alpha=0.3)
ax.hist(i_measure_2[fin_good], bins=np.arange(-0.3, 0.32, 0.02), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_1(points), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_2(points), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$i_{measured} - i_{input} (radians)$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - error in i
norm_kde_4 = sp.stats.gaussian_kde(i_measure_4[fin_good], bw_method=1/(i_measure_4[fin_good]).std())
norm_kde_5 = sp.stats.gaussian_kde(i_measure_5[fin_good], bw_method=1/(i_measure_5[fin_good]).std())
points = np.arange(-9, 9, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(i_measure_4[fin_good], bins=np.arange(-9, 10), linewidth=0, alpha=0.3)
ax.hist(i_measure_5[fin_good], bins=np.arange(-9, 10), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_4(points) * len(i_measure_4[fin_good]), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_5(points) * len(i_measure_5[fin_good]), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\chi_i$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - absolute error in cosi
norm_kde_1 = sp.stats.gaussian_kde(cosi_measure_1[fin_good], bw_method=0.02/(cosi_measure_1[fin_good]).std())
norm_kde_2 = sp.stats.gaussian_kde(cosi_measure_2[fin_good], bw_method=0.02/(cosi_measure_2[fin_good]).std())
points = np.arange(-0.3, 0.3, 0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(cosi_measure_1[fin_good], bins=np.arange(-0.3, 0.32, 0.02), linewidth=0, alpha=0.3)
ax.hist(cosi_measure_2[fin_good], bins=np.arange(-0.3, 0.32, 0.02), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_1(points), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_2(points), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$cos(i)_{measured} - cos(i)_{input} (radians)$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - error in cosi
norm_kde_4 = sp.stats.gaussian_kde(i_measure_4[fin_good], bw_method=1/(i_measure_4[fin_good]).std())
norm_kde_5 = sp.stats.gaussian_kde(i_measure_5[fin_good], bw_method=1/(i_measure_5[fin_good]).std())
points = np.arange(-9, 9, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(i_measure_4[fin_good], bins=np.arange(-9, 10), linewidth=0, alpha=0.3)
ax.hist(i_measure_5[fin_good], bins=np.arange(-9, 10), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_4(points) * len(i_measure_4[fin_good]), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_5(points) * len(i_measure_5[fin_good]), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\chi_{cos(i)}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - absolute error in r_sum
norm_kde_1 = sp.stats.gaussian_kde(r_sum_measure_1[fin_good], bw_method=0.02/(r_sum_measure_1[fin_good]).std())
norm_kde_2 = sp.stats.gaussian_kde(r_sum_measure_2[fin_good], bw_method=0.02/(r_sum_measure_2[fin_good]).std())
points = np.arange(-0.22, 0.22, 0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(r_sum_measure_1[fin_good], bins=np.arange(-0.22, 0.23, 0.02), linewidth=0, alpha=0.3)
ax.hist(r_sum_measure_2[fin_good], bins=np.arange(-0.22, 0.23, 0.02), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_1(points), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_2(points), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$r\_sum_{measured} - r\_sum_{input}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - error in r_sum
norm_kde_4 = sp.stats.gaussian_kde(r_sum_measure_4[fin_good], bw_method=1/(r_sum_measure_4[fin_good]).std())
norm_kde_5 = sp.stats.gaussian_kde(r_sum_measure_5[fin_good], bw_method=1/(r_sum_measure_5[fin_good]).std())
points = np.arange(-9, 9, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(r_sum_measure_4[fin_good], bins=np.arange(-9, 10), linewidth=0, alpha=0.3)
ax.hist(r_sum_measure_5[fin_good], bins=np.arange(-9, 10), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_4(points) * len(r_sum_measure_4[fin_good]), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_5(points) * len(r_sum_measure_5[fin_good]), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\chi_{r sum}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - absolute error in phi_0
norm_kde_1 = sp.stats.gaussian_kde(phi_0_measure_1[fin_good], bw_method=0.02/(phi_0_measure_1[fin_good]).std())
norm_kde_2 = sp.stats.gaussian_kde(phi_0_measure_2[fin_good], bw_method=0.02/(phi_0_measure_2[fin_good]).std())
points = np.arange(-0.15, 0.15, 0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(phi_0_measure_1[fin_good], bins=np.arange(-0.15, 0.16, 0.02), linewidth=0, alpha=0.3)
ax.hist(phi_0_measure_2[fin_good], bins=np.arange(-0.15, 0.16, 0.02), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_1(points), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_2(points), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\phi_{0, measured} - \phi_{0, true}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - error in phi_0
norm_kde_4 = sp.stats.gaussian_kde(phi_0_measure_4[fin_good], bw_method=1/(phi_0_measure_4[fin_good]).std())
norm_kde_5 = sp.stats.gaussian_kde(phi_0_measure_5[fin_good], bw_method=1/(phi_0_measure_5[fin_good]).std())
points = np.arange(-15, 15, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(phi_0_measure_4[fin_good], bins=np.arange(-15, 16), linewidth=0, alpha=0.3)
ax.hist(phi_0_measure_5[fin_good], bins=np.arange(-15, 16), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_4(points) * len(phi_0_measure_4[fin_good]), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_5(points) * len(phi_0_measure_5[fin_good]), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\chi_{\phi_0}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - absolute error in r_rat
norm_kde_1 = sp.stats.gaussian_kde(r_rat_measure_1[fin_good], bw_method=0.02/(r_rat_measure_1[fin_good]).std())
norm_kde_2 = sp.stats.gaussian_kde(r_rat_measure_2[fin_good], bw_method=0.02/(r_rat_measure_2[fin_good]).std())
points = np.arange(-0.5, 0.5, 0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(r_rat_measure_1[fin_good], bins=np.arange(-0.5, 0.55, 0.05), linewidth=0, alpha=0.3)
ax.hist(r_rat_measure_2[fin_good], bins=np.arange(-0.5, 0.55, 0.05), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_1(points), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_2(points), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$r\_rat_{measured} - r\_rat_{input}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - error in r_rat
norm_kde_4 = sp.stats.gaussian_kde(r_rat_measure_4[fin_good], bw_method=1/(r_rat_measure_4[fin_good]).std())
norm_kde_5 = sp.stats.gaussian_kde(r_rat_measure_5[fin_good], bw_method=1/(r_rat_measure_5[fin_good]).std())
points = np.arange(-14, 14, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(r_rat_measure_4[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.hist(r_rat_measure_5[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_4(points) * len(r_rat_measure_4[fin_good]), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_5(points) * len(r_rat_measure_5[fin_good]), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\chi_{r ratio}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - absolute error in log_rr
norm_kde_1 = sp.stats.gaussian_kde(log_rr_measure_1[fin_good], bw_method=0.02/(log_rr_measure_1[fin_good]).std())
norm_kde_2 = sp.stats.gaussian_kde(log_rr_measure_2[fin_good], bw_method=0.02/(log_rr_measure_2[fin_good]).std())
points = np.arange(-0.5, 0.5, 0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(log_rr_measure_1[fin_good], bins=np.arange(-0.5, 0.55, 0.05), linewidth=0, alpha=0.3)
ax.hist(log_rr_measure_2[fin_good], bins=np.arange(-0.5, 0.55, 0.05), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_1(points), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_2(points), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$log_{10}(r\_rat_{measured}) - log_{10}(r\_rat_{input})$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - error in log_rr
norm_kde_4 = sp.stats.gaussian_kde(log_rr_measure_4[fin_good], bw_method=1/(log_rr_measure_4[fin_good]).std())
norm_kde_5 = sp.stats.gaussian_kde(log_rr_measure_5[fin_good], bw_method=1/(log_rr_measure_5[fin_good]).std())
points = np.arange(-14, 14, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(log_rr_measure_4[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.hist(log_rr_measure_5[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_4(points) * len(log_rr_measure_4[fin_good]), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_5(points) * len(log_rr_measure_5[fin_good]), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\chi_{log_{10}(r\_rat)}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - absolute error in sb_rat
norm_kde_1 = sp.stats.gaussian_kde(sb_rat_measure_1[fin_good], bw_method=0.02/(sb_rat_measure_1[fin_good]).std())
norm_kde_2 = sp.stats.gaussian_kde(sb_rat_measure_2[fin_good], bw_method=0.02/(sb_rat_measure_2[fin_good]).std())
points = np.arange(-0.5, 0.5, 0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(sb_rat_measure_1[fin_good], bins=np.arange(-0.5, 0.55, 0.05), linewidth=0, alpha=0.3)
ax.hist(sb_rat_measure_2[fin_good], bins=np.arange(-0.5, 0.55, 0.05), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_1(points), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_2(points), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$sb\_rat_{measured} - sb\_rat_{input}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - error in sb_rat
norm_kde_4 = sp.stats.gaussian_kde(sb_rat_measure_4[fin_good], bw_method=1/(sb_rat_measure_4[fin_good]).std())
norm_kde_5 = sp.stats.gaussian_kde(sb_rat_measure_5[fin_good], bw_method=1/(sb_rat_measure_5[fin_good]).std())
points = np.arange(-14, 14, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(sb_rat_measure_4[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.hist(sb_rat_measure_5[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_4(points) * len(sb_rat_measure_4[fin_good]), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_5(points) * len(sb_rat_measure_5[fin_good]), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\chi_{sb ratio}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - absolute error in log_sb
norm_kde_1 = sp.stats.gaussian_kde(log_sb_measure_1[fin_good], bw_method=0.02/(log_sb_measure_1[fin_good]).std())
norm_kde_2 = sp.stats.gaussian_kde(log_sb_measure_2[fin_good], bw_method=0.02/(log_sb_measure_2[fin_good]).std())
points = np.arange(-0.5, 0.5, 0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(log_sb_measure_1[fin_good], bins=np.arange(-0.5, 0.55, 0.05), linewidth=0, alpha=0.3)
ax.hist(log_sb_measure_2[fin_good], bins=np.arange(-0.5, 0.55, 0.05), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_1(points), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_2(points), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$log_{10}(sb\_rat_{measured}) - log_{10}(sb\_rat_{input})$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
# KDE and hist - error in log_sb
norm_kde_4 = sp.stats.gaussian_kde(log_sb_measure_4[fin_good], bw_method=1/(log_sb_measure_4[fin_good]).std())
norm_kde_5 = sp.stats.gaussian_kde(log_sb_measure_5[fin_good], bw_method=1/(log_sb_measure_5[fin_good]).std())
points = np.arange(-14, 14, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(log_sb_measure_4[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.hist(log_sb_measure_5[fin_good], bins=np.arange(-14, 15), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde_4(points) * len(log_sb_measure_4[fin_good]), color='tab:blue', linewidth=4, label='formulae')
ax.plot(points, norm_kde_5(points) * len(log_sb_measure_5[fin_good]), color='tab:orange', linewidth=4, label='eclipse model')
ax.set_xlabel('$\chi_{log_{10}(sb\_rat)}$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()

# frequency analysis
# number of pulsations found vs. input
n_f_true = true_par['npulsations'][fin_good].to_numpy()
n_f_tot = obs_par['total_freqs'][fin_good].to_numpy()
n_f_pass = obs_par['passed_both'][fin_good].to_numpy()
n_f_hpass = obs_par['passed_harmonics'][fin_good].to_numpy()
sorter_n_f = np.argsort(n_f_true)
data_range = np.array([0, np.max(n_f_tot)])
short_t = (obs_par['t_tot'][fin_good] < 50)
long_t = (obs_par['t_tot'][fin_good] > 50)
fig, ax = plt.subplots(figsize=(6, 6))
# ax.scatter(n_f_true, n_f_pass, marker='d', c='tab:blue', label='passing criteria')
# ax.scatter(n_f_true, n_f_hpass, marker='^', c='tab:green', label='harmonics passing criteria')
# ax.scatter(n_f_true, n_f_pass - n_f_hpass, marker='o', c='tab:orange', label='passing - passing harmonics')
# ax.errorbar(n_f_true, n_f_pass - n_f_hpass, yerr=np.sqrt(n_f_pass - n_f_hpass),
            # marker='o', c='tab:blue', linestyle='none', label='passing - passing harmonics')
ax.errorbar(n_f_true[short_t], n_f_pass[short_t] - n_f_hpass[short_t], yerr=np.sqrt(n_f_pass[short_t] - n_f_hpass[short_t]),
            marker='o', c='tab:blue', linestyle='none', label='month')
ax.errorbar(n_f_true[long_t], n_f_pass[long_t] - n_f_hpass[long_t], yerr=np.sqrt(n_f_pass[long_t] - n_f_hpass[long_t]),
            marker='o', c='tab:orange', linestyle='none', label='year')
ax.plot([0, 100], [0, 100], c='tab:grey', alpha=0.2)
# for p, i in enumerate(true_par.index[fin_good]):
#     n = obs_par['stage'][i]
    # ax.annotate(f'{i}', (n_f_true[p], n_f_pass[p] - n_f_hpass[p]), alpha=0.6)
ax.set_xlabel('number of input sinusoids', fontsize=14)
ax.set_ylabel('number of output sinusoids', fontsize=16)
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
# KDE and hist - n freq
n_sin_dif = ((n_f_pass - n_f_hpass) - n_f_true)
n_sin_measure = n_sin_dif / np.clip(np.sqrt(np.abs(n_f_pass - n_f_hpass)), 1, None)
norm_kde = sp.stats.gaussian_kde(n_sin_measure, bw_method=1/n_sin_measure.std())
points = np.arange(-9, 9, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
# ax.hist(n_sin_measure, bins=np.arange(-9, 10), linewidth=0, alpha=0.3)
ax.hist(n_sin_measure[short_t], bins=np.arange(-9, 10), linewidth=0, alpha=0.3, label='month')
ax.hist(n_sin_measure[long_t], bins=np.arange(-9, 10), linewidth=0, alpha=0.3, label='year')
ax.plot(points, norm_kde(points) * len(n_sin_measure), color='tab:blue', linewidth=4)
ax.set_xlabel(r'$\frac{(n - n_h) - n_{input}}{\sqrt{n - n_h}}$', fontsize=20)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.legend()
plt.tight_layout()
plt.show()
# individual frequencies (for case 26)
case = '026'
sin_true = np.loadtxt(syn_dir + f'/pulse_data/sim_{case}_lc_pulse_info.dat', delimiter=',')
times, signal, signal_err = np.loadtxt(all_files[int(case)], usecols=(0, 1, 2), unpack=True)
freqs, ampls = sts.tsf.scargle(times, signal)
results = sts.ut.read_parameters_hdf5(syn_dir + f'/sim_{case}_lc_analysis/sim_{case}_lc_analysis_8.hdf5', verbose=False)
const, slope, f_n, a_n, ph_n = results['sin_mean']
c_err, sl_err, f_n_err, a_n_err, ph_n_err = results['sin_err']
passed_sigma, passed_snr, passed_both, passed_h = results['sin_select']
matcher_f = np.zeros(len(sin_true), dtype=int)
for i in range(len(sin_true)):
    matcher_f[i] = np.arange(len(f_n[passed_both]))[np.argmin(np.abs(f_n[passed_both] - sin_true[i, 0]))]
# hist/KDE of sinusoid parameters - f, a, ph
f_measure = (f_n[passed_both][matcher_f] - sin_true[:, 0]) / f_n_err[passed_both][matcher_f]
f_measure = f_measure[np.abs(f_measure) < 30]
norm_kde = sp.stats.gaussian_kde(f_measure, bw_method=1 / f_measure.std())
points = np.arange(-5, 5, 0.01)
fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(f_measure, bins=np.arange(-5, 5.5, 0.25), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde(points) * len(f_measure) / 3, color='tab:blue', linewidth=4)
ax.set_xlabel('$\chi_f$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()
a_measure = (a_n[passed_both][matcher_f] - sin_true[:, 1]) / a_n_err[passed_both][matcher_f]
a_measure = a_measure[np.abs(a_measure) < 30]
norm_kde = sp.stats.gaussian_kde(a_measure, bw_method=1 / a_measure.std())
fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(a_measure, bins=np.arange(-5, 5.5, 0.25), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde(points) * len(a_measure) / 3, color='tab:blue', linewidth=4)
ax.set_xlabel('$\chi_a$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()
phase_shift = (2*np.pi * f_n[passed_both][matcher_f] * np.mean(times))
sin_true_ph_rad = sin_true[:, 2] * (2 * np.pi) % (2 * np.pi)  # phases were mistakenly multiplied by 2pi
ph_measure = ((ph_n[passed_both][matcher_f] - phase_shift) % (2 * np.pi) - sin_true_ph_rad) / ph_n_err[passed_both][matcher_f]
ph_measure = ph_measure[np.abs(ph_measure) < 30]
norm_kde = sp.stats.gaussian_kde(ph_measure, bw_method=1 / ph_measure.std())
fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(ph_measure, bins=np.arange(-5, 5.5, 0.25), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde(points) * len(ph_measure) / 3, color='tab:blue', linewidth=4)
ax.set_xlabel('$\chi_\phi$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()


# KEPLER PERIOD TESTS
# kepler eb catalogue test results
kep_dir = '~/Kepler_EB_catalogue'
summary_file = os.path.join(kep_dir, '01433410.00.lc_analysis', '01433410.00.lc_analysis_summary.csv')
hdr = np.loadtxt(summary_file, usecols=(0), delimiter=',', unpack=True, dtype=str)
obs_par_dtype = np.dtype([('id', '<U20'), ('stage', '<U20')] + [(name, float) for name in hdr[2:]])
kep_ebs = np.loadtxt(os.path.join(kep_dir, 'kepler_eb_files.dat'), dtype=str)[1:]
obs_par = np.ones(len(kep_ebs), dtype=obs_par_dtype)
not_done = []
for k, file in enumerate(kep_ebs):
    target_id = os.path.splitext(os.path.basename(file))[0]
    data_dir = os.path.join(kep_dir, f'{target_id}_analysis')
    summary_file = os.path.join(data_dir, f'{target_id}_analysis_summary.csv')
    if os.path.isfile(summary_file):
        obs_par[k] = tuple(np.loadtxt(summary_file, usecols=(1), delimiter=',', unpack=True, dtype=str))
    else:
        not_done.append(summary_file)
obs_par = pd.DataFrame(obs_par, columns=hdr)
obs_par.to_csv(os.path.join(kep_dir + '_summary.csv'), index=False)

# load kepler eb catalogue and result summary
kepobs_par = pd.read_csv(os.path.join(kep_dir + '_summary.csv'))
kepcat_par = pd.read_csv(os.path.join(kep_dir, 'kepler_eb_catalog.csv'), skiprows=7)

# periods
kep_zero = (np.char.find(kep_ebs, '.00.') != -1)  # files with index .00.
kep_p_avail = (kepobs_par['id'][kep_zero].to_numpy() != '1')  # period saved
min_max = [np.min(kepcat_par['period']), np.max(kepcat_par['period'])]
obs_p = kepobs_par['period'][kep_zero][kep_p_avail].to_numpy()
obs_p_err = kepobs_par['p_err'][kep_zero][kep_p_avail].to_numpy()
cat_p = kepcat_par['period'][kep_p_avail].to_numpy()
cat_p_err = kepcat_par['period_err'][kep_p_avail].to_numpy()
cat_morph = kepcat_par['morph'][kep_p_avail].to_numpy()
p_diff = obs_p - cat_p
p_diff_2 = p_diff / cat_p
obs_p_err2 = obs_p_err / cat_p
p_diff_3 = (p_diff) / obs_p_err
select_good_p_3 = (np.abs(p_diff_2) < 0.01) & (obs_p != -1)
p_diff_mult = []
p_diff_m = []
select_good_p_m = []
for m in [1/5, 1/4, 1/3, 1/2, 2, 3, 4, 5]:
    p_diff_mult.append(obs_p - cat_p / m)
    p_diff_m.append(p_diff_mult[-1] / obs_p_err)
    select_good_p_m.append((np.abs(p_diff_mult[-1] / cat_p) < 0.01) & (obs_p != -1))
select_good_p_all_m = np.sum(select_good_p_m, axis=0, dtype=bool)
# intrinsic variability (at non-harmonic frequencies)
obs_std_1 = kepobs_par['std_1'][kep_zero][kep_p_avail].to_numpy()
obs_std_2 = kepobs_par['std_2'][kep_zero][kep_p_avail].to_numpy()
obs_std_4 = kepobs_par['std_4'][kep_zero][kep_p_avail].to_numpy()
obs_std_5 = np.sqrt(obs_std_2**2 - obs_std_4**2)
obs_std_5_rat = obs_std_5 / obs_std_1
var_mask = (obs_std_5_rat > 6) & (cat_morph < 0.5)
# eccentricities
obs_e_form = kepobs_par['e_form'][kep_zero][kep_p_avail].to_numpy()
obs_e_l = kepobs_par['e_low'][kep_zero][kep_p_avail].to_numpy()
obs_e_u = kepobs_par['e_upp'][kep_zero][kep_p_avail].to_numpy()
obs_e_err = np.vstack((obs_e_l, obs_e_u))
obs_e_phys = kepobs_par['e_phys'][kep_zero][kep_p_avail].to_numpy()
obs_ecosw_phys = kepobs_par['ecosw_phys'][kep_zero][kep_p_avail].to_numpy()
obs_ecosw_l = kepobs_par['ecosw_low'][kep_zero][kep_p_avail].to_numpy()
obs_ecosw_u = kepobs_par['ecosw_upp'][kep_zero][kep_p_avail].to_numpy()
obs_ecosw_err = np.vstack((obs_ecosw_l, obs_ecosw_u))
# KDE and hist - percentage error in p
norm_kde = sp.stats.gaussian_kde(p_diff_2[select_good_p_3], bw_method=0.000004/p_diff_2[select_good_p_3].std())
points = np.arange(-0.00009, 0.00009, 0.0000005)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(p_diff_2[select_good_p_3], bins=np.arange(-0.00009, 0.000091, 0.000004), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde(points) * len(p_diff_2[select_good_p_3])*0.000004, color='tab:blue', linewidth=4)
ax.set_xlabel(r'$\frac{P_{measured} - P_{catalogue}}{P_{catalogue}}$', fontsize=18)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.get_xaxis().get_offset_text().set_fontsize(14)
plt.tight_layout()
plt.show()
# KDE and hist - error in p
norm_kde = sp.stats.gaussian_kde(p_diff_3[select_good_p_3], bw_method=1/p_diff_3[select_good_p_3].std())
points = np.arange(-8, 8, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(p_diff_3[select_good_p_3], bins=np.arange(-8, 8.1, 0.2), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde(points) * len(p_diff_3[select_good_p_3])*0.2, color='tab:blue', linewidth=4)
ax.set_xlabel('$\chi_p$', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()
# at half period and multiples
p_diff_all_m = np.copy(p_diff_3)
for p_d, select in zip(p_diff_m, select_good_p_m):
    p_diff_all_m[select] = p_d[select]
# norm_kde = sp.stats.gaussian_kde(p_diff_all_m[select_good_p_all_m], bw_method=1/p_diff_all_m[select_good_p_all_m].std())
norm_kde = sp.stats.gaussian_kde(p_diff_m[4][select_good_p_m[4]], bw_method=1/p_diff_m[4][select_good_p_m[4]].std())
points = np.arange(-8, 8, 0.01)
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(p_diff_m[4][select_good_p_m[4]], bins=np.arange(-8, 8.1, 0.4), linewidth=0, alpha=0.3)
# ax.hist(p_diff_all_m[select_good_p_all_m], bins=np.arange(-8, 8.1, 0.4), linewidth=0, alpha=0.3)
ax.plot(points, norm_kde(points) * len(p_diff_all_m[select_good_p_all_m])*0.4, color='tab:blue', linewidth=4)
ax.set_xlabel('$\chi_p$ at half period', fontsize=14)
# ax.set_xlabel('$\chi_p$ at other multiples', fontsize=14)
ax.set_ylabel('number of cases', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()

# eccentricities [select correct p, e between 0 and 1, error < 0.1, morph < 0.5]
good_e_3 = (obs_e_form[select_good_p_3] > 0)
good_e_3 &= (obs_e_err[0][select_good_p_3] < 0.1) & (obs_e_err[1][select_good_p_3] < 0.1)
good_e_3 &= (cat_morph[select_good_p_3] < 0.5)
good_e_p_3 = (obs_e_phys[select_good_p_3] > 0)
good_e_p_3 &= (obs_e_err[0][select_good_p_3] < 0.1) & (obs_e_err[1][select_good_p_3] < 0.1)
good_e_p_3 &= (cat_morph[select_good_p_3] < 0.5)
periods_line = np.logspace(0, 2.4, 1000)
line_theo = np.sqrt(1 - (5 / periods_line)**(2/3))
line_theo[~np.isfinite(line_theo)] = 0
plotting_kic = ['04544587', '7943535', '8196180', '11867071']
fig, ax = plt.subplots(figsize=(12, 12))
ax.errorbar(obs_p[select_good_p_3][good_e_p_3], obs_e_phys[select_good_p_3][good_e_p_3],
            xerr=obs_p_err[select_good_p_3][good_e_p_3], yerr=obs_e_err[:, select_good_p_3][:, good_e_p_3],
            marker='.', color='tab:blue', linestyle='none', capsize=2, zorder=1)
for kic in plotting_kic:  # janky way to plot a few points in different colour
    mask = [kic in item for item in kepobs_par['id'][kep_zero][kep_p_avail].to_numpy().astype(str)]
    mask = np.array(mask)
    i = np.arange(len(kepobs_par[kep_zero][kep_p_avail]))[mask][0]
    p = obs_p[i]
    e = obs_e_phys[i]
    p_e = obs_p_err[i]
    e_e = obs_e_err[:, i]
    ax.errorbar(p, e, xerr=p_e, yerr=e_e.reshape((2, 1)), marker='.', color='tab:orange', linestyle='none', capsize=2, zorder=1)
ax.plot(periods_line, line_theo, c='k', linestyle='--', zorder=2)
ax.set_xlabel('period (d)', fontsize=14)
ax.set_ylabel('eccentricity', fontsize=14)
plt.tight_layout()
plt.xscale('log')
plt.show()
# variability
good_e_p_3 = (obs_e_phys[var_mask][select_good_p_3[var_mask]] > 0)
good_e_p_3 &= (obs_e_err[0][var_mask][select_good_p_3[var_mask]] < 0.1) & (obs_e_err[1][var_mask][select_good_p_3[var_mask]] < 0.1)
good_e_p_3 &= (cat_morph[var_mask][select_good_p_3[var_mask]] < 0.5)
periods_line = np.logspace(0, 2.4, 1000)
line_theo = np.sqrt(1 - (5 / periods_line)**(2 / 3))
line_theo_2 = np.sqrt(1 - (6.5 / periods_line)**(2 / 3))
line_theo[~np.isfinite(line_theo)] = 0
line_theo_2[~np.isfinite(line_theo_2)] = 0
plotting_kic = ['08719324', '09899216', '05034333', '11706658', '07833144']
fig, ax = plt.subplots(figsize=(12, 12))
ax.errorbar(obs_p[var_mask][select_good_p_3[var_mask]][good_e_p_3], obs_e_phys[var_mask][select_good_p_3[var_mask]][good_e_p_3],
            xerr=obs_p_err[var_mask][select_good_p_3[var_mask]][good_e_p_3], yerr=obs_e_err[:, var_mask][:, select_good_p_3[var_mask]][:, good_e_p_3],
            marker='.', color='tab:blue', linestyle='none', capsize=2, zorder=1)
for kic in plotting_kic:  # janky way to plot a few points in different colour
    mask = [kic in item for item in kepobs_par['id'][kep_zero][kep_p_avail].to_numpy().astype(str)]
    mask = np.array(mask)
    i = np.arange(len(kepobs_par[kep_zero][kep_p_avail]))[mask][0]
    p = obs_p[i]
    e = obs_e_phys[i]
    p_e = obs_p_err[i]
    e_e = obs_e_err[:, i]
    ax.errorbar(p, e, xerr=p_e, yerr=e_e.reshape((2, 1)), marker='.', color='tab:red', linestyle='none', capsize=2, zorder=1)
ax.plot(periods_line, line_theo, c='k', linestyle='--', zorder=2)
ax.plot(periods_line, line_theo_2, c='grey', linestyle='--', zorder=2)
ax.set_xlabel('period (d)', fontsize=14)
ax.set_ylabel('eccentricity', fontsize=14)
plt.tight_layout()
plt.xscale('log')
plt.show()
# ecosw
fig, ax = plt.subplots(figsize=(12, 12))
ax.errorbar(obs_p[select_good_p_3][good_e_p_3], obs_ecosw_phys[select_good_p_3][good_e_p_3],
            xerr=obs_p_err[select_good_p_3][good_e_p_3], yerr=obs_ecosw_err[:, select_good_p_3][:, good_e_p_3],
            marker='.', color='tab:blue', linestyle='none', capsize=2)
ax.set_xlabel('period (d)', fontsize=14)
ax.set_ylabel('e cos(w)', fontsize=14)
plt.xscale('log')
plt.tight_layout()
plt.show()
