import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import scipy.stats as st
from utils import pvalue_asterisk, data_to_axes, formatted_ax

ROOT = os.getcwd()
EXP_GLOB = os.path.join(ROOT, 'derivatives/experiment_*')
output_path = os.path.join(ROOT, 'fig')

n_sources = 2
n_sessions = 100
n_trials = 256

exp_dirs = sorted(glob.glob(EXP_GLOB))
n_exps = len(exp_dirs)
response_KLD_all = []
s1mean_responses_on_all = []
s1mean_responses_off_all = []
for exp_id, exp_dir in enumerate(exp_dirs):
    neuronal_responses = np.load(f'{exp_dir}/neuronal_responses.npy')
    source_states = np.load(f'{exp_dir}/hidden_source_states.npy')
    source_states = source_states[:, 0] * 1 + source_states[:, 1] * 2
    n_channels = neuronal_responses.shape[2]
    
    response_KLD = np.zeros((n_sessions, n_channels))
    lambda10 = np.mean(neuronal_responses[:, source_states == 1, :], axis = 1)
    lambda01 = np.mean(neuronal_responses[:, source_states == 2, :], axis = 1)
    invalid_mask = (lambda10 == 0) | (lambda01 == 0)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        response_KLD = lambda10 * np.log(lambda10 / lambda01) - lambda10 + lambda01
    response_KLD[invalid_mask] = -1
    valid_channels = np.where(~(response_KLD == -1).any(axis = 0))[0]
    response_KLD = response_KLD[:, valid_channels]
    np.save(f'{exp_dir}/response_KLD.npy', response_KLD)
    response_KLD_all.append(response_KLD)
    
    

    s1_preference = np.mean(neuronal_responses[:, source_states == 1, :], axis = (0, 1))
    s2_preference = np.mean(neuronal_responses[:, source_states == 2, :], axis = (0, 1))
    preference_diff = s1_preference - s2_preference
    response_session_mean = np.mean(neuronal_responses, axis = 1)
    response_session_mean_min = np.min(response_session_mean, axis = 0)
    response_mean = np.mean(response_session_mean, axis = 0)
    s1_preferring_electrodes = np.arange(n_channels)[(preference_diff > 0) & (response_session_mean_min > 0) & (response_mean > 0)]
    s2_preferring_electrodes = np.arange(n_channels)[(preference_diff < 0) & (response_session_mean_min > 0) & (response_mean > 0)]
    np.save(f'{exp_dir}/s1_preferring_electrodes.npy', s1_preferring_electrodes)
    np.save(f'{exp_dir}/s2_preferring_electrodes.npy', s2_preferring_electrodes)
    s1on_mask = (source_states % 2 == 1)
    s1on_responses = neuronal_responses[:, s1on_mask, :]
    s1off_responses = neuronal_responses[:, ~s1on_mask, :]
    s1mean_responses_on = np.mean(s1on_responses[:, :, s1_preferring_electrodes], axis = (1, 2))
    s1mean_responses_off = np.mean(s1off_responses[:, :, s1_preferring_electrodes], axis = (1, 2))
    s1mean_responses_on_all.append(s1mean_responses_on)
    s1mean_responses_off_all.append(s1mean_responses_off)

    
ax = formatted_ax()
response_KLD_all = np.concatenate(response_KLD_all, axis = 1)
response_KLD_change = response_KLD_all - response_KLD_all[0:1, :]
response_KLD_change_mean = np.mean(response_KLD_change, axis = 1)
response_KLD_change_se = np.std(response_KLD_change, axis = 1, ddof = 1) / np.sqrt(response_KLD_all.shape[1])
_, pvalue_KLD = st.wilcoxon(response_KLD_change[-1])
ax.plot(np.arange(1, n_sessions + 1), response_KLD_change_mean, color = 'black')
ax.fill_between(np.arange(1, n_sessions + 1), response_KLD_change_mean - response_KLD_change_se, response_KLD_change_mean + response_KLD_change_se, color = 'black', alpha = 0.2)
ax.axhline(0, linestyle = '--', linewidth = 1, color = 'r')
ymin = data_to_axes(0, 0 + 0.02, ax)[1]
ymax = data_to_axes(0, response_KLD_change_mean[-1], ax)[1]
ax.axvline(102, ymin = ymin, ymax = ymax, color = 'black', linewidth = 2, solid_capstyle = 'butt', clip_on = False)
ax.text(105, response_KLD_change_mean[-1]/2, pvalue_asterisk(pvalue_KLD), ha='left', va='bottom', fontsize=20)
ax.set_xlabel('Session #', size = 18)
ax.set_ylabel('Response KLD change', size = 18)
plt.savefig(f'{output_path}/Fig2c.pdf', dpi = 1200)
plt.close()

fig2c_df = pd.DataFrame()
fig2c_df['session #'] = np.arange(1, 101)
fig2c_df['response_KLD_change_mean'] = response_KLD_change_mean
fig2c_df['response_KLD_change_se'] = response_KLD_change_se
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    fig2c_df.to_excel(writer, sheet_name = 'Fig2c', index = False)

ax = formatted_ax()
s1mean_responses_on_all = np.stack(s1mean_responses_on_all, axis = 1)
s1mean_responses_off_all = np.stack(s1mean_responses_off_all, axis = 1)
s1mean_responses_on_change = s1mean_responses_on_all - s1mean_responses_on_all[0:1, :]
s1mean_responses_off_change = s1mean_responses_off_all - s1mean_responses_off_all[0:1, :]
s1mean_responses_on_change_mean = np.mean(s1mean_responses_on_change, axis = 1)
s1mean_responses_off_change_mean = np.mean(s1mean_responses_off_change, axis = 1)
s1mean_responses_on_change_se = np.std(s1mean_responses_on_change, axis = 1, ddof = 1) / np.sqrt(n_exps)
s1mean_responses_off_change_se = np.std(s1mean_responses_off_change, axis = 1, ddof = 1) / np.sqrt(n_exps)
_, pvalue_s1onoff = st.wilcoxon(s1mean_responses_on_change[-1], s1mean_responses_off_change[-1])
ax.plot(np.arange(1, n_sessions + 1), s1mean_responses_on_change_mean, color = 'crimson')
ax.plot(np.arange(1, n_sessions + 1), s1mean_responses_off_change_mean, color = 'mediumblue')
ax.fill_between(np.arange(1, n_sessions + 1), s1mean_responses_on_change_mean - s1mean_responses_on_change_se, s1mean_responses_on_change_mean + s1mean_responses_on_change_se, color = 'crimson', alpha = 0.2)
ax.fill_between(np.arange(1, n_sessions + 1), s1mean_responses_off_change_mean - s1mean_responses_off_change_se, s1mean_responses_off_change_mean + s1mean_responses_off_change_se, color = 'mediumblue', alpha = 0.2)
ax.axhline(0, linestyle = '--', linewidth = 1, color = 'r')
ymin = data_to_axes(0, s1mean_responses_off_change_mean[-1], ax)[1]
ymax = data_to_axes(0, s1mean_responses_on_change_mean[-1], ax)[1]
ax.axvline(x = n_sessions + 2, ymin = ymin, ymax = ymax, color = 'black', linewidth = 2, solid_capstyle = 'butt', clip_on = False)
ax.text(n_sessions + 5,(s1mean_responses_off_change_mean[-1] + s1mean_responses_on_change_mean[-1]) / 2 - 0.2, pvalue_asterisk(pvalue_s1onoff), ha='left', va='bottom', fontsize=20)
ax.set_xlabel('Session #', size = 18)
ax.set_ylabel('Response change [spike/trial]', size = 18)
plt.savefig(f'{output_path}/Fig2d.pdf', dpi = 1200)
plt.close()
fig2d_df = pd.DataFrame()
fig2d_df['session #'] = np.arange(1, 101)
fig2d_df['s1mean_change_mean_s1on'] = s1mean_responses_on_change_mean
fig2d_df['s1mean_change_mean_s1off'] = s1mean_responses_off_change_mean
fig2d_df['s1mean_change_se_s1on'] = s1mean_responses_on_change_se
fig2d_df['s1mean_change_se_s1off'] = s1mean_responses_off_change_se
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    fig2d_df.to_excel(writer, sheet_name = 'Fig2d', index = False)