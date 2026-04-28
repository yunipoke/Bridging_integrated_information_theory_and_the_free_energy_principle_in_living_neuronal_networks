import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from utils import formatted_ax, forest_plot, plot_session_transitions, iqr, mutual_information
from multiprocessing import Pool

ROOT = os.getcwd()
EXP_GLOB = os.path.join(ROOT, 'derivatives/experiment_*')
output_path = os.path.join(ROOT, 'fig')

n_sources = 2
n_sessions = 100
n_trials = 256
Phi_R_color = 'mediumpurple'

exp_dirs = sorted(glob.glob(EXP_GLOB))
n_exps = len(exp_dirs)
Phi_R_mc_all = []
maximal_Phi_R_mc_all = []
maximal_scale_all = []
main_complex_size_all = []
mean_coreness_all = []
mean_response_iqr_all = []
response_iqr_inside_all = []
response_iqr_outside_all = []
for exp_id, exp_dir in enumerate(exp_dirs):
    neuronal_responses = np.load(f'{exp_dir}/neuronal_responses.npy')
    s1_preferring_electrodes = np.load(f'{exp_dir}/s1_preferring_electrodes.npy')
    s2_preferring_electrodes = np.load(f'{exp_dir}/s2_preferring_electrodes.npy')
    source_states = np.load(f'{exp_dir}/hidden_source_states.npy')
    source_states = source_states[:, 0] * 1 + source_states[:, 1] * 2
    n_channels = neuronal_responses.shape[2]

    preferring_electrodes = np.concatenate([s1_preferring_electrodes, s2_preferring_electrodes])
    n_prefs = len(preferring_electrodes)
    n_stims = (source_states != 0).sum()
    n_obs = 290

    temporal_scales = [1 << i for i in range(7)]
    
    if not os.path.exists(f'{exp_dir}/pairwise_Phi_R'):
        os.mkdir(f'{exp_dir}/pairwise_Phi_R')

    def calculate_pairwise_Phi_R(i_session):
        spike_idx = np.load(f'{exp_dir}/pref_spike_series/session_{i_session + 1:03d}.npy')
        spike_series = np.zeros((n_prefs, n_trials, n_obs), dtype = int)
        np.add.at(spike_series, tuple(spike_idx.T), 1)
        spike_series = spike_series[:, source_states != 0, :]
        spike_cumsum = np.zeros((n_prefs, n_stims, n_obs + 1))
        spike_cumsum[:, :, 1:] = np.cumsum(spike_series, axis = 2)

        for tau in temporal_scales:
            try:
                os.mkdir(f'{exp_dir}/pairwise_Phi_R/tau={tau}ms')
            except:
                pass

            values = spike_cumsum[:, :, tau:] - spike_cumsum[:, :, :-tau]
            medians = np.median(values, axis = (1, 2), keepdims = True)
            from_state = spike_cumsum[:, :, tau:-tau] - spike_cumsum[:, :, :-2 * tau]
            to_state = spike_cumsum[:, :, 2 * tau:] - spike_cumsum[:, :, tau:-tau]
            from_state = (from_state > medians) * 1
            to_state = (to_state > medians) * 1

            Phi_R_matrix = np.zeros((n_prefs, n_prefs))
            for i_pref in range(n_prefs):
                for j_pref in range(i_pref):
                    from_bin = from_state[i_pref] + from_state[j_pref] * 2 
                    to_bin = to_state[i_pref] + to_state[j_pref] * 2     
                    transition_hist = np.bincount((from_bin * 4 + to_bin).ravel(), minlength = 16)
                    tpm = transition_hist.reshape((4, 4)).astype(float)
                    row_sum = np.sum(tpm, axis = 1)
                    from_prob = row_sum / np.sum(row_sum)
                    tpm[row_sum != 0] /= row_sum[row_sum != 0, np.newaxis]
                    tpm[row_sum == 0] += 0.25
                    tpm[tpm < 0] = 0
                    tpm[tpm > 1] = 1

                    p_simu = from_prob.reshape((4, 1)) * tpm
                    whole_MI = mutual_information(p_simu)
                    p_simu_4d = p_simu.reshape((2, 2, 2, 2))
                    p_simu_XX = p_simu_4d.sum(axis = (1, 3))
                    part_MI_X = mutual_information(p_simu_XX)
                    p_simu_YY = p_simu_4d.sum(axis = (0, 2))
                    part_MI_Y = mutual_information(p_simu_YY)
                    p_simu_XY = p_simu_4d.sum(axis = (1, 2))
                    cross_MI_XY = mutual_information(p_simu_XY)
                    p_simu_YX = p_simu_4d.sum(axis = (0, 3))
                    cross_MI_YX = mutual_information(p_simu_YX)

                    min_MI = min(part_MI_X, part_MI_Y, cross_MI_XY, cross_MI_YX)
                    Phi_R = whole_MI - part_MI_X - part_MI_Y + min_MI
                    Phi_R_matrix[i_pref, j_pref] = Phi_R_matrix[j_pref, i_pref] = Phi_R
            np.save(f'{exp_dir}/pairwise_Phi_R/tau={tau}ms/session{i_session+1:03d}', Phi_R_matrix)
    
    n_workers = 34
    with Pool(n_workers) as p:
        p.map(calculate_pairwise_Phi_R, range(n_sessions))
    
    # For complex extraction, we utilized the original MATLAB source (https://github.com/JunKitazono/BidirectionallyConnectedCores.git)
    
    Phi_R_mc = np.load(f'{exp_dir}/Phi_R_mc_multitemporal.npy')
    coreness = np.load(f'{exp_dir}/coreness_multitemporal.npy')
    inside_main_complex = np.load(f'{exp_dir}/inside_main_complex_multitemporal.npy')

    Phi_R_mc /= n_prefs * (n_prefs - 1) / 2
    maximal_Phi_R_mc = np.max(Phi_R_mc, axis = 0)
    maximal_scale = np.argmax(Phi_R_mc, axis = 0)

    coreness = coreness[maximal_scale, np.arange(n_sessions), :]
    inside_main_complex = inside_main_complex[maximal_scale, np.arange(n_sessions), :]

    np.save(f'{exp_dir}/maximal_Phi_R_mc', maximal_Phi_R_mc)
    np.save(f'{exp_dir}/coreness', coreness)
    np.save(f'{exp_dir}/inside_main_complex', inside_main_complex)

    coreness /= n_prefs * (n_prefs - 1) / 2
    main_complex_size = np.sum(inside_main_complex, axis = 1)
    main_complex_size = main_complex_size.astype(float)
    main_complex_size /= n_prefs

    Phi_R_mc_all.append(Phi_R_mc)
    maximal_Phi_R_mc_all.append(maximal_Phi_R_mc)
    maximal_scale_all.append(maximal_scale)
    mean_coreness_all.append(np.mean(coreness, axis = 1))
    main_complex_size_all.append(main_complex_size)
    
    neuronal_responses = neuronal_responses[:, :, preferring_electrodes]
    response_iqr_for_el = np.zeros((n_sessions, n_prefs))
    for i_state in range(1 << n_sources):
        response_iqr_for_el += iqr(neuronal_responses[:, source_states == i_state, :], axis = 1)
    response_iqr_for_el /= (1 << n_sources)
    mean_response_iqr = np.mean(response_iqr_for_el, axis = 1)
    mean_response_iqr_all.append(mean_response_iqr)
    response_iqr_inside = np.sum(np.multiply(response_iqr_for_el, inside_main_complex), axis = 1) / np.sum(inside_main_complex, axis = 1)
    response_iqr_outside = np.sum(np.multiply(response_iqr_for_el, 1 - inside_main_complex), axis = 1) / np.sum(1 - inside_main_complex, axis = 1)
    response_iqr_inside_all.append(response_iqr_inside)
    response_iqr_outside_all.append(response_iqr_outside)

maximal_Phi_R_mc_all = np.array(maximal_Phi_R_mc_all)
plot_session_transitions(maximal_Phi_R_mc_all, r'$\Phi_{R}^{mc}$', f'{output_path}/Fig4c', Phi_R_color, sheet_name = 'Fig4c', excel_value_name = 'Phi_R_mc')

Phi_R_mc_all = np.array(Phi_R_mc_all)
ax = formatted_ax(cmap = 'viridis', n_palette = len(temporal_scales))

fig4d_df = pd.DataFrame()
fig4d_df['session #'] = np.arange(1, n_sessions + 1)
for i_tau in range(7):

    tau = temporal_scales[i_tau]
    X = Phi_R_mc_all[:, i_tau, :].copy()
    X -= np.outer(X[:, 0], np.ones(n_sessions))

    mean = np.mean(X, axis = 0)
    
    fig4d_df[f'tau={tau}ms'] = mean

    ax.plot(np.arange(1, n_sessions + 1), mean, linewidth = 0.8, label = rf'$\tau$={tau}ms')
ax.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
ax.set_xlabel('Session #', size = 18)
ax.set_ylabel(r'$\Phi_{R}^{mc}$ change', size = 18)
ax.set_xlim(0, n_sessions + 1)
plt.savefig(f'{output_path}/Fig4d.pdf', dpi = 1200)
plt.close()

with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    fig4d_df.to_excel(writer, sheet_name = 'Fig4d', index = False)

mean_coreness_all = np.array(mean_coreness_all)
main_complex_size_all = np.array(main_complex_size_all)
plot_session_transitions(main_complex_size_all - main_complex_size_all[:, 0:1], 'Change of main-complex size \n relative to the whole graph', f'{output_path}/Fig4e', 'black', plot_change = False, plot_significance = False, sheet_name = 'Fig4e', excel_value_name = 'main_complex_size')

ax = formatted_ax()
rhos_Phi_size = []
for i_exp in range(n_exps):
    rho, _ = st.spearmanr(maximal_Phi_R_mc_all[i_exp], main_complex_size_all[i_exp])
    rhos_Phi_size.append(rho)
    ax.scatter(maximal_Phi_R_mc_all[i_exp], main_complex_size_all[i_exp], s = 1)
ax.set_xlabel(r'$\Phi_{R}^{mc}$', size = 18)
ax.set_ylabel('Main-complex size \n relative to the whole graph', size = 18)
ax.set_xscale('log')
plt.savefig(f'{output_path}/Fig4f.pdf', dpi= 1200)
plt.close()
fig4f_df = pd.DataFrame()
fig4f_df['experiment #'] = np.repeat(np.arange(1, n_exps + 1), n_sessions)
fig4f_df['session #'] = np.tile(np.arange(1, n_sessions + 1), n_exps)
fig4f_df['Phi_R_mc'] = maximal_Phi_R_mc_all.ravel()
fig4f_df['main_complex_size'] = main_complex_size_all.ravel()
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    fig4f_df.to_excel(writer, sheet_name = 'Fig4f', index = False)
forest_plot(rhos_Phi_size, f'{output_path}/Fig4g', sheet_name = 'Fig4g')

mean_response_iqr_all = np.array(mean_response_iqr_all)
ax = formatted_ax()
rhos_IQR_coreness = []
for i_exp in range(n_exps):
    rho, _ = st.spearmanr(mean_response_iqr_all[i_exp], mean_coreness_all[i_exp])
    rhos_IQR_coreness.append(rho)
    ax.scatter(mean_response_iqr_all[i_exp], mean_coreness_all[i_exp], s = 1)
ax.set_xlabel('Mean response IQR', size = 18)
ax.set_ylabel('Mean coreness', size = 18)
ax.set_yscale('log')
plt.savefig(f'{output_path}/Fig4h.pdf', dpi = 1200)
plt.close()
fig4h_df = pd.DataFrame()
fig4h_df['experiment #'] = np.repeat(np.arange(1, n_exps + 1), n_sessions)
fig4h_df['session #'] = np.tile(np.arange(1, n_sessions + 1), n_exps)
fig4h_df['mean_resp_iqr'] = mean_response_iqr_all.ravel()
fig4h_df['mean_coreness'] = mean_coreness_all.ravel()
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    fig4h_df.to_excel(writer, sheet_name = 'Fig4h', index = False)
forest_plot(rhos_IQR_coreness, f'{output_path}/Fig4i', sheet_name = 'Fig4i')

response_iqr_inside_all = np.array(response_iqr_inside_all)
response_iqr_outside_all = np.array(response_iqr_outside_all)
ax = formatted_ax()
ax.axis('square')
ax.set_aspect('equal', adjustable = 'box')
for i_exp in range(n_exps):
    ax.scatter(response_iqr_inside_all[i_exp], response_iqr_outside_all[i_exp], s = 1, alpha = 0.8)
ax.set_xlabel('Mean response IQR \n in main complex', size = 18)
ax.set_ylabel('Mean response IQR \n out of main complex', size = 18)
ax.plot(np.arange(2500) / 100, np.arange(2500) / 100 , color = 'pink', alpha = 0.8)
ax.set_xticks(np.arange(5) * 5)
ax.set_yticks(np.arange(5) * 5)
plt.savefig(f'{output_path}/Fig4j.pdf', dpi = 1200)
plt.close()
fig4j_df = pd.DataFrame()
fig4j_df['experiment #'] = np.repeat(np.arange(1, n_exps + 1), n_sessions)
fig4j_df['session #'] = np.tile(np.arange(1, n_sessions + 1), n_exps)
fig4j_df['resp_iqr_inside'] = response_iqr_inside_all.ravel()
fig4j_df['resp_iqr_outside'] = response_iqr_outside_all.ravel()
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    fig4j_df.to_excel(writer, sheet_name = 'Fig4j', index = False)
print(st.wilcoxon(response_iqr_inside_all.ravel(), response_iqr_outside_all.ravel()))
