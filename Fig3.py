import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from utils import formatted_ax, forest_plot, sig, logit, plot_session_transitions, iqr

ROOT = os.getcwd()
EXP_GLOB = os.path.join(ROOT, "experiment_*")
output_path = os.path.join(ROOT, 'fig')

n_sources = 2
n_sessions = 100
n_trials = 256
n_obs = 32
lambda_value = 3000
eps = 1e-6

vfe_color = 'lightseagreen'
bs_color = 'dodgerblue'
acc_color = 'orangered'
source_state_color = ['black', 'crimson', 'mediumblue', 'limegreen']

exp_dirs = sorted(glob.glob(EXP_GLOB))
n_exps = len(exp_dirs)
VFE_all = []
Bayesian_surprise_all = []
Accuracy_all = []
states_Bayesian_surprise_all = []
response_iqr_all = []
for exp_id, exp_dir in enumerate(exp_dirs):
    neuronal_responses = np.load(f'{exp_dir}/derivatives/neuronal_responses.npy')
    s1_preferring_electrodes = np.load(f'{exp_dir}/derivatives/s1_preferring_electrodes.npy')
    s2_preferring_electrodes = np.load(f'{exp_dir}/derivatives/s2_preferring_electrodes.npy')
    source_states = np.load(f'{exp_dir}/derivatives/hidden_source_states.npy')
    observations = np.load(f'{exp_dir}/derivatives/observations.npy')
    source_states = source_states[:, 0] * 1 + source_states[:, 1] * 2
    n_channels = neuronal_responses.shape[2]

    ## The analysis script for FEP-related quantities was created, referencing the original source (https://github.com/takuyaisomura/reverse_engineering.git)

    ensemble_responses = np.zeros((n_sessions, n_trials, n_sources))
    ensemble_responses[:, :, 0] = np.mean(neuronal_responses[:, :, s1_preferring_electrodes], axis = 2)
    ensemble_responses[:, :, 1] = np.mean(neuronal_responses[:, :, s2_preferring_electrodes], axis = 2)
    x = ensemble_responses.copy()
    for i_state in range(1 << n_sources):
        x_init = np.mean(x[0, source_states == i_state, :], axis = 0)
        x[:, source_states == i_state, :] -= x_init
    x_trend = np.mean(x, axis = 1)
    x_trend -= x_trend[0, :]
    x -= x_trend[:, None, :]
    x_mean = np.mean(x, axis = (0, 1))
    x_std = np.std(x, axis = (0, 1))
    x -= x_mean
    x /= x_std
    x = x / 2 + 1 / 2
    x = np.clip(x, 0, 1)

    phi1 = np.zeros((n_sessions, n_sources))
    phi0 = np.zeros((n_sessions, n_sources))
    initial_sessions = 10
    phi1[:initial_sessions, :] = np.log(np.mean(x[0:initial_sessions, :, :], axis = (0, 1)))
    phi1[initial_sessions:, :] = np.log(np.mean(x[initial_sessions - 1:-1, :, :], axis = 1))
    phi0[:initial_sessions, :] = np.log(1 - np.mean(x[0:initial_sessions, :, :], axis = (0, 1)))
    phi0[initial_sessions:, :] = np.log(1 - np.mean(x[initial_sessions - 1:-1, :, :], axis = 1))
    w1 = np.zeros((n_sessions, n_sources, n_obs))
    w0 = np.zeros((n_sessions, n_sources, n_obs))
    hebb1 = np.ones((n_sources, n_obs)) * lambda_value / 2
    hebb0 = np.ones((n_sources, n_obs)) * lambda_value / 2
    home1 = np.ones((n_sources, n_obs)) * lambda_value
    home0 = np.ones((n_sources, n_obs)) * lambda_value

    VFE = np.zeros((n_sessions, n_trials, n_sources))
    Bayesian_surprise = np.zeros((n_sessions, n_trials, n_sources))
    Accuracy = np.zeros((n_sessions, n_trials, n_sources))
    for i_session in range(n_sessions):
        w1[i_session] = logit(hebb1 / home1)
        w0[i_session] = logit(hebb0 / home0)
        hebb1 = hebb1 + np.dot(x[i_session].T, observations)
        hebb0 = hebb0 + np.dot((1 - x[i_session]).T, observations)
        home1 = home1 + np.dot(x[i_session].T, np.ones((n_trials, n_obs)))
        home0 = home0 + np.dot((1 - x[i_session]).T, np.ones((n_trials, n_obs)))

        x_stack = np.vstack((x[i_session].T, (1 - x[i_session]).T))
        o_stack = np.vstack((observations.T, (1 - observations).T))
        w_stack = np.hstack((np.vstack((sig(w1[i_session]), sig(w0[i_session]))), np.vstack((1 - sig(w1[i_session]), 1 - sig(w0[i_session])))))
        Bayesian_surprise_mat = np.multiply(x_stack, np.log(x_stack + eps) - np.stack((phi1[i_session], phi0[i_session])).reshape((4, 1)))
        Bayesian_surprise[i_session, :, 0] = Bayesian_surprise_mat[0, :] + Bayesian_surprise_mat[2, :]
        Bayesian_surprise[i_session, :, 1] = Bayesian_surprise_mat[1, :] + Bayesian_surprise_mat[3, :]
        Accuracy_mat = np.multiply(x_stack, np.dot(np.log(w_stack + eps), o_stack))
        Accuracy[i_session, :, 0] = Accuracy_mat[0, :] + Accuracy_mat[2, :]
        Accuracy[i_session, :, 1] = Accuracy_mat[1, :] + Accuracy_mat[3, :]
        VFE[i_session] = Bayesian_surprise[i_session] - Accuracy[i_session]
    VFE_all.append(VFE)
    Bayesian_surprise_all.append(Bayesian_surprise)
    Accuracy_all.append(Accuracy)
    np.save(f'{exp_dir}/derivatives/variational_free_energy.npy', VFE)
    np.save(f'{exp_dir}/derivatives/Bayesian_surprise.npy', Bayesian_surprise)
    np.save(f'{exp_dir}/derivatives/accuracy.npy', Accuracy)

    states_Bayesian_surprise = np.zeros((n_sessions, 1 << n_sources, n_sources))
    for i_state in range(1 << n_sources):
        states_Bayesian_surprise[:, i_state, :] = np.mean(Bayesian_surprise[:, source_states == i_state, :], axis = 1)
    states_Bayesian_surprise_all.append(states_Bayesian_surprise)
    
    states_response_iqr = np.zeros((n_sessions, 1 << n_sources, n_sources))
    for i_state in range(1 << n_sources):
        states_response_iqr[:, i_state, :] = iqr(ensemble_responses[:, source_states == i_state, :], axis = 1)
    response_iqr = np.mean(states_response_iqr, axis = (1, 2))
    response_iqr_all.append(response_iqr)

VFE_all = np.array(VFE_all)
VFE_session = np.sum(VFE_all, axis = (2, 3))
plot_session_transitions(VFE_session, 'Variational free energy', f'{output_path}/Fig3b', vfe_color, save_excel = True, sheet_name = 'Fig3b', excel_value_name = 'vfe')
Bayesian_surprise_all = np.array(Bayesian_surprise_all)
Bayesian_surprise_session = np.sum(Bayesian_surprise_all, axis = (2, 3))
plot_session_transitions(Bayesian_surprise_session, 'Bayesian surprise', f'{output_path}/Fig3c', bs_color, save_excel = True, sheet_name = 'Fig3c', excel_value_name = 'bs')
Accuracy_all = np.array(Accuracy_all)
Accuracy_session = np.sum(Accuracy_all, axis = (2, 3))
plot_session_transitions(Accuracy_session, 'Accuracy', f'{output_path}/Fig3d', acc_color, save_excel = True, sheet_name = 'Fig3d', excel_value_name = 'acc')

states_Bayesian_surprise_all = np.array(states_Bayesian_surprise_all)
fig3e_df = pd.DataFrame()
fig3e_df['experiment #'] = np.repeat(np.arange(1, n_exps + 1), n_sessions)
fig3e_df['session #'] = np.tile(np.arange(1, n_sessions + 1), n_exps)

for i_state in [1, 2]:
    s1_state = (i_state >> 0) & 1
    s2_state = (i_state >> 1) & 1
    s1_Bayesian_surprise = states_Bayesian_surprise_all[:, :, i_state, 0]
    s2_Bayesian_surprise = states_Bayesian_surprise_all[:, :, i_state, 1]
    fig3e_df[f's1_bs_{s1_state}{s2_state}'] = s1_Bayesian_surprise.ravel()
    fig3e_df[f's2_bs_{s1_state}{s2_state}'] = s2_Bayesian_surprise.ravel()
    ax = formatted_ax()
    ax.scatter(s1_Bayesian_surprise.ravel(), s2_Bayesian_surprise.ravel(), s = 1, alpha = 0.2, color = source_state_color[i_state])
    ax.axis('square')
    ax.set_aspect('equal', adjustable = 'box')
    ax.set_xlabel(r'Bayesian surprise for $s^{{(1)}}$', size = 18)
    ax.set_ylabel(r'Bayesian surprise for $s^{{(2)}}$', size = 18)
    ax.plot(np.arange(10000) / 100, np.arange(10000) / 100 , color = 'pink', alpha = 0.8)
    ax.set_xticks(np.arange(6) / 10)
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_xticks())
    ax.set_xlim(-0.01)
    ax.set_ylim(-0.01)
    ax.set_title(rf'$(s^{{(1)}}, s^{{(2)}}) = ({s1_state}, {s2_state})$', size = 18)
    plt.savefig(f'{output_path}/Fig3e_{s1_state}{s2_state}.pdf', dpi = 1200)
    plt.close()
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    fig3e_df.to_excel(writer, sheet_name = 'Fig3e', index = False)

ax = formatted_ax()
response_iqr_all = np.array(response_iqr_all)
rhos_iqr_bs = []
for i_exp in range(n_exps):
    rho, _ = st.spearmanr(response_iqr_all[i_exp], Bayesian_surprise_session[i_exp])
    rhos_iqr_bs.append(rho)
    ax.scatter(response_iqr_all[i_exp], Bayesian_surprise_session[i_exp], s = 1)
ax.set_xlabel('Neuronal response IQR', size = 18)
ax.set_ylabel('Bayesian surprise', size = 18)
plt.savefig(f'{output_path}/Fig3f.pdf', dpi = 1200)
plt.close()
fig3f_df = pd.DataFrame()
fig3f_df['experiment #'] = np.repeat(np.arange(1, n_exps + 1), n_sessions)
fig3f_df['session #'] = np.tile(np.arange(1, n_sessions + 1), n_exps)
fig3f_df['resp_iqr'] = response_iqr_all.ravel()
fig3f_df['bs'] = Bayesian_surprise_session.ravel()
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    fig3f_df.to_excel(writer, sheet_name = 'Fig3f', index = False)
forest_plot(rhos_iqr_bs, f'{output_path}/Fig3g', sheet_name = 'Fig3g')