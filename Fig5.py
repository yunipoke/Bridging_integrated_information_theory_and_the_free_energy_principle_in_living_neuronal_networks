import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from utils import formatted_ax, forest_plot, Z_score_2d, corr_color
from multiprocessing import Pool

ROOT = os.getcwd()
EXP_GLOB = os.path.join(ROOT, 'derivatives/experiment_*')
output_path = os.path.join(ROOT, 'fig')

n_sources = 2
n_sessions = 100
n_trials = 256

exp_dirs = sorted(glob.glob(EXP_GLOB))
n_exps = len(exp_dirs)
VFE_all = []
Bayesian_surprise_all = []
Accuracy_all = []
Phi_R_mc_all = []
Bayesian_surpise_contrast_all = []
coreness_contrast_all = []
for exp_id, exp_dir in enumerate(exp_dirs):
    s1_preferring_electrodes = np.load(f'{exp_dir}/s1_preferring_electrodes.npy')
    s2_preferring_electrodes = np.load(f'{exp_dir}/s2_preferring_electrodes.npy')
    n_s1prefs = len(s1_preferring_electrodes)
    n_prefs = len(s1_preferring_electrodes) + len(s2_preferring_electrodes)
    VFE = np.load(f'{exp_dir}/variational_free_energy.npy')
    VFE = np.sum(VFE, axis = (1, 2))
    VFE_all.append(VFE)
    Bayesian_surprise = np.load(f'{exp_dir}/Bayesian_surprise.npy')
    s1_Bayesian_surprise = np.sum(Bayesian_surprise[:, :, 0], axis = 1)
    s2_Bayesian_surprise = np.sum(Bayesian_surprise[:, :, 1], axis = 1)
    Bayesian_surpise_contrast = (s1_Bayesian_surprise - s2_Bayesian_surprise) / (s1_Bayesian_surprise + s2_Bayesian_surprise)
    Bayesian_surpise_contrast_all.append(Bayesian_surpise_contrast)
    Bayesian_surprise = np.sum(Bayesian_surprise, axis = (1, 2))
    Bayesian_surprise_all.append(Bayesian_surprise)
    Accuracy = np.load(f'{exp_dir}/accuracy.npy')
    Accuracy = np.sum(Accuracy, axis = (1, 2))
    Accuracy_all.append(Accuracy)
    Phi_R_mc = np.load(f'{exp_dir}/Phi_R_mc.npy')
    Phi_R_mc /= n_prefs * (n_prefs - 1) / 2
    Phi_R_mc_all.append(Phi_R_mc)
    coreness = np.load(f'{exp_dir}/coreness.npy')
    coreness /= n_prefs * (n_prefs - 1) / 2
    s1_coreness = np.mean(coreness[:, :n_s1prefs], axis = 1)
    s2_coreness = np.mean(coreness[:, n_s1prefs:], axis = 1)
    coreness_contrast = (s1_coreness - s2_coreness) / (s1_coreness + s2_coreness)
    coreness_contrast_all.append(coreness_contrast)
VFE_all = np.array(VFE_all)
Bayesian_surprise_all = np.array(Bayesian_surprise_all)
Accuracy_all = np.array(Accuracy_all)
Phi_R_mc_all = np.array(Phi_R_mc_all)
Bayesian_surpise_contrast_all = np.array(Bayesian_surpise_contrast_all)
coreness_contrast_all = np.array(coreness_contrast_all)
VFE_Z = Z_score_2d(VFE_all)
Bayesian_surprise_Z = Z_score_2d(Bayesian_surprise_all)
Accuracy_Z = Z_score_2d(Accuracy_all)
Phi_R_mc_Z = Z_score_2d(Phi_R_mc_all)

ax = formatted_ax()
rhos_VFE_Phi = []
corr_sign = []
for i_exp in range(n_exps):
    rho, _ = st.spearmanr(VFE_all[i_exp], Phi_R_mc_all[i_exp])
    rhos_VFE_Phi.append(rho)
    ax.scatter(VFE_all[i_exp], Phi_R_mc_all[i_exp], s = 1)
ax.set_xlabel('Variational free energy', size = 18)
ax.set_ylabel(r'$\Phi_{R}^{mc}$', size = 18)
ax.set_yscale('log')
plt.savefig(f'{output_path}/Fig5a_top.pdf', dpi = 1200)
plt.close()
ax = formatted_ax()
for i_exp in range(n_exps):
    rho = rhos_VFE_Phi[i_exp]
    if rho > 0.3:
        corr_sign.append(1)
    elif rho < -0.3:
        corr_sign.append(-1)
    else:
        corr_sign.append(0)
    ax.scatter(VFE_Z[i_exp], Phi_R_mc_Z[i_exp], s = 1, color = corr_color(rho))
ax.set_xlabel('Variational free energy (Z-score)', size = 18)
ax.set_ylabel(r'$\Phi_{R}^{mc}$ (Z-score)', size = 18)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.set_xticks(np.linspace(-4, 4, 5))
ax.set_yticks(np.linspace(-4, 4, 5))
plt.savefig(f'{output_path}/Fig5a_bottom.pdf', dpi = 1200)
plt.close()
fig5a_df = pd.DataFrame()
fig5a_df['experiment #'] = np.repeat(np.arange(1, n_exps + 1), n_sessions)
fig5a_df['session #'] = np.tile(np.arange(1, n_sessions + 1), n_exps)
fig5a_df['vfe'] = VFE_all.ravel()
fig5a_df['Phi_R_mc'] = Phi_R_mc_all.ravel()
fig5a_df['vfe_z'] = VFE_Z.ravel()
fig5a_df['Phi_R_mc_z'] = Phi_R_mc_Z.ravel()
fig5a_df['corr_sign'] = np.repeat(corr_sign, n_sessions)
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    fig5a_df.to_excel(writer, sheet_name = 'Fig5a', index = False)
forest_plot(rhos_VFE_Phi, f'{output_path}/Fig5b', sheet_name = 'Fig5b')

ax = formatted_ax()
rhos_Bayesian_surprise_Phi = []
corr_sign = []
for i_exp in range(n_exps):
    rho, _ = st.spearmanr(Bayesian_surprise_all[i_exp], Phi_R_mc_all[i_exp])
    rhos_Bayesian_surprise_Phi.append(rho)
    ax.scatter(Bayesian_surprise_all[i_exp], Phi_R_mc_all[i_exp], s = 1)
ax.set_xlabel('Bayesian surprise', size = 18)
ax.set_ylabel(r'$\Phi_{R}^{mc}$', size = 18)
ax.set_yscale('log')
plt.savefig(f'{output_path}/Fig5c_top.pdf', dpi = 1200)
plt.close()
ax = formatted_ax()
for i_exp in range(n_exps):
    rho = rhos_Bayesian_surprise_Phi[i_exp]
    if rho > 0.3:
        corr_sign.append(1)
    elif rho < -0.3:
        corr_sign.append(-1)
    else:
        corr_sign.append(0)
    ax.scatter(Bayesian_surprise_Z[i_exp], Phi_R_mc_Z[i_exp], s = 1, color = corr_color(rho))
ax.set_xlabel('Bayesian surprise (Z-score)', size = 18)
ax.set_ylabel(r'$\Phi_{R}^{mc}$ (Z-score)', size = 18)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.set_xticks(np.linspace(-4, 4, 5))
ax.set_yticks(np.linspace(-4, 4, 5))
plt.savefig(f'{output_path}/Fig5c_bottom.pdf', dpi = 1200)
plt.close()
fig5c_df = pd.DataFrame()
fig5c_df['experiment #'] = np.repeat(np.arange(1, n_exps + 1), n_sessions)
fig5c_df['session #'] = np.tile(np.arange(1, n_sessions + 1), n_exps)
fig5c_df['bs'] = Bayesian_surprise_all.ravel()
fig5c_df['Phi_R_mc'] = Phi_R_mc_all.ravel()
fig5c_df['bs_z'] = Bayesian_surprise_Z.ravel()
fig5c_df['Phi_R_mc_z'] = Phi_R_mc_Z.ravel()
fig5c_df['corr_sign'] = np.repeat(corr_sign, n_sessions)
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    fig5c_df.to_excel(writer, sheet_name = 'Fig5c', index = False)
forest_plot(rhos_Bayesian_surprise_Phi, f'{output_path}/Fig5d', sheet_name = 'Fig5d')

ax = formatted_ax()
rhos_Accuracy_Phi = []
corr_sign = []
for i_exp in range(n_exps):
    rho, _ = st.spearmanr(Accuracy_all[i_exp], Phi_R_mc_all[i_exp])
    rhos_Accuracy_Phi.append(rho)
    ax.scatter(Accuracy_all[i_exp], Phi_R_mc_all[i_exp], s = 1)
ax.set_xlabel('Accuracy', size = 18)
ax.set_ylabel(r'$\Phi_{R}^{mc}$', size = 18)
ax.set_yscale('log')
plt.savefig(f'{output_path}/Fig5e_top.pdf', dpi = 1200)
plt.close()
ax = formatted_ax()
for i_exp in range(n_exps):
    rho = rhos_Accuracy_Phi[i_exp]
    if rho > 0.3:
        corr_sign.append(1)
    elif rho < -0.3:
        corr_sign.append(-1)
    else:
        corr_sign.append(0)
    ax.scatter(Accuracy_Z[i_exp], Phi_R_mc_Z[i_exp], s = 1, color = corr_color(rho))
ax.set_xlabel('Accuracy (Z-score)', size = 18)
ax.set_ylabel(r'$\Phi_{R}^{mc}$ (Z-score)', size = 18)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.set_xticks(np.linspace(-4, 4, 5))
ax.set_yticks(np.linspace(-4, 4, 5))
plt.savefig(f'{output_path}/Fig5e_bottom.pdf', dpi = 1200)
plt.close()
fig5e_df = pd.DataFrame()
fig5e_df['experiment #'] = np.repeat(np.arange(1, n_exps + 1), n_sessions)
fig5e_df['session #'] = np.tile(np.arange(1, n_sessions + 1), n_exps)
fig5e_df['acc'] = Accuracy_all.ravel()
fig5e_df['Phi_R_mc'] = Phi_R_mc_all.ravel()
fig5e_df['acc_z'] = Accuracy_Z.ravel()
fig5e_df['Phi_R_mc_z'] = Phi_R_mc_Z.ravel()
fig5e_df['corr_sign'] = np.repeat(corr_sign, n_sessions)
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    fig5e_df.to_excel(writer, sheet_name = 'Fig5e', index = False)
forest_plot(rhos_Accuracy_Phi, f'{output_path}/Fig5f', sheet_name = 'Fig5f')

ax = formatted_ax()
rhos_bs_coreness= []
for i_exp in range(n_exps):
    ax.scatter(Bayesian_surpise_contrast_all[i_exp], coreness_contrast_all[i_exp], s = 2)
    rho, _ = st.spearmanr(Bayesian_surpise_contrast_all[i_exp], coreness_contrast_all[i_exp])
    rhos_bs_coreness.append(rho)
ax.set_xlabel(r'$s^{{(1)}} - s^{{(2)}}$ Bayesian surprise' + '\ncontrast', size = 18)
ax.set_ylabel(r'$s^{{(1)}} - s^{{(2)}}$ coreness' + '\ncontrast', size = 18)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xticks(np.linspace(-1, 1, 5))
ax.axis('square')
ax.set_aspect('equal', adjustable = 'box')
plt.savefig(f'{output_path}/Fig5g.pdf', dpi = 1200)
plt.close()
fig5g_df = pd.DataFrame()
fig5g_df['experiment #'] = np.repeat(np.arange(1, n_exps + 1), n_sessions)
fig5g_df['session #'] = np.tile(np.arange(1, n_sessions + 1), n_exps)
fig5g_df['bs_contrast'] = Bayesian_surpise_contrast_all.ravel()
fig5g_df['coreness_contrast'] = coreness_contrast_all.ravel()
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    fig5g_df.to_excel(writer, sheet_name = 'Fig5g', index = False)
forest_plot(rhos_bs_coreness, f'{output_path}/Fig5h', sheet_name = 'Fig5h')