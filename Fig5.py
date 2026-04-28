import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from utils import formatted_ax, forest_plot, forest_plot_beta, Z_score_2d, corr_color, med_quan, _jitter, _count_pos, _count_sig, _finite
from multiprocessing import Pool
import statsmodels.api as sm

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
    Phi_R_mc = np.load(f'{exp_dir}/maximal_Phi_R_mc.npy')
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

exit()

t = np.arange(n_sessions) + 1
n_models = 6
beta_bs = np.zeros((n_models, n_exps))
beta_se_bs = np.zeros((n_models, n_exps))
p_bs = np.zeros((n_models, n_exps))
beta_acc = np.zeros((n_models, n_exps))
beta_se_acc = np.zeros((n_models, n_exps))
p_acc = np.zeros((n_models, n_exps))
R2 = np.zeros((n_models, n_exps))
for i_exp in range(n_exps):
    mod_bs = sm.OLS(Phi_R_mc_Z[i_exp], sm.add_constant(np.c_[Bayesian_surprise_Z[i_exp]]))
    mod_acc = sm.OLS(Phi_R_mc_Z[i_exp], sm.add_constant(np.c_[Accuracy_Z[i_exp]]))
    mod_bs_acc = sm.OLS(Phi_R_mc_Z[i_exp], sm.add_constant(np.c_[Bayesian_surprise_Z[i_exp], Accuracy_Z[i_exp]]))
    mod_bs_t = sm.OLS(Phi_R_mc_Z[i_exp], sm.add_constant(np.c_[Bayesian_surprise_Z[i_exp], t]))
    mod_acc_t = sm.OLS(Phi_R_mc_Z[i_exp], sm.add_constant(np.c_[Accuracy_Z[i_exp], t]))
    mod_bs_acc_t = sm.OLS(Phi_R_mc_Z[i_exp], sm.add_constant(np.c_[Bayesian_surprise_Z[i_exp], Accuracy_Z[i_exp], t]))

    fit_bs = mod_bs.fit()
    fit_acc = mod_acc.fit()
    fit_bs_acc = mod_bs_acc.fit()
    fit_bs_t = mod_bs_t.fit()
    fit_acc_t = mod_acc_t.fit()
    fit_bs_acc_t = mod_bs_acc_t.fit()
    
    beta_bs[0, i_exp] = fit_bs.params[1]
    beta_bs[1, i_exp] = None
    beta_bs[2, i_exp] = fit_bs_acc.params[1]
    beta_bs[3, i_exp] = fit_bs_t.params[1]
    beta_bs[4, i_exp] = None
    beta_bs[5, i_exp] = fit_bs_acc_t.params[1]

    beta_se_bs[0, i_exp] = fit_bs.bse[1]
    beta_se_bs[1, i_exp] = None
    beta_se_bs[2, i_exp] = fit_bs_acc.bse[1]
    beta_se_bs[3, i_exp] = fit_bs_t.bse[1]
    beta_se_bs[4, i_exp] = None
    beta_se_bs[5, i_exp] = fit_bs_acc_t.bse[1]

    beta_acc[0, i_exp] = None
    beta_acc[1, i_exp] = fit_acc.params[1]
    beta_acc[2, i_exp] = fit_bs_acc.params[2]
    beta_acc[3, i_exp] = None
    beta_acc[4, i_exp] = fit_acc_t.params[1]
    beta_acc[5, i_exp] = fit_bs_acc_t.params[2]

    beta_se_acc[0, i_exp] = None
    beta_se_acc[1, i_exp] = fit_acc.bse[1]
    beta_se_acc[2, i_exp] = fit_bs_acc.bse[2]
    beta_se_acc[3, i_exp] = None
    beta_se_acc[4, i_exp] = fit_acc_t.bse[1]
    beta_se_acc[5, i_exp] = fit_bs_acc_t.bse[2]

    p_bs[0, i_exp] = fit_bs.pvalues[1]
    p_bs[1, i_exp] = None
    p_bs[2, i_exp] = fit_bs_acc.pvalues[1]
    p_bs[3, i_exp] = fit_bs_t.pvalues[1]
    p_bs[4, i_exp] = None
    p_bs[5, i_exp] = fit_bs_acc_t.pvalues[1]

    p_acc[0, i_exp] = None
    p_acc[1, i_exp] = fit_acc.pvalues[1]
    p_acc[2, i_exp] = fit_bs_acc.pvalues[2]
    p_acc[3, i_exp] = None
    p_acc[4, i_exp] = fit_acc_t.pvalues[1]
    p_acc[5, i_exp] = fit_bs_acc_t.pvalues[2]
    
    R2[0, i_exp] = fit_bs.rsquared
    R2[1, i_exp] = fit_acc.rsquared
    R2[2, i_exp] = fit_bs_acc.rsquared
    R2[3, i_exp] = fit_bs_t.rsquared
    R2[4, i_exp] = fit_acc_t.rsquared
    R2[5, i_exp] = fit_bs_acc_t.rsquared

for i_model in range(n_models):
    print(med_quan(R2[i_model]), med_quan(beta_bs[i_model]), (beta_bs[i_model] > 0).sum(), (p_bs[i_model] < 0.05).sum(), med_quan(beta_acc[i_model]), (beta_acc[i_model] > 0).sum(), (p_acc[i_model] < 0.05).sum())

dR2_no_t = R2[0, :] - R2[1, :]
dR2_t    = R2[3, :] - R2[4, :]

b_bs_no_t = beta_bs[2, :]    
b_acc_no_t = beta_acc[2, :]  
b_bs_t = beta_bs[5, :]       
b_acc_t = beta_acc[5, :]     

ax = formatted_ax()
ax.axhline(0, linewidth = 1)

x0, x1 = 0.0, 1.0
j0 = _jitter(n_exps, seed = 1)
j1 = _jitter(n_exps, seed = 2)

ax.scatter(np.full(n_exps, x0) + j0, dR2_no_t, s = 14, color = 'black')
ax.scatter(np.full(n_exps, x1) + j1, dR2_t, s = 14, color = 'black')

for i in range(n_exps):
    if np.isfinite(dR2_no_t[i]) and np.isfinite(dR2_t[i]):
        ax.plot([x0 + j0[i], x1 + j1[i]], [dR2_no_t[i], dR2_t[i]], linewidth = 0.4, color = 'black')

n_pos0, n0 = _count_pos(dR2_no_t)
n_pos1, n1 = _count_pos(dR2_t)
med0 = np.nanmedian(dR2_no_t)
med1 = np.nanmedian(dR2_t)

ax.text(0.00, -0.20,
        f"no t:  ΔR²>0 = {n_pos0}/{n0},  median={med0:.3f}\n"
        f"+t  :  ΔR²>0 = {n_pos1}/{n1},  median={med1:.3f}",
        transform=ax.transAxes, va="top")

ax.set_xlim(-0.5, 1.5)
ax.set_xticks([0, 1])
ax.set_xticklabels([r"no $t$", r"+ $t$"])
ax.set_ylabel(r"$\Delta R^2 = R^2(\Phi_R^{mc}\sim BS) - R^2(\Phi_R^{mc}\sim Acc)$")
plt.savefig(f'{output_path}/FigS1a.pdf', dpi = 1200)
plt.close()
figS1a_df = pd.DataFrame()
figS1a_df['experiment #'] = np.arange(n_exps)
figS1a_df['dR2_no_t'] = dR2_no_t
figS1a_df['dR2_t'] = dR2_t
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    figS1a_df.to_excel(writer, sheet_name = 'FigS1a', index = False)

ax = formatted_ax('husl', 4)
ax.axhline(0, linewidth = 1)

groups = [
    (r"$\beta_{BS}$ in $\Phi_R^{mc}\sim BS+Acc$", b_bs_no_t, p_bs[2, :]),
    (r"$\beta_{Acc}$ in $\Phi_R^{mc}\sim BS+Acc$", b_acc_no_t, p_acc[2, :]),
    (r"$\beta_{BS}$ in $\Phi_R^{mc}\sim BS+Acc+t$", b_bs_t, p_bs[5, :]),
    (r"$\beta_{Acc}$ in $\Phi_R^{mc}\sim BS+Acc+t$", b_acc_t, p_acc[5, :]),
]

for gi, (label, vals, pvals) in enumerate(groups):
    x = np.full(n_exps, gi, dtype = float) + _jitter(n_exps, seed = 10 + gi)
    ax.scatter(x, vals, s = 14)

ax.set_xlim(-0.6, len(groups)-0.4)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], rotation = 25, ha="right")
ax.set_ylabel("Standardized coefficient")
plt.savefig(f'{output_path}/FigS1b.pdf', dpi = 1200)
plt.close()
figS1b_df = pd.DataFrame()
figS1b_df['experiment #'] = np.arange(n_exps)
figS1b_df['beta_bs_no_t'] = b_bs_no_t
figS1b_df['beta_acc_no_t'] = b_acc_no_t
figS1b_df['beta_bs_t'] = b_bs_t
figS1b_df['beta_acc_t'] = b_acc_t
with pd.ExcelWriter("SourceData.xlsx", mode = 'a', if_sheet_exists = 'replace') as writer:
    figS1b_df.to_excel(writer, sheet_name = 'FigS1b', index = False)

forest_plot_beta(beta_bs[0], beta_se_bs[0], r"$\beta_{BS}$ in $\Phi_R^{mc}\sim BS$", f'{output_path}/FigS2a', sheet_name = 'FigS2a')
forest_plot_beta(beta_bs[2], beta_se_bs[2], r"$\beta_{BS}$ in $\Phi_R^{mc}\sim BS+Acc$", f'{output_path}/FigS2b', sheet_name = 'FigS2b')
forest_plot_beta(beta_bs[3], beta_se_bs[3], r"$\beta_{BS}$ in $\Phi_R^{mc}\sim BS+t$", f'{output_path}/FigS2c', sheet_name = 'FigS2c')
forest_plot_beta(beta_bs[5], beta_se_bs[5], r"$\beta_{BS}$ in $\Phi_R^{mc}\sim BS+Acc+t$", f'{output_path}/FigS2d', sheet_name = 'FigS2d')

forest_plot_beta(beta_acc[1], beta_se_acc[1], r"$\beta_{Acc}$ in $\Phi_R^{mc}\sim Acc$", f'{output_path}/FigS3a', sheet_name = 'FigS3a')
forest_plot_beta(beta_acc[2], beta_se_acc[2], r"$\beta_{Acc}$ in $\Phi_R^{mc}\sim BS+Acc$", f'{output_path}/FigS3b', sheet_name = 'FigS3b')
forest_plot_beta(beta_acc[4], beta_se_acc[4], r"$\beta_{Acc}$ in $\Phi_R^{mc}\sim Acc+t$", f'{output_path}/FigS3c', sheet_name = 'FigS3c')
forest_plot_beta(beta_acc[5], beta_se_acc[5], r"$\beta_{Acc}$ in $\Phi_R^{mc}\sim BS+Acc+t$", f'{output_path}/FigS3d', sheet_name = 'FigS3d')
