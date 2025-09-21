import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import Divider, Size 
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import pandas as pd
import seaborn as sns

plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm' 
plt.rcParams['mathtext.fontset'] = 'cm' 
sns.set_context("paper", 1, {"lines.linewidth": 4})
ax_w_px = 300 
ax_h_px = 250 
fig_dpi = 100
ax_w_inch = ax_w_px / fig_dpi
ax_h_inch = ax_h_px / fig_dpi
margin_inch = 200 / fig_dpi
ax_margin_inch = (margin_inch, margin_inch, margin_inch, margin_inch)
fig_w_inch = ax_w_inch + ax_margin_inch[0] + ax_margin_inch[2] 
fig_h_inch = ax_h_inch + ax_margin_inch[1] + ax_margin_inch[3]
ax_p_w = [Size.Fixed(ax_margin_inch[0]),Size.Fixed(ax_w_inch)]
ax_p_h = [Size.Fixed(ax_margin_inch[1]),Size.Fixed(ax_h_inch)]

def pvalue_asterisk(pvalue):
    if pvalue >= 0.05:
        return 'n.s.'
    elif pvalue >= 0.01:
        return '*'
    elif pvalue >= 0.005:
        return '**'
    elif pvalue >= 0.001:
        return '***'
    else:
        return '****'

def data_to_axes(x, y, ax):
    return ax.transAxes.inverted().transform(ax.transData.transform((x, y)))
def formatted_ax(n_palette=27):
    sns.set()
    sns.set_palette('husl', n_palette)
    fig = plt.figure( dpi=fig_dpi, figsize=(fig_w_inch, fig_h_inch))
    divider = Divider(fig, (0.0, 0.0, 1.0, 1.0), ax_p_w, ax_p_h, aspect=False)
    ax = Axes(fig, divider.get_position())
    ax.set_axes_locator(divider.new_locator(nx=1,ny=1))
    fig.add_axes(ax)
    ax.tick_params(axis='both', which='major', labelsize = 16)
    return ax


def forest_plot(rho, save_name, n=100, labels=None, show_summary=True, excel_save = True, excel_name = 'SourceData.xlsx', sheet_name = ''):

    rho = np.asarray(rho, dtype=float)
    K = len(rho)

    if np.isscalar(n):
        n = np.full(K, n, dtype=int)
    n = np.asarray(n)

    z   = 0.5 * np.log((1 + rho) / (1 - rho))
    se  = np.sqrt((1 + rho**2 / 2) / (n - 3))
    ci_z = np.vstack([z - 1.96*se, z + 1.96*se]).T
    ci_r = np.tanh(ci_z)

    var_i = se**2
    Q   = np.sum((1/var_i) * (z - z.mean())**2)
    dfQ = K - 1
    I2  = max(0.0, (Q - dfQ) / Q) * 100.0
    tau2 = max(0., (Q - (K-1)) /
                    (np.sum(1/var_i) - np.sum((1/var_i)**2)/np.sum(1/var_i)))
    w_re  = 1 / (var_i + tau2)
    z_re  = np.sum(w_re * z) / np.sum(w_re)
    se_re = np.sqrt(1 / np.sum(w_re))
    ci_re = z_re + np.array([-1.96, 1.96]) * se_re
    rho_re = np.tanh(z_re)
    ci_re_r = np.tanh(ci_re)

    zval   = z_re / se_re
    p_two  = st.norm.sf(abs(zval)) * 2
    p_label = pvalue_asterisk(p_two)
    print(p_two, Q, tau2, I2)

    sigma_bar2 = var_i.mean()
    se_pred    = np.sqrt(tau2 + sigma_bar2)
    pi_z       = z_re + np.array([-1.96, 1.96]) * se_pred
    pi_r       = np.tanh(pi_z)

    df = pd.DataFrame({
        "rho": rho,
        "ci_low": ci_r[:, 0],
        "ci_high": ci_r[:, 1],
        "label": labels if labels is not None else [f"Exp {i+1}" for i in range(K)]
    }).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.3*K + (1 if show_summary else 0.5)))

    ax.axvline(0, linestyle="--", lw=3, color="red")

    ax.errorbar(df["rho"], df.index,
                xerr=[df["rho"]-df["ci_low"], df["ci_high"]-df["rho"]],
                fmt="o", capsize=6, color="black", lw=4, markersize=10)
    

    if show_summary:
        y0 = -1
        ax.errorbar(rho_re, y0,
                    xerr=[[rho_re - ci_re_r[0]], [ci_re_r[1] - rho_re]],
                    fmt="D", markersize=15, mfc="blue", mec="blue",
                    capsize=7, lw=4, color = 'blue', zorder = 3)

        ax.errorbar(rho_re, y0,
            xerr=[[rho_re-pi_r[0]], [pi_r[1]-rho_re]],
            fmt="none", color="skyblue",
            capsize=8, alpha=0.5, lw = 7, zorder = 2)

        ax.text(-1.1-0.05, y0, "Overall",
                va="center", ha="right", size=18)

        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min, x_max * 1.15)  
        ax.text(1.1+0.05, y0, '****', c='white', va="center", ha="left", size=24)
        ax.text(1.1+0.05, y0-0.3, p_label, va="center", ha="left", size=24)

    ax.set_xlim(-1.1, 1.1)
    ax.set_yticks(df.index)
    ax.set_yticklabels(df["label"], size=16)
    ax.set_xlabel(r"Spearman $\rho$", size=24)
    ax.set_xticks(np.linspace(-1, 1, 5))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize = 16)
    plt.tight_layout()

    plt.savefig(f'{save_name}.pdf', dpi=1200)
    plt.close()
    
    if excel_save:
        df = pd.DataFrame()
        df['experiment #'] = np.arange(1, K + 1)
        df['rho'] = rho
        df['ci_low'] = ci_r[:, 0]
        df['ci_high'] = ci_r[:, 1]
        df['overall_rho'] = np.full(K, np.nan)
        df.loc[0, 'overall_rho'] = rho_re
        df['overall_ci_low'] = np.full(K, np.nan)
        df.loc[0, 'overall_ci_low'] = ci_re_r[0]
        df['overall_ci_high'] = np.full(K, np.nan)
        df.loc[0, 'overall_ci_high'] = ci_re_r[1]
        df['overall_pi_low'] = np.full(K, np.nan)
        df.loc[0, 'overall_pi_low'] = pi_r[0]
        df['overall_pi_high'] = np.full(K, np.nan)
        df.loc[0, 'overall_pi_high'] = pi_r[1]
        with pd.ExcelWriter(excel_name, mode = 'a', if_sheet_exists = 'replace') as writer:
            df.to_excel(writer, sheet_name = sheet_name, index = False)
    


def logit(x):
    return np.log(x/(1-x))

def sig(x):
    return 1/(1+np.exp(-x))

def plot_session_transitions(X, value_name, save_name, color, plot_change = True, plot_significance = True, save_excel = True, excel_name = 'SourceData.xlsx', sheet_name = '', excel_value_name = ''):

    assert X.ndim == 2
    n_exps, n_obs = X.shape
    
    _, p_value = st.wilcoxon(X[:, 0], X[:, -1])

    Y = X.copy()
    
    if plot_change:
        Y -= np.outer(Y[:, 0], np.ones(100))

    mean = np.mean(Y, axis = 0)
    se = np.std(Y, axis = 0, ddof = 1) / np.sqrt(n_exps)

    ax = formatted_ax()
    ax.plot(np.arange(1, n_obs + 1), mean, linewidth = 1.5, color = color)
    ax.fill_between(np.arange(1, n_obs + 1), mean - se, mean + se, alpha = 0.2, color = color)
    ax.set_xlabel('Session #', size = 18)
    ylabel_name = value_name
    if plot_change:
        ylabel_name += ' change'
    ax.set_ylabel(ylabel_name, size = 18)
    
    if plot_change and plot_significance:
        ax.axhline(y = 0, color = 'r', linestyle = '--', linewidth = 1)
        ymin = data_to_axes(0, mean[-1], ax)[1]
        ymax = data_to_axes(0, 0, ax)[1]
        ax.axvline(x = n_obs + 2, ymin = ymin, ymax = ymax, color = 'black', linestyle = '-', linewidth = 2, solid_capstyle = 'butt', clip_on = False)
        
        pvalue_label = pvalue_asterisk(p_value)
        
        y_text = mean[-1] / 2
        x_text = 5 + n_obs
        ax.text(x_text, y_text, pvalue_label, ha='left', va='bottom', fontsize = 18)
        
    ax.set_xlim(0, n_obs + 1)
    plt.savefig(f'{save_name}.pdf', dpi = 1200)
    plt.close()
    
    if save_excel:
        df = pd.DataFrame()
        df['session #'] = np.arange(1, n_obs + 1)
        df[f'{excel_value_name}_mean'] = mean
        df[f'{excel_value_name}_se'] = se
        with pd.ExcelWriter(excel_name, mode = 'a', if_sheet_exists = 'replace') as writer:
            df.to_excel(writer, sheet_name = sheet_name, index = False)

def iqr(X, axis = 0):
    q1 = np.percentile(X, 25, axis = axis)
    q3 = np.percentile(X, 75, axis = axis)
    return q3 - q1

def mutual_information(p_xy):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        p_x = p_xy.sum(axis = 1, keepdims = True)  
        p_y = p_xy.sum(axis = 0, keepdims = True)  
        ratio = p_xy / (p_x * p_y)
        ratio[ratio == 0] = 1
        mi = np.nansum(p_xy * np.log(ratio))
    return mi
# def mutual_information(p_xy):
#     p_x = p_xy.sum(axis = 1, keepdims = True)
#     p_y = p_xy.sum(axis = 0, keepdims = True)
#     valid = p_xy > 0
#     return np.sum(p_xy[valid] * np.log(p_xy[valid] / (p_x * p_y)[valid]))

def Z_score_2d(X):
    mean = X.mean(axis = 1, keepdims = True)
    std = X.std(axis = 1, keepdims = True)
    return (X - mean) / std

def corr_color(rho):
    if rho > 0.3:
        return 'red'
    elif rho < -0.3:
        return 'blue'
    else:
        return 'black'