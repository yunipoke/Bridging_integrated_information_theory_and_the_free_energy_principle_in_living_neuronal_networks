import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from utils import formatted_ax

ROOT = os.getcwd()
EXP_GLOB = os.path.join(ROOT, 'experiment_*')
output_path = os.path.join(ROOT, 'fig')

n_sources = 2
source_state_color = ['black', 'crimson', 'mediumblue', 'limegreen']

psth_all = []
exp_dirs = sorted(glob.glob(EXP_GLOB))
n_exps = len(exp_dirs)
for exp_id, exp_dir in enumerate(exp_dirs):
    psth = np.load(f'{exp_dir}/derivatives/psth.npy')
    psth_all.append(psth)

psth_all = np.array(psth_all)
psth_mean = np.mean(psth_all, axis = 0)
psth_se = np.std(psth_all, axis = 0, ddof = 1) / np.sqrt(n_exps)

ax = formatted_ax()
artifact = 10
plot_post_stimulus = np.arange(artifact, 1000 - artifact * 2)

for i_state in range(1 << n_sources):
    s1_state = (i_state >> 0) & 1
    s2_state = (i_state >> 1) & 1
    ax.plot(plot_post_stimulus, psth_mean[i_state, plot_post_stimulus], color = source_state_color[i_state], label = rf'$(s^{{(1)}}, s^{{(2)}}) = ({s1_state}, {s2_state})$')
    ax.fill_between(plot_post_stimulus, (psth_mean - psth_se)[i_state, plot_post_stimulus], (psth_mean + psth_se)[i_state, plot_post_stimulus], color = source_state_color[i_state], alpha = 0.2)

ax.axvspan(xmin = 10, xmax = 300, alpha = 0.2, color = 'orange')
ax.set_xlabel('Time after stimulus [ms]', size = 18)
ax.set_ylabel('Spike count [spike/ms]', size = 18)
plt.savefig(f'{output_path}/Fig2b.pdf', dpi = 1200)
plt.close()

fig2b_df = pd.DataFrame()
fig2b_df['post_stim'] = np.arange(1000)
for i_state in range(1 << n_sources):
    s1_state = (i_state >> 0) & 1
    s2_state = (i_state >> 1) & 1
    fig2b_df[f'spike_mean{s1_state}{s2_state}'] = psth_mean[i_state, :]
    fig2b_df[f'spike_se{s1_state}{s2_state}'] = psth_se[i_state, :]

with pd.ExcelWriter("SourceData.xlsx", mode = 'w') as writer:
    fig2b_df.to_excel(writer, sheet_name = 'Fig2b', index = False)