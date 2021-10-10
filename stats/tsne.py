import os
import yaml
import pickle
import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn
import matplotlib.pyplot as plt
from stats.thresholds import th

np.random.seed(456)
torch.manual_seed(456)

func = np.median

ROOT_DIR = r"."
root_dir_mimic = os.path.join(ROOT_DIR, 'data', 'sepsis_mimiciii')
# Currently hard-coded for the illustrative run1 defined in `config_sepsis.yaml`
root_dir_run = os.path.join(ROOT_DIR, 'results', 'run1')
params = yaml.safe_load(open(os.path.join(root_dir_run, 'config.yaml'), 'r'))  # the used params in the given run directory

p = os.path.join(ROOT_DIR, "plots")
if not os.path.exists(p):
    os.mkdir(p)

p_tsne = os.path.join(ROOT_DIR, "plots", "tsne")
if os.path.exists(p_tsne):
    raise ValueError("The folder >> " + p + " >> already exists.")
else:
    os.mkdir(p_tsne)
    os.mkdir(os.path.join(p_tsne, "tsne_dn"))
    os.mkdir(os.path.join(p_tsne, "tsne_rn"))

print("Loading data ...")
sepsis_raw_data = pd.read_csv(os.path.join(root_dir_mimic, 'sepsis_final_data_K1_RAW.csv'))
with open(r"./plots/value_data.pkl", "rb") as f:
    data = pickle.load(f)
print("Done.")

neg_trajs = data[data.category == -1].traj.unique()
pos_trajs = data[data.category == 1].traj.unique()

pos_indecies = data[data.category == 1].index.to_numpy()  # k in pos_indecies == data.index & row: states/transformed_states
neg_indecies = data[data.category == -1].index.to_numpy()

q_dn = data['q_dn'].to_numpy().tolist()
v_dn = np.array([func(q) for q in q_dn], dtype=np.float32)
v_dn = np.clip(v_dn, -1, 0)

q_rn = data['q_rn'].to_numpy().tolist()
v_rn = np.array([func(q) for q in q_rn], dtype=np.float32)
v_rn = np.clip(v_rn, 0, 1)

print("making PCA and tSNE ...")
states = np.array(data["s"].tolist(), dtype=np.float32)
pca_state_model = PCA(n_components=20, random_state=1).fit(states)
transformed_states = pca_state_model.transform(states)
transformed_states = TSNE(n_components=2).fit_transform(transformed_states)

# Plots
### D-Network
# dot_size = 6
# fig = plt.figure(figsize = (5.8, 2.7), dpi=300)
dot_size = 2
fig = plt.figure(figsize = (3, 1.5), dpi=300)
ax1 = fig.add_subplot(1, 2, 1) 
ax1.set_title('D-Network', fontsize=7)
ax1.scatter(transformed_states[:, 0] , transformed_states[:, 1], c=v_dn, marker='.', s=dot_size, lw=0, cmap='plasma', vmin=-1, vmax=0)
ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# v1 = np.linspace(-1, 0, 4, endpoint=True)
# cb = plt.colorbar(ticks=v1, ax=ax1)
# cb.set_label('D-Network Values', size=7, rotation=270)
# for l in cb.ax.yaxis.get_ticklabels():
#     l.set_size(7)
# cb.ax.get_yaxis().labelpad = 10

vv = np.where(v_dn < -0.8)[0]  # re-plotting low values to be seen on the top
ax1.scatter(transformed_states[vv, 0] , transformed_states[vv, 1], c=v_dn[vv], marker='.', s=dot_size, lw=0, cmap='plasma', vmin=-1, vmax=0)

### R-Network
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('R-Network', fontsize=7)
ax2.scatter(transformed_states[:, 0] , transformed_states[:, 1], c=v_rn, marker='.', s=dot_size, lw=0, cmap='plasma', vmin=0, vmax=1)
ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

vv = np.where(v_rn < 0.2)[0]  # replotting low values to be seen on the top
ax2.scatter(transformed_states[vv, 0] , transformed_states[vv, 1], c=v_rn[vv], marker='.', s=dot_size, lw=0, cmap='plasma', vmin=0, vmax=1)
fig.tight_layout()
seaborn.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
seaborn.despine(ax=ax2, top=True, right=True, left=True, bottom=True)
fig.subplots_adjust(wspace=0.3)
plt.savefig(os.path.join(p_tsne, 'tsne_n6220p6220_cliped_median.png'), dpi=300)
print("Done.")


#----------------------------------------------

def full_plot(v_full, vmin, vmax, colorbar=False):
    dot_size = 2
    fig2 = plt.figure(figsize = (2.1, 2.1), dpi=300)
    ax10 = fig2.add_subplot(1, 1, 1) 
    # ax10.set_title('Complete', fontsize=7)
    p = ax10.scatter(transformed_states[:, 0] , transformed_states[:, 1], c=v_full, marker='.', s=dot_size, lw=0, cmap='plasma', vmin=vmin, vmax=vmax)
    ax10.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    # vv = np.where(v_full < -0.8)[0]  # re-plotting low values to be seen on the top
    # ax10.scatter(transformed_states[vv, 0] , transformed_states[vv, 1], c=v_full[vv], marker='.', s=dot_size, lw=0, cmap='plasma', vmin=-1, vmax=0)
    if colorbar:
        vf1 = np.linspace(-1, 1, 5, endpoint=True)
        cb = plt.colorbar(p, ticks=vf1, ax=ax10)
        # # cb.set_label('D-Network Values', size=7, rotation=270)
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_size(7)
        cb.ax.get_yaxis().labelpad = 8
    fig2.tight_layout(pad=0.0)
    seaborn.despine(ax=ax10, top=True, right=True, left=True, bottom=True)
    return fig2, ax10


dist = lambda x, y: (x[0] - y[0])**2 + (x[1] - y[1])**2
def traj_plot(traj, ax, flag=True, jump_th=100, label=True):
    indeces = data[data.traj == traj].index.to_numpy()
    ts = transformed_states[indeces, :]
    if flag:  # plot traj
        ax.plot(ts[:, 0], ts[:, 1], 'o-', lw=1, color='black', alpha=0.3, markerfacecolor='none', markersize=4)
    ax.plot(ts[0, 0], ts[0, 1], '', marker='x', color='black', alpha=0.6, markersize=7, markeredgewidth=1)
    ax.plot(ts[-1, 0], ts[-1, 1], '', marker='s', color='black', markerfacecolor='none', alpha=0.6, markersize=7, markeredgewidth=1)
    if label:
        dt = np.array([dist(ts[k], ts[k+1]) for k in range(len(ts)-1)])
        dtw = np.where(dt > jump_th)[0]  # where a jump happens in the tsne space
        for i in dtw:
            if ts[i][0] > ts[i+1][0]:
                x, y = ts[i+1][0] + 2, ts[i+1][1] + 4
            else:
                x, y = ts[i+1][0] - 6, ts[i+1][1] + 4
            ax.text(x, y, str(i+1), fontsize=7)
        ax.text(ts[0, 0] - 2, ts[0, 1] - 10, "0", fontsize=7)

good_neg_trajs = []
for traj in neg_trajs:
    q_dn_traj = data[data.traj == traj]['q_dn'].tolist()
    v_dn_traj = np.array([func(q) for q in q_dn_traj], dtype=np.float32)
    q_rn_traj = data[data.traj == traj]['q_rn'].tolist()
    v_rn_traj = np.array([func(q) for q in q_rn_traj], dtype=np.float32)
    if len(v_dn_traj) <= 7:  # should be at least 24 hours
        continue
    if all(v_dn_traj[-4:] < th.dn_red) and all(v_rn_traj[-4:] < th.rn_red):  # dead-end at last 12 hours
        good_neg_trajs.append(int(traj))
print("good neg traj for plotting:")
print(good_neg_trajs)
with open(os.path.join(p, "good_neg_trajs.yaml"), "w") as y:
    yaml.safe_dump({"selected_trajs_idx": good_neg_trajs}, y)


v_full = v_dn  # if want to use v_dn+v_rn --> change the vmin / vmax to -1/+1 accordingly
vmin = -1
vmax = 0
for traj in good_neg_trajs:
    fig, ax = full_plot(v_full=v_full, vmin=vmin, vmax=vmax)
    traj_plot(traj, ax, label=True)
    fig.savefig(os.path.join(p_tsne, 'tsne_dn', 'tsne_dn'+str(traj)+'.png'), dpi=300)
    plt.close("all")

v_full = v_rn 
vmin = 0
vmax = 1
for traj in good_neg_trajs:
    fig, ax = full_plot(v_full=v_full, vmin=vmin, vmax=vmax)
    traj_plot(traj, ax, label=True)
    fig.savefig(os.path.join(p_tsne, 'tsne_rn', 'tsne_rn'+str(traj)+'.png'), dpi=300)
    plt.close("all")


# ## grabbing intersting regions of the tsne plot
# x = np.where(np.logical_and.reduce((transformed_states[:, 0] < -30, transformed_states[:, 0] > -75, \
#              transformed_states[:, 1] < -10, transformed_states[:, 1] > -51)))[0]
