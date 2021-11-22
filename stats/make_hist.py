import os
import yaml
import pickle
import numpy as np
import torch
import pandas as pd
import pyprind
import seaborn
import matplotlib.pyplot as plt
from stats.plot_utils import get_state_value_info, plt_values
from stats.thresholds import th

np.random.seed(1234)
torch.manual_seed(1234)
rng = np.random.RandomState(1234)

func = np.median

ROOT_DIR = r"."
root_dir_mimic = os.path.join(ROOT_DIR, 'data', 'sepsis_mimiciii')
# Currently hard-coded for the illustrative run1 defined in `config_sepsis.yaml`
root_dir_run = os.path.join(ROOT_DIR, 'results', 'run1')
params = yaml.safe_load(open(os.path.join(root_dir_run, 'config.yaml'), 'r'))  # the used params in the given run directory

print("Loading data ...")
with open(r"./plots/value_data.pkl", "rb") as f:
    data = pickle.load(f)
with open(r"./plots/flag_data.pkl", "rb") as f:
    bokeh = pickle.load(f)
print("Done.")

q_types = ['selected', 'median']
# step_indeces = [-2, -3, -4, -7, -13, -19]
step_indeces = [-19, -13, -7, -4, -3, -2]


# Hist plot of values
plt.rcParams.update({'font.size': 7})
pf = os.path.join(ROOT_DIR, 'plots', 'hist_values')
if not os.path.exists(pf):
    os.mkdir(pf)
else:
    if len(os.listdir(pf)) != 0:  # not empty
        raise ValueError("The folder >> " + pf + " >> already exists and is not empty.")


def bar_plot_batch(v1, v2, time, ax, title):  # v1: nonsurvivors
    nb_bar = len(v1)
    w = 0.4  # so that two bars fits nicely
    t = np.arange(len(time)) + 1
    rects1 = ax.bar(t - w/2, v1, w, label='Non-survivors', color='navy', alpha=1)
    rects2 = ax.bar(t + w/2, v2, w, label='Survivors', color='green', alpha=1)
    ax.set_ylim(0, 1)
    ax.set_xticks(list(t))
    ax.set_xticklabels(time)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0%', '50%', '100%'])
    # ax.set_xlabel('Treatment Value', fontsize=7)
    # ax.set_ylabel(y_string, fontsize=7)
    ax.set_title(title, fontsize=7)
    # leg.get_frame().set_linewidth(0.0)
    # leg.get_frame().set_facecolor('none')  # rm bg of legend

plots = {"V_D": [], "V_R": [], "Q_D": [], "Q_R": []}
for item in plots:
    fig, axs = plt.subplots(1, 3, figsize=(4, 1.6), dpi=300)
    plots[item] = [fig, axs]

for flag_id, flag in enumerate(["red", "yellow", "noflag"]):
    for v in ["V_D", "V_R", "Q_D", "Q_R"]:
        bar_plot_batch(bokeh["nonsurvivors"][v][flag], bokeh["survivors"][v][flag], bokeh["time"], plots[v][1][flag_id], v + " " + flag)

for item in plots:
    plots[item][0].tight_layout()
    seaborn.despine(plots[item][0])
    plots[item][0].savefig(os.path.join(pf, item + ".pdf"))
plt.close("all")



plots = dict()
for typ in q_types:
    fig, full_axs = plt.subplots(2, len(step_indeces), figsize=(8.5, 1.8), dpi=300)
    plots[typ] = {'fig': fig, 'axs': full_axs}
for i, testing_idx in enumerate(step_indeces):
    print("Computing information for test data, step index {0}".format(testing_idx))
    info = get_state_value_info(testing_idx, data)
    for fig_id, typ in enumerate(q_types):
        axs = plots[typ]['axs'][:, i]
        string = "Last Treatment" if testing_idx == - 1 else "-" + str((abs(testing_idx)-1)*4) + "H"
        # vrb = True if testing_idx == step_indeces[-1] else False
        vrb = True
        plt_values(info, type_q=typ, axs=axs, nb_bar=10, y_string=string, x_verbose=vrb)
for typ in q_types:
    file_name = os.path.join(pf, typ + '_full.pdf')
    plots[typ]['fig'].tight_layout()
    seaborn.despine(plots[typ]['fig'])
    plots[typ]['fig'].savefig(file_name)
plt.close("all")



info = {"survivors": {}, "nonsurvivors": {}}
nonsurvivor_trajectories = sorted(data[data.category == -1].traj.unique().tolist())
survivor_trajectories = sorted(data[data.category == +1].traj.unique().tolist())
for jj, trajectories in enumerate([nonsurvivor_trajectories, survivor_trajectories]):
    if jj == 0:
        traj_type = "nonsurvivors"
        print("----- Non-survivors")
    else:
        traj_type = "survivors"
        print("+++++ Survivors")
    dn_q_traj = []
    rn_q_traj = []
    dn_q_selected_traj = []
    rn_q_selected_traj = []
    a_traj = []
    bar = pyprind.ProgBar(len(trajectories))
    for traj in trajectories:
        bar.update()
        d = data[data.traj == traj]
        a_traj.append(d.a.tolist())
        dn_q_traj.append(np.array(d.q_dn.tolist(), dtype=np.float32))
        rn_q_traj.append(np.array(d.q_rn.tolist(), dtype=np.float32))
        dn_q_selected_action = [d.q_dn.tolist()[t][d.a.tolist()[t]] for t in range(d.q_dn.shape[0])] 
        dn_q_selected_traj.append(dn_q_selected_action)
        rn_q_selected_action = [d.q_rn.tolist()[t][d.a.tolist()[t]] for t in range(d.q_rn.shape[0])] 
        rn_q_selected_traj.append(rn_q_selected_action)

    info[traj_type]["dn_q_selected_traj"] = dn_q_selected_traj
    info[traj_type]["rn_q_selected_traj"] = rn_q_selected_traj
    info[traj_type]["a_traj"] = a_traj
    info[traj_type]["dn_v_median_traj"] = [np.median(q, axis=1) for q in dn_q_traj]
    info[traj_type]["rn_v_median_traj"] = [np.median(q, axis=1) for q in rn_q_traj]
