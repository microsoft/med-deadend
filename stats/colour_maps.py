import os
import yaml
import pickle
import numpy as np
import torch
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from stats.thresholds import th


np.random.seed(1234)
torch.manual_seed(1234)
rng = np.random.RandomState(1234)
ROOT_DIR = r"."
plt.rcParams.update({'font.size': 7})
func = np.median

p = os.path.join(ROOT_DIR, "plots", "color_plots")
if not os.path.exists(p):
    os.mkdir(p)
else:
    if len(os.listdir(p)) != 0:  # not empty
        raise ValueError("The folder >> " + p + " >> already exists and is not empty.")


root_dir_mimic = os.path.join(ROOT_DIR, 'data', 'sepsis_mimiciii')
# Currently hard-coded for the illustrative run1 defined in `config_sepsis.yaml`
root_dir_run = os.path.join(ROOT_DIR, 'results', 'run1')
params = yaml.safe_load(open(os.path.join(root_dir_run, 'config.yaml'), 'r'))  # the used params in the given run directory

print("Loading data ...")
sepsis_raw_data = pd.read_csv(os.path.join(root_dir_mimic, 'sepsis_final_data_K1_RAW.csv'))
with open(r"./plots/value_data.pkl", "rb") as f:
    data = pickle.load(f)
print("Done.")

states_neg = data[data.category == -1]['s'].to_numpy().tolist()
states_pos = data[data.category == 1]['s'].to_numpy().tolist()
states_pos = states_pos[:len(states_neg)]  # same number as negative
states = [k for k in states_pos + states_neg]
states = np.array(states, dtype=np.float32)

q_dn_neg = data[data.category == -1]['q_dn'].to_numpy().tolist()
v_dn_neg = [func(q) for q in q_dn_neg]
q_dn_pos = data[data.category == 1]['q_dn'].to_numpy().tolist()
v_dn_pos = [func(q) for q in q_dn_pos][:len(states_neg)]  # same number as negative
v_d_network = np.array(v_dn_pos + v_dn_neg, dtype=np.float32)
v_d_network = np.clip(v_d_network, -1, 0)

q_rn_neg = data[data.category == -1]['q_rn'].to_numpy().tolist()
v_rn_neg = [func(q) for q in q_rn_neg]
q_rn_pos = data[data.category == 1]['q_rn'].to_numpy().tolist()
v_rn_pos = [func(q) for q in q_rn_pos][:len(states_neg)]  # same number as negative
v_r_network = np.array(v_rn_pos + v_rn_neg, dtype=np.float32)
v_r_network = np.clip(v_r_network, 0, 1)

def worker(trajs, dn_th, rn_th, nb_traj, plot_file_name, verbose, random=True):
    selected_trajs = []
    selected_trajs_dn = []
    selected_trajs_rn = []
    selected_trajs_a = []
    selected_trajs_q_dn = []
    selected_trajs_q_rn = []
    for traj in trajs:
        d = data[data.traj == traj]
        q_traj_dn = d['q_dn'].to_numpy().tolist()
        v_dn = np.array([func(q) for q in q_traj_dn], dtype=np.float32).clip(-1, 0)
        q_traj_rn = d['q_rn'].to_numpy().tolist()
        v_rn = np.array([func(q) for q in q_traj_rn], dtype=np.float32).clip(0, 1)
        selected_treatments = d['a'].to_numpy().tolist()
        selected_q_dn = [d['q_dn'].to_numpy()[k][a].clip(-1, 0) for (k, a) in enumerate(d['a'])]
        selected_q_rn = [d['q_rn'].to_numpy()[k][a].clip(0, 1) for (k, a) in enumerate(d['a'])]
        if len(v_dn) <= 7:  # should be at least 24 hours
            continue
        if all(v_dn[-4:] < dn_th) and all(v_rn[-4:] < rn_th):  # dead-end at last 12 hours
            if verbose:
                print('confirmed:', traj)
            selected_trajs.append(traj)
            selected_trajs_dn.append(v_dn)
            selected_trajs_rn.append(v_rn)
            selected_trajs_a.append(selected_treatments)
            selected_trajs_q_dn.append(selected_q_dn)
            selected_trajs_q_rn.append(selected_q_rn)
    if random:
        t_indeces = np.random.choice(np.arange(0, len(selected_trajs)), nb_traj, replace=False)
    else:
        t_indeces = np.arange(0, nb_traj)
    print("selected trajs: ", np.array(selected_trajs)[t_indeces])
    selected_trajs = np.array(selected_trajs)[t_indeces]
    selected_trajs_dn = np.array(selected_trajs_dn)[t_indeces]
    selected_trajs_rn = np.array(selected_trajs_rn)[t_indeces]
    selected_trajs_a = np.array(selected_trajs_a)[t_indeces]
    selected_trajs_q_dn = np.array(selected_trajs_q_dn)[t_indeces]
    selected_trajs_q_rn = np.array(selected_trajs_q_rn)[t_indeces]

    fig2 = plt.figure(figsize=(7, 6.2), dpi=300)
    ax1 = fig2.add_subplot(5, 1, 1) 
    ax1.set_xlabel('Time in ICU', fontsize = 7)
    ax1.set_ylabel('Patient', fontsize = 7)
    ax1.set_title('D-Network Median', fontsize=7)
    
    ax2 = fig2.add_subplot(5, 1, 2) 
    ax2.set_xlabel('Time in ICU', fontsize = 7)
    ax2.set_ylabel('Patient', fontsize = 7)
    ax2.set_title('R-Network Median', fontsize=7)
    
    ax3 = fig2.add_subplot(5, 1, 3) 
    ax3.set_title('D-Network Q Selected', fontsize=7)
     
    ax4 = fig2.add_subplot(5, 1, 4) 
    ax4.set_title('R-Network Q Selected', fontsize=7)

    ax5 = fig2.add_subplot(5, 1, 5) 
    ax5.set_title('Selected Treatments', fontsize=7)

    v_dn = np.array([np.pad(k, (0, 20 - len(k)), 'constant') for k in selected_trajs_dn]).reshape(nb_traj, 20)
    v_rn = np.array([np.pad(k, (0, 20 - len(k)), 'constant') for k in selected_trajs_rn]).reshape(nb_traj, 20)
    aa = np.array([np.pad(k, (0, 20 - len(k)), 'constant') for k in selected_trajs_a]).reshape(nb_traj, 20).astype(np.float32)
    q_dn = np.array([np.pad(k, (0, 20 - len(k)), 'constant') for k in selected_trajs_q_dn]).reshape(nb_traj, 20)
    q_rn = np.array([np.pad(k, (0, 20 - len(k)), 'constant') for k in selected_trajs_q_rn]).reshape(nb_traj, 20)
    m = [np.zeros_like(a) for a in selected_trajs_dn]
    mask = np.array([np.pad(k, (0, 20 - len(k)), 'constant', constant_values=(1, 1)) for k in m]).reshape(nb_traj, 20)
    seaborn.heatmap(v_dn, vmin=-1, vmax=0, linewidth=1, annot=True, square=False, annot_kws={'fontsize': 6}, cbar=True, cmap='plasma', fmt=".2f", mask=mask, ax=ax1)
    seaborn.heatmap(v_rn, vmin=0, vmax=1, linewidth=1, annot=True, square=False, annot_kws={'fontsize': 6}, cbar=True, cmap='plasma', fmt=".2f", mask=mask, ax=ax2)
    seaborn.heatmap(q_dn, vmin=-1, vmax=0, linewidth=1, annot=True, square=False, annot_kws={'fontsize': 6}, cbar=True, cmap='plasma', fmt=".2f", mask=mask, ax=ax3)
    seaborn.heatmap(q_rn, vmin=0, vmax=1, linewidth=1, annot=True, square=False, annot_kws={'fontsize': 6}, cbar=True, cmap='plasma', fmt=".2f", mask=mask, ax=ax4)
    seaborn.heatmap(aa, vmin=-1, vmax=-1, linewidth=1, annot=True, square=False, annot_kws={'fontsize': 6}, cbar=True, cmap='binary', fmt=".0f", mask=mask, ax=ax5)
    ax1.set_ylim([0,5])
    ax1.set_yticklabels(selected_trajs, rotation='horizontal')
    ax2.set_ylim([0,5])
    ax2.set_yticklabels(selected_trajs, rotation='horizontal')
    ax3.set_ylim([0,5])
    ax3.set_yticklabels(selected_trajs, rotation='horizontal')
    ax4.set_ylim([0,5]) 
    ax4.set_yticklabels(selected_trajs, rotation='horizontal')
    ax5.set_ylim([0,5])
    ax5.set_yticklabels(selected_trajs, rotation='horizontal')
    fig2.tight_layout()
    plt.savefig(plot_file_name)
    plt.close("all")
    print("Done.")


selected_trajs_idx = yaml.safe_load(open("./plots/good_neg_trajs.yaml", "r"))
selected_trajs_idx = selected_trajs_idx["selected_trajs_idx"]
selected_trajs_idx = np.array_split(selected_trajs_idx, len(selected_trajs_idx) // 4 + 1)

for k in selected_trajs_idx:
    s = ""
    for i in k:
        s += str(i) + "_"
    s += ".pdf"
    s = os.path.join(p, s)
    worker(trajs=k, dn_th=th.dn_red, rn_th=th.rn_red, nb_traj=len(k), plot_file_name=s, verbose=True, random=False)
