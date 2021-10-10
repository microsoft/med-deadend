import os
import yaml
import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn
import matplotlib.pyplot as plt
import pyprind

from ai import AI
from cortex import Cortex

np.random.seed(586)
torch.manual_seed(586)
rng = np.random.RandomState(586)

from IPython.core import debugger
debug = debugger.Pdb().set_trace

def load_best_ai(root_dir_run, params, sided_Q):
    checkpoint_dir = os.path.join(root_dir_run, 'ai_' + sided_Q + '_checkpoints')
    f = os.listdir(checkpoint_dir)
    f = [k for k in f if k[-3:] == ".pt"]
    last_checkpoint_idx = max([int(k[10:][:-3]) for k in f])
    last_ai_checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint' + str(last_checkpoint_idx) + '.pt'))
    best_ai = np.argmin(last_ai_checkpoint['validation_loss'])
    print("Best AI: ", best_ai, ' :: ', root_dir_run)
    best_ai_check_point = torch.load(os.path.join(checkpoint_dir, 'checkpoint' + str(best_ai) + '.pt'))
    ai = AI(state_dim=params['ais_size'], nb_actions=25, gamma=1.0, learning_rate=0, update_freq=0, 
            use_ddqn=False, rng=rng, device='cpu', sided_Q=sided_Q, network_size=params['ai_network_size'])
    ai.network.load_state_dict(best_ai_check_point['ai_network_state_dict'])
    print("AI loaded")
    return ai


def get_dn_rn_info(ai_dn, ai_rn, encoded_data, sepsis_data):
    traj_indeces = encoded_data['traj'].unique()
    data = {'traj': [], 'step': [], 's': [], 'a': [], 'q_dn': [], 'q_rn': [], 'category': [], 
            'grad_q_dn': [], 'grad_q_rn': []}
    state_cols = [k for k in encoded_data.columns if k[:2] == 's:']
    reward_col = [k for k in encoded_data.columns if k[:2] == 'r:'][0]  # only one `r` col
    action_col = [k for k in encoded_data.columns if k[:2] == 'a:'][0]  # only one `a` col
    bar = pyprind.ProgBar(len(traj_indeces))
    print("Making Q-values...")
    for traj in traj_indeces:
        bar.update()
        traj_states = encoded_data[encoded_data['traj'] == traj][state_cols].to_numpy().astype(np.float32)
        traj_q_dn = ai_dn.get_q(traj_states)
        traj_q_rn = ai_rn.get_q(traj_states)
        traj_q_dn = np.clip(traj_q_dn, -1, 0)
        traj_q_rn = np.clip(traj_q_rn, 0, 1)
        traj_r = sepsis_data[sepsis_data['traj'] == traj][reward_col].to_numpy().astype(np.float32)
        traj_a = sepsis_data[sepsis_data['traj'] == traj][action_col].to_numpy().astype(np.int32)
        steps = sepsis_data[sepsis_data['traj'] == traj]["step"].to_numpy().astype(np.int32)
        
        for i, action in enumerate(traj_a):
            data['traj'].append(traj)
            data['step'].append(steps[i])
            data['s'].append(traj_states[i, :])
            data['a'].append(action)
            data['q_dn'].append(traj_q_dn[i, :])
            data['q_rn'].append(traj_q_rn[i, :])

            s = traj_states[i, :]
            grad_q_dn = ai_dn.get_grad(s)
            grad_q_rn = ai_rn.get_grad(s)
            grad_q_dn[abs(grad_q_dn) < 0.01] = 0  # noise removal
            grad_q_rn[abs(grad_q_rn) < 0.01] = 0  # noise removal
            data['grad_q_dn'].append(grad_q_dn)
            data['grad_q_rn'].append(grad_q_rn)

            if traj_r[-1] == -1.0:
                data['category'].append(-1)
            elif traj_r[-1] == 1.0:
                data['category'].append(1)
            else:
                raise ValueError('last reward of a trajectory is neither of -+1.')
    data = pd.DataFrame(data)
    print("Q values made.")
    return data


def get_state_value_info(testing_idx, data): 
    # data should be made using `get_dn_rn_info`
    # UNCOMMENT the items you want to use. 
    death_trajectories = sorted(data[data.category == -1].traj.unique().tolist())
    recovery_trajectories = sorted(data[data.category == +1].traj.unique().tolist())
    info = {'death': {'traj': [], 'dn_min_q': [], 'dn_max_q': [], 'dn_median_q': [], 'dn_mean_q': [], 'dn_selected_q': [], 'dn_rank': [],
                    'rn_min_q': [], 'rn_max_q': [], 'rn_median_q': [], 'rn_mean_q': [], 'rn_selected_q': [], 'rn_rank': []}, 
    'recovery': {'traj': [], 'dn_min_q': [], 'dn_max_q': [], 'dn_median_q': [], 'dn_mean_q': [], 'dn_selected_q': [], 'dn_rank': [],
                    'rn_min_q': [], 'rn_max_q': [], 'rn_median_q': [], 'rn_mean_q': [], 'rn_selected_q': [], 'rn_rank': []}}
    bar = pyprind.ProgBar(len(death_trajectories))
    for t in death_trajectories:
        bar.update()
        if len(data[data.traj == t].q_dn) < -testing_idx:
                continue
        q_dn = data[data.traj == t].q_dn.iloc[testing_idx]  # e.g., q of last state of each traj (25 elements)
        q_rn = data[data.traj == t].q_rn.iloc[testing_idx]
        q_dn = np.clip(q_dn, -1, 0)
        q_rn = np.clip(q_rn, 0, 1)
        selected_a = data[data.traj == t].a.iloc[testing_idx]
        info['death']['traj'].append(t)
        # info['death']['dn_min_q'].append(np.min(q_dn))
        # info['death']['dn_max_q'].append(np.max(q_dn))
        info['death']['dn_median_q'].append(np.median(q_dn))
        # info['death']['dn_mean_q'].append(np.mean(q_dn))
        info['death']['dn_selected_q'].append(q_dn[selected_a])
        # info['death']['dn_rank'].append(min(len(q_dn) - np.where(np.sort(q_dn) == q_dn[selected_a])[0]))  # 1 is max value ... 25 is min value
        # info['death']['rn_min_q'].append(np.min(q_rn))
        # info['death']['rn_max_q'].append(np.max(q_rn))
        info['death']['rn_median_q'].append(np.median(q_rn))
        # info['death']['rn_mean_q'].append(np.mean(q_rn))
        info['death']['rn_selected_q'].append(q_rn[selected_a])
        # info['death']['rn_rank'].append(min(len(q_rn) - np.where(np.sort(q_rn) == q_rn[selected_a])[0]))
    bar = pyprind.ProgBar(len(recovery_trajectories))
    for t in recovery_trajectories:
        bar.update()
        if len(data[data.traj == t].q_dn) < -testing_idx:
                continue
        q_dn = data[data.traj == t].q_dn.iloc[testing_idx]  # e.g., q of last state of each traj (25 elements)
        q_rn = data[data.traj == t].q_rn.iloc[testing_idx]
        q_dn = np.clip(q_dn, -1, 0)
        q_rn = np.clip(q_rn, 0, 1)
        selected_a = data[data.traj == t].a.iloc[testing_idx]
        info['recovery']['traj'].append(t)
        # info['recovery']['dn_min_q'].append(np.min(q_dn))
        # info['recovery']['dn_max_q'].append(np.max(q_dn))
        info['recovery']['dn_median_q'].append(np.median(q_dn))
        # info['recovery']['dn_mean_q'].append(np.mean(q_dn))
        info['recovery']['dn_selected_q'].append(q_dn[selected_a])
        # info['recovery']['dn_rank'].append(min(len(q_dn) - np.where(np.sort(q_dn) == q_dn[selected_a])[0]))
        # info['recovery']['rn_min_q'].append(np.min(q_rn))
        # info['recovery']['rn_max_q'].append(np.max(q_rn))
        info['recovery']['rn_median_q'].append(np.median(q_rn))
        # info['recovery']['rn_mean_q'].append(np.mean(q_rn))
        info['recovery']['rn_selected_q'].append(q_rn[selected_a])
        # info['recovery']['rn_rank'].append(min(len(q_rn) - np.where(np.sort(q_rn) == q_rn[selected_a])[0]))

    print(min(info['death']['dn_selected_q']), max(info['death']['dn_selected_q']))  # --> max should be -1 (if testing=-1) 
    return info


def autolabel(ax, rects, hist, total):
    # Attach a text label above each bar in *rects*, displaying its height/value.
    for k, rect in enumerate(rects):
        height = rect.get_height()
        # h = str(round(height, 1))
        h = str(hist[k]) + " (" + str(total[k]) + ")"
        ax.annotate(h,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 2),  # points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', fontsize=5, rotation=90)


def plt_values(info, type_q, axs, nb_bar, y_string, x_verbose):  # type_q = 'selected', 'median', 'min', 'max', 'mean'
    plt.rcParams.update({'font.size': 7})
    hr_death, bins_r_death = np.histogram(info['death']['rn_'+ type_q + '_q'], np.linspace(0, 1, num=nb_bar+1, endpoint=True), density=False)
    hd_death, bins_d_death = np.histogram(info['death']['dn_' + type_q + '_q'], np.linspace(-1, 0, num=nb_bar+1, endpoint=True), density=False)
    hr_recovery, bins_r_recovery = np.histogram(info['recovery']['rn_' + type_q + '_q'], np.linspace(0, 1, num=nb_bar+1, endpoint=True), density=False)
    hd_recovery, bins_d_recovery = np.histogram(info['recovery']['dn_' + type_q + '_q'], np.linspace(-1, 0, num=nb_bar+1, endpoint=True), density=False)
    hr_death_total = np.sum(hr_death)
    hd_death_total = np.sum(hd_death)
    hr_recovery_total = np.sum(hr_recovery)
    hd_recovery_total = np.sum(hd_recovery)
    # fig, axs = plt.subplots(1, 2, figsize=(6, 1.5), dpi=300)
    bar_weadth = 1. / (nb_bar)
    w = (bar_weadth - (bar_weadth / 3)) / 2  # so that two bars fits nicely
    x = np.linspace(0, 1, num=nb_bar+1, endpoint=True)[:-1] + bar_weadth/2
    y = np.linspace(-1, 0, num=nb_bar+1, endpoint=True)[:-1] + bar_weadth/2
    rects1 = axs[0].bar(x - w/2 - w/6, hr_death/hr_death_total, w, label='Non-survivors', color='navy', alpha=1)
    rects2 = axs[0].bar(x + w/2 + w/6, hr_recovery/hr_recovery_total, w, label='Survivors', color='green', alpha=1)
    rects3 = axs[1].bar(y - w/2 - w/6, hd_death/hd_death_total, w, label='Non-survivors', color='navy', alpha=1) 
    rects4 = axs[1].bar(y + w/2 + w/6, hd_recovery/hd_recovery_total, w, label='Survivors', color='green', alpha=1)

    ## NOTE: Uncomment to also label the bars (don't forget to increase fig height)
    # autolabel(axs[0], rects1, hr_death, [hr_death_total]*10)
    # autolabel(axs[0], rects2, hr_recovery, [hr_recovery_total]*10)
    # autolabel(axs[1], rects3, hd_death, [hd_death_total]*10)
    # autolabel(axs[1], rects4, hd_recovery, [hd_recovery_total]*10)

    xx = np.linspace(0, 1, num=nb_bar+1, endpoint=True)
    yy = np.linspace(-1, 0, num=nb_bar+1, endpoint=True)

    axs[0].set_ylim(0, 1)
    axs[0].set_xticks(list(x))
    if x_verbose:
        axs[0].set_xticklabels(np.arange(1, 11))
        # axs[0].set_xticklabels(["{0}:{1}".format(round(i,1), round(j,1)) for i, j in zip(xx[:-1], xx[1:])])
        # axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=60, ha="right")
    else:
        axs[0].set_xticklabels("" * len(xx - 1))
    axs[0].set_yticks([0, 0.5, 1])
    axs[0].set_yticklabels(['0%', '50%', '100%'])
    # axs[0].set_xlabel('Treatment Value', fontsize=7)
    # axs[0].set_ylabel(y_string, fontsize=7)
    axs[1].set_ylim(0, 1)
    axs[1].set_xticks(list(y))
    if x_verbose:
        axs[0].set_xticklabels(np.arange(1, 11))
        # axs[1].set_xticklabels(["{0}:{1}".format(round(i,1), round(j,1)) for i, j in zip(yy[:-1], yy[1:])])
        # axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=60, ha="right")
    else:
        axs[1].set_xticklabels("" * len(yy - 1))
    axs[1].set_yticks([0, 0.5, 1])
    axs[1].set_yticklabels(['0%', '50%', '100%'])
    # axs[1].set_xlabel('Treatment Value', fontsize=7)
    # axs[1].set_ylabel(y_string, fontsize=7)
    axs[0].set_title('R-Network', fontsize=7)
    axs[1].set_title('D-Network', fontsize=7)
    # leg = axs[1].legend(loc='upper left', prop={'size': 7})
    # leg.get_frame().set_linewidth(0.0)
    # leg.get_frame().set_facecolor('none')  # rm bg of legend

