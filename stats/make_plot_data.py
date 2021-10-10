import os
import pickle
import yaml
import numpy as np
import pandas as pd
import pyprind
from stats.plot_utils import get_dn_rn_info, load_best_ai
from stats.thresholds import th

rng = np.random.RandomState(123)
ROOT_DIR = os.path.abspath(".")

RAW = False  # True == num patients ; False == percentage

# Make plots directory if it doesn't already exist
plot_path = os.path.join(ROOT_DIR, "plots")
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

root_dir_mimic = os.path.join(ROOT_DIR, 'data', 'sepsis_mimiciii')
# Currently hard-coded for the illustrative run1 defined in `config_sepsis.yaml`
root_dir_run = os.path.join(ROOT_DIR, 'results', 'run1')
params = yaml.safe_load(open(os.path.join(root_dir_run, 'config.yaml'), 'r'))  # the used params in the given run directory

sepsis_test_data = pd.read_csv(os.path.join(root_dir_mimic, 'sepsis_final_data_K1_test.csv'))
encoded_test_data = pd.read_csv(os.path.join(root_dir_run, 'encoded_test_data.csv'))

step_indices = [-19, -13, -7, -4, -3, -2]

print("Loading best AI's and making Q-values ...")
ai_dn = load_best_ai(root_dir_run, params, 'negative')
ai_rn = load_best_ai(root_dir_run, params, 'positive')  # same params as D-Network
data = get_dn_rn_info(ai_dn, ai_rn, encoded_test_data, sepsis_test_data)  # same AIS

with open("./plots/value_data.pkl", "wb") as f:
    pickle.dump(data, f)

results = {"survivors": {}, "nonsurvivors": {}}

nonsurvivor_trajectories = sorted(data[data.category == -1].traj.unique().tolist())
survivor_trajectories = sorted(data[data.category == +1].traj.unique().tolist())
for i, trajectories in enumerate([nonsurvivor_trajectories, survivor_trajectories]):
    if i == 0:
        traj_type = "nonsurvivors"
        print("----- Non-survivors")
    else:
        traj_type = "survivors"
        print("+++++ Survivors")
    dn_q_traj = []
    rn_q_traj = []
    dn_q_selected_action_traj = []
    rn_q_selected_action_traj = []
    bar = pyprind.ProgBar(len(trajectories))
    for traj in trajectories:
        bar.update()
        d = data[data.traj == traj]
        dn_q_traj.append(np.array(d.q_dn.to_numpy().tolist(), dtype=np.float32))
        rn_q_traj.append(np.array(d.q_rn.to_numpy().tolist(), dtype=np.float32))
        dn_q_selected_action = [d.q_dn.tolist()[t][d.a.tolist()[t]] for t in range(d.q_dn.shape[0])] 
        dn_q_selected_action_traj.append(dn_q_selected_action)
        rn_q_selected_action = [d.q_rn.tolist()[t][d.a.tolist()[t]] for t in range(d.q_rn.shape[0])] 
        rn_q_selected_action_traj.append(rn_q_selected_action)

    results[traj_type]["dn_q_selected_action_traj"] = dn_q_selected_action_traj
    results[traj_type]["rn_q_selected_action_traj"] = rn_q_selected_action_traj
    results[traj_type]["dn_v_median_traj"] = [np.median(q, axis=1) for q in dn_q_traj]
    results[traj_type]["rn_v_median_traj"] = [np.median(q, axis=1) for q in rn_q_traj]
    # results[traj_type]["dn_v_max_traj"] = [np.max(q, axis=1) for q in dn_q_traj]
    # results[traj_type]["dn_v_max5_traj"] = [np.sort(q, axis=1)[:, -5] for q in dn_q_traj]
    # results[traj_type]["rn_v_max_traj"] = [np.max(q, axis=1) for q in rn_q_traj]
    # results[traj_type]["rn_v_max5_traj"] = [np.sort(q, axis=1)[:, -5] for q in rn_q_traj]


bokeh = {"time": [], "survivors": {}, "nonsurvivors": {}}
for i in ["survivors", "nonsurvivors"]:
    bokeh[i] = {"V_D": {"red": [], "yellow": [], "noflag": []}, "Q_D": {"red": [], "yellow": [], "noflag": []}, 
                "V_R": {"red": [], "yellow": [], "noflag": []}, "Q_R": {"red": [], "yellow": [], "noflag": []},
                "V_FULL": {"red": [], "yellow": [], "noflag": []}, "Q_FULL": {"red": [], "yellow": [], "noflag": []}}

for i, time_index in enumerate(step_indices):
    print("Time: {0:3} H".format((time_index + 1) * 4))
    bokeh["time"].append(str((time_index + 1) * 4) + " Hours")
    for traj_type in ["survivors", "nonsurvivors"]:
        x = results[traj_type]
        v_dn = np.array([v[time_index] for v in x["dn_v_median_traj"] if len(v) > abs(time_index)])
        q_dn = np.array([q[time_index] for q in x["dn_q_selected_action_traj"] if len(q) > abs(time_index)])
        v_rn = np.array([v[time_index] for v in x["rn_v_median_traj"] if len(v) > abs(time_index)])
        q_rn = np.array([q[time_index] for q in x["rn_q_selected_action_traj"] if len(q) > abs(time_index)])

        assert(len(v_dn) == len(v_rn))
        assert(len(q_dn) == len(q_rn))
        assert(len(v_dn) == len(q_rn))
        if RAW:
            l = 1
        else:
            l = len(v_dn)

        bokeh[traj_type]["V_D"]["noflag"].append(sum(v_dn > th.dn_yel) / l)
        bokeh[traj_type]["V_R"]["noflag"].append(sum(v_rn > th.rn_yel) / l)
        bokeh[traj_type]["V_FULL"]["noflag"].append(sum(np.logical_or((v_dn > th.dn_yel), (v_rn > th.rn_yel))) / l)
        bokeh[traj_type]["Q_D"]["noflag"].append(sum(q_dn > th.dn_yel) / l)
        bokeh[traj_type]["Q_R"]["noflag"].append(sum(q_rn > th.rn_yel) / l)
        bokeh[traj_type]["Q_FULL"]["noflag"].append(sum(np.logical_or((q_dn > th.dn_yel), (q_rn > th.rn_yel))) / l)

        bokeh[traj_type]["V_D"]["red"].append(sum(v_dn < th.dn_red) / l)
        bokeh[traj_type]["V_R"]["red"].append(sum(v_rn < th.rn_red) / l)
        bokeh[traj_type]["V_FULL"]["red"].append(sum(np.logical_and((v_dn < th.dn_red), (v_rn < th.rn_red))) / l)
        bokeh[traj_type]["Q_D"]["red"].append(sum(q_dn < th.dn_red) / l)
        bokeh[traj_type]["Q_R"]["red"].append(sum(q_rn < th.rn_red) / l)
        bokeh[traj_type]["Q_FULL"]["red"].append(sum(np.logical_and((q_dn < th.dn_red), (q_rn < th.rn_red))) / l)

        bokeh[traj_type]["V_D"]["yellow"].append(sum(np.logical_and((v_dn <= th.dn_yel), (v_dn >= th.dn_red))) / l)
        bokeh[traj_type]["V_R"]["yellow"].append(sum(np.logical_and((v_rn <= th.rn_yel), (v_rn >= th.rn_red))) / l)
        bokeh[traj_type]["V_FULL"]["yellow"].append(sum(np.logical_or(
                        np.logical_and.reduce((v_dn <= th.dn_yel, v_dn >= th.dn_red, v_rn <= th.rn_yel)),
                        np.logical_and.reduce((v_rn <= th.rn_yel, v_rn >= th.rn_red, v_dn <= th.dn_yel))) ) / l)
        bokeh[traj_type]["Q_D"]["yellow"].append(sum(np.logical_and((q_dn <= th.dn_yel), (q_dn >= th.dn_red))) / l)
        bokeh[traj_type]["Q_R"]["yellow"].append(sum(np.logical_and((q_rn <= th.rn_yel), (q_rn >= th.rn_red))) / l)
        bokeh[traj_type]["Q_FULL"]["yellow"].append(sum(np.logical_or(
                        np.logical_and.reduce((q_dn <= th.dn_yel, q_dn >= th.dn_red, q_rn <= th.rn_yel)),
                        np.logical_and.reduce((q_rn <= th.rn_yel, q_rn >= th.rn_red, q_dn <= th.dn_yel))) ) / l)

if RAW:
    p = os.path.join(ROOT_DIR, "./plots/flag_data_raw.pkl")
else:
    p = os.path.join(ROOT_DIR, "./plots/flag_data.pkl")
with open(p, "wb") as f:
    pickle.dump(bokeh, f)
