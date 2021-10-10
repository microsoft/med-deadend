import os
import pickle
import yaml
import numpy as np
import torch
import pandas as pd
import pyprind
import matplotlib.pyplot as plt
import seaborn
from stats.thresholds import th 

plt.close("all")

ANALISYS = True
DURATION_HIST = True
COMPLETE = False    # full list of vitals vs. short list

plt.rcParams.update({'font.size': 7})


np.random.seed(456)
torch.manual_seed(456)

func = np.median

ROOT_DIR = r"."
root_dir_mimic = os.path.join(ROOT_DIR, 'data', 'sepsis_mimiciii')
# Currently hard-coded for the illustrative run1 defined in `config_sepsis.yaml`
root_dir_run = os.path.join(ROOT_DIR, 'results', 'run1')
params = yaml.safe_load(open(os.path.join(root_dir_run, 'config.yaml'), 'r'))  # the used params in the given run directory

print("Loading data ...")
sepsis_raw_data = pd.read_csv(os.path.join(root_dir_mimic, 'sepsis_final_data_K1_RAW.csv'))
with open(r"./plots/value_data.pkl", "rb") as f:
    data = pickle.load(f)

## Treatment Analysis

def mk_d(category):  
    d = data[data.category == category].copy()
    d["v_dn"] = d.q_dn.apply(lambda x: np.median(x))
    d["v_rn"] = d.q_rn.apply(lambda x: np.median(x))
    d["v_max_dn"] = d.q_dn.apply(lambda x: np.max(x))
    d["v_max_rn"] = d.q_rn.apply(lambda x: np.max(x))
    d["v_5max_dn"] = d.q_dn.apply(lambda x: np.sort(x)[-5])
    d["v_5max_rn"] = d.q_rn.apply(lambda x: np.sort(x)[-5])
    d["most_secure_IV_dn"] = d.q_dn.apply(lambda x: np.mean(np.where(x == np.max(x))[0]) // 5)
    d["most_secure_VP_dn"] = d.q_dn.apply(lambda x: np.mean(np.where(x == np.max(x))[0]) % 5)
    d["most_secure_IV_rn"] = d.q_rn.apply(lambda x: np.mean(np.where(x == np.max(x))[0]) // 5)
    d["most_secure_VP_rn"] = d.q_rn.apply(lambda x: np.mean(np.where(x == np.max(x))[0]) % 5)
    d["q_dn_a"] = d.apply(lambda x: x.q_dn[x.a], axis=1)
    d["q_rn_a"] = d.apply(lambda x: x.q_rn[x.a], axis=1)
    d["IV"] = d.a.apply(lambda x: x // 5)
    d["VP"] = d.a.apply(lambda x: x % 5)
    d["t_rank_dn"] = d.apply(lambda x: np.where(np.sort(x.q_dn) == x.q_dn_a)[0][-1], axis=1)  # rank (towards best) [note: 24=best]
    d["t_rank_rn"] = d.apply(lambda x: np.where(np.sort(x.q_rn) == x.q_rn_a)[0][-1], axis=1) 
    for i in range(25):  # mk values of actions separately for plotting
        iv = i // 5
        vp = i % 5
        d["q_dn: iv=" + str(iv) + " vp=" + str(vp)] = d.q_dn.apply(lambda x: x[i])
        d["q_rn: iv=" + str(iv) + " vp=" + str(vp)] = d.q_rn.apply(lambda x: x[i])

    red = np.logical_and(d.v_dn <= th.dn_red, d.v_rn <= th.rn_red)
    yellow = np.logical_or(np.logical_and.reduce((d.v_dn <= th.dn_yel, d.v_dn >= th.dn_red, d.v_rn <= th.rn_yel)),
                        np.logical_and.reduce((d.v_rn <= th.rn_yel, d.v_rn >= th.rn_red, d.v_dn <= th.dn_yel)))
    d["red"] = red
    d["yellow"] = yellow

    red_actions = d.loc[red].a
    yellow_actions = d.loc[yellow].a
    no_actions = d.loc[np.logical_or(d.v_dn > th.dn_yel, d.v_rn > th.rn_yel)].a
    a_red_hist, _ = np.histogram(red_actions, bins=np.arange(26))
    a_yellow_hist, _ = np.histogram(yellow_actions, bins=np.arange(26))
    a_no_hist, _ = np.histogram(no_actions, bins=np.arange(26))
    return d, a_red_hist, a_yellow_hist, a_no_hist

# To see which treatment is used the most:
# asorted = np.argsort(a_red_hist)[::-1]
# asorted // 5  # IV
# asorted % 5   # VP

d_neg, a_red_hist_neg, a_yellow_hist_neg, a_no_hist_neg = mk_d(category=-1)
d_pos, a_red_hist_pos, a_yellow_hist_pos, a_no_hist_pos = mk_d(category=1)


## HEATMAP

for j in ["survivors", "nonsurvivors"]:
    if j == "survivors":
        a_red_hist, a_no_hist = a_red_hist_pos, a_no_hist_pos
    else:
        a_red_hist, a_no_hist = a_red_hist_neg, a_no_hist_neg
    plt.rcParams.update({'font.size': 6.5})
    fig, axs = plt.subplots(1, 2, figsize=(4.3, 2), dpi=300)
    for k, a in enumerate([a_red_hist, a_no_hist]):
        ax = axs[k]
        a = a * 100 / a.sum()
        ttl = "Inside Red-flag States" if k == 0 else "Outside Red-flag States"
        cb = True if k == 1 else False
        seaborn.heatmap(a.reshape(5, 5), vmin=0, vmax=20, linewidth=0, annot=True, square=True, annot_kws={'fontsize': 6.5}, 
                        cbar=True, cbar_kws={"ticks": [0, 5, 10, 15, 20], "drawedges": False, "fraction": 0.045}, cmap='plasma', 
                        fmt=".1f", ax=ax)
        ax.set_xlabel("VP")
        ax.set_ylabel("IV")
        ax.set_title(ttl)
    fig.axes[-1].set_yticklabels(["0%", "5%", "10%", "15%", "20%"])
    fig.axes[-2].set_yticklabels(["0%", "5%", "10%", "15%", "20%"])
    fig.tight_layout()
    fig.savefig(r"./plots/actions_heatmap_" + j + ".pdf")


#### PRE+POST-flag analysis:

window_pre = 6  # looking at "X steps" before the first flag (flag is window_pre + 1)
window_post = 4  # looking at "X steps" after the first flag (flag is at 0)
for category in [1, -1]:
    monitor = pd.DataFrame()
    d = d_pos if category == 1 else d_neg
    bar = pyprind.ProgBar(len(d.traj.unique()))
    for traj in d.traj.unique():
        bar.update()
        dt = d[d.traj == traj]
        raw = sepsis_raw_data[sepsis_raw_data.traj == traj]
        cond = np.logical_and(dt["v_dn"] < th.dn_yel, dt["v_rn"] < th.rn_yel)
        if any(cond):
            first_flag = np.where(cond == True)[0][0]
            if first_flag < window_pre:  # at least 6 steps (24 H) before the flag
                continue
            if len(cond) - first_flag <= window_post:  # flag happens too late
                continue
            raw1 = raw[raw.step.isin(np.arange(first_flag - window_pre, first_flag + window_post + 1))].drop(columns=["a:action"])  # the flag point is included
            dt1 = dt[dt.step.isin(np.arange(first_flag - window_pre, first_flag + window_post + 1))].drop(columns=["traj", "step", "s", "q_dn", "q_rn"])
            temp_df = raw1.reset_index().join(dt1.reset_index(), rsuffix="_dup")
            temp_df["step"] = np.arange(-window_pre, len(raw1.step)-window_pre)  # -6, -5, ..., 0, 1, 2, ...  --> 0 is the first flag point
            monitor = monitor.append(temp_df)
    if category == 1:
        monitor_pos = monitor
    else:
        monitor_neg = monitor
    

## VITALS:

if COMPLETE:
    vitals = sorted(["o:mechvent", "o:max_dose_vaso", "o:GCS", "o:HR", "o:SysBP", "o:MeanBP", 
              "o:DiaBP", "o:RR", "o:Temp_C", "o:FiO2_1", "o:Potassium", "o:Sodium", "o:Chloride", "o:Glucose", 
              "o:Magnesium", "o:Calcium", "o:Hb", "o:WBC_count", "o:Platelets_count", "o:PTT", "o:PT", "o:Arterial_pH", 
              "o:paO2", "o:paCO2", "o:Arterial_BE", "o:HCO3", "o:Arterial_lactate", "o:SOFA", "o:SIRS", "o:Shock_Index", 
              "o:PaO2_FiO2", "o:cumulated_balance", "o:SpO2", "o:BUN", "o:Creatinine", "o:SGOT", "o:SGPT", "o:Total_bili", 
              "o:INR", "o:input_total", "o:input_4hourly", "o:output_total", "o:output_4hourly"])
    clinical_measures = []
    clinical_measures_ttl = []
else:
    vitals = ["o:HR", "o:SysBP", "o:MeanBP", 
              "o:DiaBP", "o:RR", "o:Temp_C", "o:Arterial_lactate", 
              "o:SpO2", "o:BUN", "o:INR"]
    clinical_measures = ["o:GCS", "o:SOFA", "o:SIRS", "o:max_dose_vaso", "o:input_4hourly"]
    clinical_measures_ttl = ["GCS", "SOFA", "SIRS", "max dose VP", "Input 4-hourly"]
vitals = sorted([item for item in vitals if item not in clinical_measures])
our_measures = ["v_dn", "v_rn", "q_dn_a", "q_rn_a"]
our_measures_ttl = [r"$V_{D}$", r"$V_{R}$", r"$Q_{D}$", r"$Q_{R}$"]

if COMPLETE:
    f_name = r"./plots/pre_post_flag_vitals_full.pdf"
    figsize = (6, 7)
else:
    f_name = r"./plots/pre_post_flag_vitals.pdf"
    figsize = (5, 3)
plt.rcParams.update({'font.size': 7})
num_cols = 5
num_rows_vitals = len(vitals) // num_cols + int(len(vitals) % num_cols > 0)
num_rows_clinical = len(clinical_measures) // num_cols + int(len(clinical_measures) % num_cols > 0)
num_rows_ours = len(our_measures) // num_cols + int(len(our_measures) % num_cols > 0)
fig, axs = plt.subplots(num_rows_vitals + num_rows_clinical + num_rows_ours, num_cols, figsize=figsize, dpi=300, sharex=True)
for i, item in enumerate(vitals):
    ax = axs[i // num_cols, i % num_cols]
    seaborn.lineplot(x="step", y=item, data=monitor_pos, legend=False, color="green", ci="sd", linewidth=1, ax=ax)
    seaborn.lineplot(x="step", y=item, data=monitor_neg, legend=False, color="blue", ci="sd", linewidth=1, ax=ax)
    # ax.lines[1].set_linestyle(':')
    tt = item[2:] if item[:2] == "o:" else item
    ax.set_title(tt, fontsize=7, pad=3)
    ax.set_xticks(np.arange(-window_pre, window_post+1))
    plt.setp(ax.get_xticklabels(), rotation=42, ha="right", rotation_mode="anchor")
    ax.set_ylabel("")
for n in range(len(vitals), num_rows_vitals*num_cols):
    axs[n // num_cols, n % num_cols].set_visible(False)

for i, (item, item_ttl) in enumerate(zip(clinical_measures, clinical_measures_ttl)):
    ax = axs[i // num_cols + num_rows_vitals, i % num_cols]
    seaborn.lineplot(x="step", y=item, data=monitor_pos, legend=False, color="green", ci="sd", linewidth=1, ax=ax)
    seaborn.lineplot(x="step", y=item, data=monitor_neg, legend=False, color="blue", ci="sd", linewidth=1, ax=ax)
    # ax.lines[1].set_linestyle(':')
    ax.set_title(item_ttl, fontsize=7, pad=3)
    ax.set_xticks(np.arange(-window_pre, window_post+1))
    plt.setp(ax.get_xticklabels(), rotation=42, ha="right", rotation_mode="anchor")
    ax.set_ylabel("")
for n in range(len(clinical_measures), num_rows_clinical*num_cols):
    axs[n // num_cols + num_rows_vitals, n % num_cols].set_visible(False)

for i, (item, item_ttl) in enumerate(zip(our_measures, our_measures_ttl)):
    ax = axs[i // num_cols + num_rows_vitals + num_rows_clinical, i % num_cols]
    seaborn.lineplot(x="step", y=item, data=monitor_pos, legend=False, color="green", ci="sd", linewidth=1, ax=ax)
    seaborn.lineplot(x="step", y=item, data=monitor_neg, legend=False, color="blue", ci="sd", linewidth=1, ax=ax)
    # ax.lines[1].set_linestyle(':')
    ax.set_title(item_ttl, fontsize=7, pad=0)
    ax.set_xticks(np.arange(-window_pre, window_post+1))
    plt.setp(ax.get_xticklabels(), rotation=42, ha="right", rotation_mode="anchor")
    ax.set_xlabel('Steps', fontsize=7)
    ax.set_ylabel("")
for n in range(len(our_measures), num_rows_ours*num_cols):
    axs[n // num_cols + num_rows_vitals + num_rows_clinical, n % num_cols].set_visible(False)

seaborn.despine()
fig.tight_layout()
fig.savefig(f_name)


## SELECTED TREATMENT VALUES

f_name = r"./plots/pre_post_flag_values.pdf"
plt.rcParams.update({'font.size': 7})
fig, axs = plt.subplots(2, 2, figsize=(2.2, 2), dpi=300, sharex=True)
for i, item in enumerate(zip(["v_max_dn", "v_max_rn"], ["v_5max_dn", "v_5max_rn"], ["q_dn_a", "q_rn_a"])):
    ttl = "D-Network" if i == 0 else "R-Network"
    # ylim = (-0.61, 0) if i == 0 else (0.48, 1)
    ylim = (-0.41, 0) if i == 0 else (0.58, 1)
    ax1 = axs[i, 0]
    ax2 = axs[i, 1]
    seaborn.lineplot(x="step", y=item[2], data=monitor_neg, legend=False, color="blue", ci="sd", linewidth=1, ax=ax1)
    seaborn.lineplot(x="step", y=item[0], data=monitor_neg, legend=False, color="orange", ci="sd", linewidth=1, ax=ax1)
    seaborn.lineplot(x="step", y=item[1], data=monitor_neg, legend=False, color="black", ci="sd", linewidth=1, ax=ax1)
    seaborn.lineplot(x="step", y=item[2], data=monitor_pos, legend=False, color="green", ci="sd", linewidth=1, ax=ax2)
    seaborn.lineplot(x="step", y=item[0], data=monitor_pos, legend=False, color="orange", ci="sd", linewidth=1, ax=ax2)
    seaborn.lineplot(x="step", y=item[1], data=monitor_pos, legend=False, color="black", ci="sd", linewidth=1, ax=ax2)
    for ax in [ax1, ax2]:
        ax.set_title(ttl, fontsize=7, pad=3)
        ax.set_xticks(np.arange(-window_pre, window_post+1))
        plt.setp(ax.get_xticklabels(), rotation=42, ha="right", rotation_mode="anchor")
        ax.set_ylabel("Value")
        ax.set_ylim(ylim)

seaborn.despine()
fig.tight_layout()
fig.savefig(f_name)

## TREATMETNS

fig, axs = plt.subplots(2, 1, figsize=(1.1, 2.7), dpi=300, sharex=True)
for i, item in enumerate(["IV", "VP"]):
    ax = axs[i]
    seaborn.lineplot(x="step", y=item, data=monitor_pos, legend=False, color="green", ci="sd", linewidth=1, ax=ax)
    seaborn.lineplot(x="step", y=item, data=monitor_neg, legend=False, color="blue", ci="sd", linewidth=1, ax=ax)
    # ax.lines[1].set_linestyle(':')
    ax.set_title(item, fontsize=7)
    ax.set_xticks(np.arange(-window_pre, window_post+1))
    plt.setp(ax.get_xticklabels(), rotation=42, ha="right", rotation_mode="anchor")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel("Amount")
seaborn.despine()
fig.tight_layout()
fig.savefig(r"./plots/pre_post_flag_treatments.pdf")

