import os
import pickle
import yaml
import itertools
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
    vitals = sorted(["o:mechvent", "o:GCS", "o:HR", "o:SysBP", "o:MeanBP", 
              "o:DiaBP", "o:RR", "o:Temp_C", "o:FiO2_1", "o:Potassium", "o:Sodium", "o:Chloride", "o:Glucose", 
              "o:Magnesium", "o:Calcium", "o:Hb", "o:WBC_count", "o:Platelets_count", "o:PTT", "o:PT", "o:Arterial_pH", 
              "o:paO2", "o:paCO2", "o:Arterial_BE", "o:HCO3", "o:Arterial_lactate", "o:SOFA", "o:SIRS", "o:Shock_Index", 
              "o:PaO2_FiO2", "o:cumulated_balance", "o:SpO2", "o:BUN", "o:Creatinine", "o:SGOT", "o:SGPT", "o:Total_bili", 
              "o:INR", "o:output_total", "o:output_4hourly"])
    clinical_measures = []
    clinical_measures_ttl = []
else:
    vitals = ["o:HR", "o:SysBP", "o:MeanBP", "o:DiaBP", "o:RR", "o:Temp_C", 
              "o:Arterial_lactate", "o:SpO2", "o:BUN", "o:INR"]
    clinical_measures = ["o:GCS", "o:SOFA", "o:SIRS"]
    clinical_measures_ttl = ["GCS", "SOFA", "SIRS"]
vitals = sorted([item for item in vitals if item not in clinical_measures])
our_measures = ["v_dn", "v_rn", "q_dn_a", "q_rn_a"]
our_measures_ttl = [r"$V_{D}$", r"$V_{R}$", r"$Q_{D}$", r"$Q_{R}$"]

if COMPLETE:
    f_name = r"./plots/pre_post_flag_vitals_full.pdf"
    figsize = (6, 7)
else:
    f_name = r"./plots/pre_post_flag_vitals.pdf"
    figsize = (5.5, 3)
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

fig, axs = plt.subplots(2, 1, figsize=(1.05, 2.05), dpi=300, sharex=True)
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


## DURATION ANALYSIS

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
            ha='center', va='bottom', fontsize=5.5, rotation=90)


# testing dead-ends and if enter, stays in dead-end, and getting hists
def hitflag(trajs, data, x_hours, th_dn_red, th_dn_yel, th_rn_red, th_rn_yel):
    x = np.array([len(data[data.traj == k]["a"]) for k in trajs])
    x = len(x[x >= (x_hours / 4)])  # 12H == at lease 4 steps
    just_flag_red = []
    remain_on_flag_red = []
    just_flag_yel = []
    remain_on_flag_yel = []
    no_flag_at_start = []
    for traj in trajs:
        q_dn_traj = data[data.traj == traj]['q_dn'].tolist()
        v_dn_traj = np.array([func(q) for q in q_dn_traj], dtype=np.float32)
        q_rn_traj = data[data.traj == traj]['q_rn'].tolist()
        v_rn_traj = np.array([func(q) for q in q_rn_traj], dtype=np.float32)
        if len(v_dn_traj) <= x_hours / 4:
            continue
        cond_red = np.logical_and(v_dn_traj < th_dn_red, v_rn_traj < th_rn_red)
        cond_yel = np.logical_or(np.logical_and.reduce((v_dn_traj <= th_dn_yel, v_dn_traj >= th_dn_red, v_rn_traj <= th_rn_yel)),
                        np.logical_and.reduce((v_rn_traj <= th_rn_yel, v_rn_traj >= th_rn_red, v_dn_traj <= th_dn_yel)))
        cond_both = np.logical_and(v_dn_traj < th_dn_yel, v_rn_traj < th_rn_yel)
        for flag in ["red", "yel"]:
            cond = cond_red if flag == "red" else cond_yel
            just_flag = just_flag_red if flag == "red" else just_flag_yel
            remain_on_flag = remain_on_flag_red if flag == "red" else remain_on_flag_yel
            # analysis of `last` x_hours:
            if True in cond:
                # first_occur = np.where(cond == True)[0][0]
                t = -(x_hours//4 + 1)
                if all(cond[t:]):  # hit and stay on
                    remain_on_flag.append(traj)
                elif any(cond[t:]):  # only hit (but flase in the above condition)
                    just_flag.append(traj)
            # analysis of `first` x_hours
        if not any(cond_both[:(x_hours//4 + 1)]):  # no flag at the beginning
            no_flag_at_start.append(traj)

    for flag in ["red", "yel"]:
        just_flag = just_flag_red if flag == "red" else just_flag_yel
        remain_on_flag = remain_on_flag_red if flag == "red" else remain_on_flag_yel
        print("{0:.1f}% test cases ({1} out of {2}) END with {3} flag of at least {4}H".format(100*len(remain_on_flag)/x, len(remain_on_flag), x, flag, x_hours))
    print("{0:.1f}% test cases ({1} out of {2}) START with NO flag of at least {3}H".format(100*len(no_flag_at_start)/x, len(no_flag_at_start), x, x_hours))
    
    return x, remain_on_flag_red, remain_on_flag_yel, no_flag_at_start


def mk_hist(trajs, data, th_dn_red, th_dn_yel, th_rn_red, th_rn_yel):
    a = list(data.groupby("traj").a)
    max_len = max([len(k[1]) for k in a])
    red_hist = np.zeros(max_len, dtype=np.int64)
    yellow_hist = np.zeros(max_len, dtype=np.int64)
    trajs_len = np.zeros(max_len + 1, dtype=np.int64)
    for traj in trajs:
        trajs_len[data[data.traj==traj].shape[0]] += 1  # first element corresponds to len of zero
        q_dn_traj = data[data.traj == traj]['q_dn'].tolist()
        v_dn_traj = np.array([func(q) for q in q_dn_traj], dtype=np.float32)
        q_rn_traj = data[data.traj == traj]['q_rn'].tolist()
        v_rn_traj = np.array([func(q) for q in q_rn_traj], dtype=np.float32)
        cond_red = np.logical_and(v_dn_traj < th_dn_red, v_rn_traj < th_rn_red)
        cond_yel = np.logical_or(np.logical_and.reduce((v_dn_traj <= th_dn_yel, v_dn_traj >= th_dn_red, v_rn_traj <= th_rn_yel)),
                        np.logical_and.reduce((v_rn_traj <= th_rn_yel, v_rn_traj >= th_rn_red, v_dn_traj <= th_dn_yel)))
        for flag in ["red", "yel"]:
            cond = cond_red if flag == "red" else cond_yel
            h = red_hist if flag == "red" else yellow_hist
            for k, it in itertools.groupby(cond):
                if k == True:
                    h[len(list(it)) - 1] += 1  # list(it) == sequence of k (k == True); 1st element corresponds to len of 1 (not 0)
    return red_hist, yellow_hist, trajs_len


pos_trajs = data[data.category == 1].traj.unique()
neg_trajs = data[data.category == -1].traj.unique()

if ANALISYS:
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize =(4.5, 1.75), dpi=300)
    w = 0.3
    d = {}
    for k, trajs in zip(["Nonsurvivors", "Survivors"], [neg_trajs, pos_trajs]):
        d[k] = {"duration": [], "ln": [], "remain_red": [], "remain_yel": [], "noflagstart": []}
        print()
        print("*"*30)
        print(k)
        for h in [4, 8, 12, 24, 48]:
            ln, remain_red, remain_yel, noflagstart = hitflag(trajs, data, h, th.dn_red, th.dn_yel, th.rn_red, th.rn_yel)
            print("="*30)
            d[k]["remain_red"].append(len(remain_red))
            d[k]["remain_yel"].append(len(remain_yel))
            d[k]["noflagstart"].append(len(noflagstart))
            d[k]["ln"].append(ln)
            d[k]["duration"].append(h)
    
    t = np.arange(len(d["Survivors"]["duration"]))
    yticks = [[0, 10, 20], [0, 10, 20], [0, 50, 100], [0, 50, 100]]
    for idx, k in enumerate(['remain_red', 'remain_yel', 'noflagstart']):
        ax = fig.add_subplot(1, 3, idx+1) 
        xneg = np.array(d["Nonsurvivors"][k]) * 100 / np.array(d["Nonsurvivors"]["ln"])
        xpos = np.array(d["Survivors"][k]) * 100 / np.array(d["Survivors"]["ln"])
        rects1 = ax.bar(t-w/2-0.05, xneg, w, label="Nonsurvivors patients", color="blue")
        rects2 = ax.bar(t+w/2+0.05, xpos, w, label="Survivors patients", color="green")
        xticklables = [str(j) + " H" for j in d["Survivors"]["duration"]]
        ax.set_xticks(t)
        ax.set_xticklabels(xticklables)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
        ax.set_ylim(yticks[idx][0], yticks[idx][-1])
        ax.set_yticks(yticks[idx])
        ax.set_yticklabels([str(j) + r"%" for j in yticks[idx]])
        seaborn.despine()
        autolabel(ax, rects1, d["Nonsurvivors"][k], d["Nonsurvivors"]["ln"])
        autolabel(ax, rects2, d["Survivors"][k], d["Survivors"]["ln"])
        ax.tick_params(axis='both', which='major', labelsize=7, length=7, width=0.8, direction='out')
        fig.tight_layout()
    fig.savefig(r"plots\duration_start_end.pdf", dpi=300)


if DURATION_HIST:
    red_hist_neg, yellow_hist_neg, trajs_len_neg = mk_hist(neg_trajs, data, th.dn_red, th.dn_yel, th.rn_red, th.rn_yel)
    red_hist_pos, yellow_hist_pos, trajs_len_pos = mk_hist(pos_trajs, data, th.dn_red, th.dn_yel, th.rn_red, th.rn_yel)

    trajs_len_neg = np.cumsum(trajs_len_neg)  # element i == num patients with at most i steps (rationally x[0] == 0)
    trajs_len_neg = trajs_len_neg[-1] - trajs_len_neg  # last elmt was total num; here: i elmt == num with at least i steps
    trajs_len_pos = np.cumsum(trajs_len_pos)
    trajs_len_pos = trajs_len_pos[-1] - trajs_len_pos 
    trajs_len_neg = trajs_len_neg[:10]  # for len>36H we consider 9 steps or more
    trajs_len_pos = trajs_len_pos[:10]

    red_hist_neg = np.append(red_hist_neg[:9], red_hist_neg[9:].sum())
    yellow_hist_neg = np.append(yellow_hist_neg[:9], yellow_hist_neg[9:].sum())
    red_hist_pos = np.append(red_hist_pos[:9], red_hist_pos[9:].sum())
    yellow_hist_pos = np.append(yellow_hist_pos[:9], yellow_hist_pos[9:].sum())

    xneg = red_hist_neg * 100 / trajs_len_neg
    xneg_cum = np.cumsum(xneg)
    xpos = red_hist_pos * 100 / trajs_len_pos
    xpos_cum = np.cumsum(xpos)

    fig, ax = plt.subplots(1, 1, figsize=(3, 1.75), dpi=300)
    t = np.arange(len(red_hist_neg))
    w = 0.3
    rects1 = ax.bar(t-w/2-0.05, xneg, w, label="Nonsurvivors (" + str(len(neg_trajs)) + " patients total)", color="blue")
    rects2 = ax.bar(t+w/2+0.05, xpos, w, label="Survivors (" + str(len(pos_trajs)) + " patients total)", color="green")
    xticklables = [str(4*(k+1))+" H" for k in t]
    xticklables[-1] = ">36 H"
    ax.set_xticks(t)
    ax.set_xticklabels(xticklables)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_yticklabels(['0%', '5%', '10%', '15%', '20%'])
    # ax.set_xlabel("Flag Duration", fontsize=7)
    # ax.set_ylabel(r"% Patients", fontsize=7)
    seaborn.despine()
    leg = ax.legend(loc='upper right', prop={'size': 7})
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')  # rm bg of legend

    autolabel(ax, rects1, red_hist_neg, trajs_len_neg)
    autolabel(ax, rects2, red_hist_pos, trajs_len_pos)
    ax.tick_params(axis='both', which='major', labelsize=7, length=7, width=0.8, direction='out')
    fig.tight_layout()
    fig.savefig(r"plots\duration_each.pdf", dpi=300)
