import os
import yaml
import pickle
import numpy as np
import pandas as pd
import pyprind
import matplotlib.pyplot as plt 
import seaborn
from stats.thresholds import th 

rng = np.random.RandomState(123)
fontsize = 6.5

OBS_PLOT = True
WRITE_META = True
COMPLETE = True     # full list of vital vs. short list

if OBS_PLOT:
    if COMPLETE:
        folder_path = r"./plots/obs_complete"
        figsize = (5, 5.5)
    else:
        folder_path = r"./plots/obs"
        figsize = (6.2, 2.8)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

ROOT_DIR = r"."
root_dir_mimic = os.path.join(ROOT_DIR, 'data', 'sepsis_mimiciii')
# Currently hard-coded for the illustrative run1 defined in `config_sepsis.yaml`
root_dir_run = os.path.join(ROOT_DIR, 'results', 'run1')
params = yaml.safe_load(open(os.path.join(root_dir_run, 'config.yaml'), 'r'))  # the used params in the given run directory

print("Loading data ...")
sepsis_raw_data = pd.read_csv(os.path.join(root_dir_mimic, 'sepsis_final_data_K1_RAW.csv'))
with open(r"./plots/value_data.pkl", "rb") as f:
    data = pickle.load(f)
print("Done.")


data["v_dn"] = data.q_dn.apply(lambda x: np.median(x))
data["v_rn"] = data.q_rn.apply(lambda x: np.median(x))
data["q_dn_a"] = data.apply(lambda x: x.q_dn[x.a], axis=1)
data["q_rn_a"] = data.apply(lambda x: x.q_rn[x.a], axis=1)
data["IV"] = data.a.apply(lambda x: x // 5)
data["VP"] = data.a.apply(lambda x: x % 5)

for i in range(25):  # mk values of actions separately for plotting
    iv = i // 5
    vp = i % 5
    data["q_dn: iv=" + str(iv) + " vp=" + str(vp)] = data.q_dn.apply(lambda x: x[i])
    data["q_rn: iv=" + str(iv) + " vp=" + str(vp)] = data.q_rn.apply(lambda x: x[i])

red = np.logical_and(data.v_dn <= th.dn_red, data.v_rn <= th.rn_red)
yellow = np.logical_or(np.logical_and.reduce((data.v_dn <= th.dn_yel, data.v_dn >= th.dn_red, data.v_rn <= th.rn_yel)),
                       np.logical_and.reduce((data.v_rn <= th.rn_yel, data.v_rn >= th.rn_red, data.v_dn <= th.dn_yel)))
data["red"] = red
data["yellow"] = yellow

selected_trajs_idx = yaml.safe_load(open("./plots/good_neg_trajs.yaml", "r"))
selected_trajs_idx = selected_trajs_idx["selected_trajs_idx"]
print("selected trajs:")
print(selected_trajs_idx)

if COMPLETE:
    vitals = sorted(["o:mechvent", "o:GCS", "o:HR", "o:SysBP", "o:MeanBP", 
              "o:DiaBP", "o:RR", "o:Temp_C", "o:FiO2_1", "o:Potassium", "o:Sodium", "o:Chloride", "o:Glucose", 
              "o:Magnesium", "o:Calcium", "o:Hb", "o:WBC_count", "o:Platelets_count", "o:PTT", "o:PT", "o:Arterial_pH", 
              "o:paO2", "o:paCO2", "o:Arterial_BE", "o:HCO3", "o:Arterial_lactate", "o:SOFA", "o:SIRS", "o:Shock_Index", 
              "o:PaO2_FiO2", "o:cumulated_balance", "o:SpO2", "o:BUN", "o:Creatinine", "o:SGOT", "o:SGPT", "o:Total_bili", 
              "o:INR", "o:output_total", "o:output_4hourly"])
    clinical_measures = []
    clinical_measures_ttl = []
    num_cols = 5
else:
    vitals = ["o:HR", "o:SysBP", "o:MeanBP", "o:DiaBP", "o:RR", "o:Temp_C", 
              "o:Arterial_lactate", "o:SpO2", "o:BUN", "o:INR"]
    clinical_measures = ["o:GCS", "o:SOFA", "o:SIRS"]
    clinical_measures_ttl = ["GCS", "SOFA", "SIRS"]
    num_cols = 7
vitals = sorted([item for item in vitals if item not in clinical_measures])
our_measures = ["v_dn", "v_rn", "q_dn_a", "q_rn_a"]
our_measures_ttl = [r"$V_{D}$", r"$V_{R}$", r"$Q_{D}$", r"$Q_{R}$"]

if OBS_PLOT:
    plt.rcParams.update({'font.size': fontsize})
    with open(os.path.join(os.path.abspath(r"plots"), "charts_meta_data.txt"), "w") as f:
        bar = pyprind.ProgBar(len(selected_trajs_idx))
        for traj in selected_trajs_idx:
            bar.update()
            d = sepsis_raw_data[sepsis_raw_data.traj == traj]
            dq = data[data.traj == traj]
            rflags = dq.step[dq.red]
            yflags = dq.step[dq.yellow]
            icustayid = d["m:icustayid"].astype(np.int64).unique()
            if len(icustayid) == 1:
                icustayid = icustayid[0]
            else:
                raise ValueError("more than one icustay_id in traj", traj)
            try:
                onset_step = np.where(d["m:charttime"].values < d["m:presumed_onset"].values[0])[0][-1]
            except:
                onset_step = 0
            num_rows_vitals = len(vitals) // num_cols + int(len(vitals) % num_cols > 0)
            num_rows_clinical = len(clinical_measures) // num_cols + int(len(clinical_measures) % num_cols > 0)
            num_rows_ours = len(our_measures) // num_cols + int(len(our_measures) % num_cols > 0)
            fig, axs = plt.subplots(num_rows_vitals + num_rows_clinical + num_rows_ours, num_cols, figsize=figsize, dpi=300, sharex=True)
            for i, k in enumerate(vitals):
                ax = axs[i // num_cols, i % num_cols]
                seaborn.lineplot(x='step', y=k, data=d, ax=ax, lw=2)
                m = d[k].min()
                m2 = (d[k].max() - d[k].min()) / 10
                ax.plot(onset_step, m-3*m2, "*", mew=1, mec="black", mfc="black", ms=4, clip_on=False)
                ax.plot(rflags, [m-m2]*len(rflags), "o", mew=0, mec=None, mfc="red", ms=2)
                ax.plot(yflags, [m-m2]*len(yflags), "o", mew=0, mec=None, mfc="#FFCC00", ms=2)
                ax.set_ylabel("")
                ax.set_title(k[2:], {"fontsize": fontsize}, pad=3)
            for n in range(len(vitals), num_rows_vitals*num_cols):
                axs[n // num_cols, n % num_cols].set_visible(False)
            for i, (k, k_ttl) in enumerate(zip(clinical_measures, clinical_measures_ttl)):
                ax = axs[i // num_cols + num_rows_vitals, i % num_cols]
                seaborn.lineplot(x='step', y=k, data=d, ax=ax, lw=2)
                m = d[k].min()
                m2 = (d[k].max() - d[k].min()) / 10
                ax.plot(onset_step, m-3*m2, "*", mew=1, mec="black", mfc="black", ms=4, clip_on=False)
                ax.plot(rflags, [m-m2]*len(rflags), "o", mew=0, mec=None, mfc="red", ms=2)
                ax.plot(yflags, [m-m2]*len(yflags), "o", mew=0, mec=None, mfc="#FFCC00", ms=2)
                ax.set_ylabel("")
                ax.set_title(k_ttl, {"fontsize": fontsize}, pad=3)
            for n in range(len(clinical_measures), num_rows_clinical*num_cols):
                axs[n // num_cols + num_rows_vitals, n % num_cols].set_visible(False)
            for i, (k, k_ttl) in enumerate(zip(our_measures, our_measures_ttl)):
                ax = axs[i // num_cols + num_rows_vitals + num_rows_clinical, i % num_cols]
                seaborn.lineplot(x='step', y=k, data=dq, ax=ax, lw=2)
                m = dq[k].min()
                m2 = (dq[k].max() - dq[k].min()) / 10
                ax.plot(onset_step, m-3*m2, "*", mew=1, mec="black", mfc="black", ms=4, clip_on=False)
                ax.plot(rflags, [m-m2]*len(rflags), "o", mew=0, mec=None, mfc="red", ms=2)
                ax.plot(yflags, [m-m2]*len(yflags), "o", mew=0, mec=None, mfc="#FFCC00", ms=2)
                plt.xticks(ticks=[5, 10, 15, 20], labels=["5", "10", "15", "20"], fontsize=fontsize)
                ax.tick_params(axis="x", pad=1)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                ax.set_xlabel('Steps', fontsize=fontsize)
                ax.set_ylabel("")
                ax.set_title(k_ttl, {"fontsize": fontsize}, pad=0)
            for n in range(len(our_measures), num_rows_ours*num_cols):
                axs[n // num_cols + num_rows_vitals + num_rows_clinical, n % num_cols].set_visible(False)
            # fig.suptitle("Traj:" + str(traj) +" icustay_id:" + str(icustayid), fontsize=9)
            seaborn.despine(fig=fig)
            plt.tight_layout(pad=0.7)
            f_name = os.path.join(folder_path, "traj" + str(traj) +"_icustay" + str(icustayid) + ".pdf")
            fig.savefig(f_name)
            plt.close("all")

if WRITE_META:
    with open(os.path.join(os.path.abspath(r"plots"), "charts_meta_data.txt"), "w") as f:
        bar = pyprind.ProgBar(len(selected_trajs_idx))
        for traj in selected_trajs_idx:
            bar.update()
            d = sepsis_raw_data[sepsis_raw_data.traj == traj]
            charttime = d["m:charttime"].astype(np.int64)
            icustayid = d["m:icustayid"].astype(np.int64).unique()
            if len(icustayid) == 1:
                icustayid = icustayid[0]
            else:
                raise ValueError("more than one icustay_id in traj", traj)
            presumed_onset = d["m:presumed_onset"].astype(np.int64).unique()
            if len(presumed_onset) == 1:
                presumed_onset = presumed_onset[0]
            else:
                raise ValueError("more than one presumed onset in traj", traj)
            f.write("\n")
            f.write("=" * 30)
            f.write("\n")
            s = "Traj: " + str(traj) + "   ||   " + "icustay_id: " + str(icustayid) + " | " + "presumed_onset: " + str(presumed_onset)
            f.write(s)
            f.write("\n")
            f.write(">>>>> chart timestamps:\n")
            for k, c in enumerate(charttime):
                f.write(f"step {k:2}: {c}")
                f.write("\n")
