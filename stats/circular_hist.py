import pickle 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle


with open(r"./plots/flag_data.pkl", "rb") as f:
    data = pickle.load(f)

terminal_color = {"nonsurvivor": "navy", "survivor": "green"}
flag_color = {"noflag": "#66CCCC", "red": "#FF9999", "yellow": "#FFCC00"}
fontsize = 7
psize = 147
inner_radius = 45
outer_radius = 120
w = outer_radius - inner_radius

minr = 0
maxr = 100
a = (outer_radius - inner_radius) / (maxr - minr)
b = -a * minr + inner_radius

def rmap(mic):
    return a * mic + b

def deg(rad):
    return (180.0 * rad) / np.pi

num_big = 6 + 1
big_angle = 2.0 * np.pi / num_big
small_angle = big_angle / (len(data["time"]) * 2 + 2)

fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
ax.set_xlim(-psize, psize)
ax.set_ylim(-psize, psize)
ax.set_aspect("equal")
# ax.axis('off')
plt.setp(ax.spines.values(), linewidth=0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor("#f5f5eeff")


def wedge_hist(end_big_angle, mapped_surv_values, mapped_nonsurv_values, color_big, ax):
    ax.add_patch(Wedge((0, 0), outer_radius, deg(end_big_angle-big_angle), deg(end_big_angle), width=w, color=color_big, lw=0))

    # radial axes
    ax.add_patch(Wedge((0, 0), outer_radius+10, deg(end_big_angle), deg(end_big_angle), width=w+15, color="black", lw=0.5))

    # small wedges
    start = end_big_angle - big_angle + small_angle
    for vsurv, vnonsurv in zip(mapped_surv_values, mapped_nonsurv_values):
        ax.add_patch(Wedge((0, 0), vsurv, deg(start), deg(start+small_angle), width=vsurv-inner_radius, color=terminal_color["survivor"], lw=0))
        ax.add_patch(Wedge((0, 0), vnonsurv, deg(start+small_angle), deg(start+2*small_angle), width=vnonsurv-inner_radius, color=terminal_color["nonsurvivor"], lw=0))
        start += (2 * small_angle)
    
    # lables (hours)
    labels_texts = data["time"]
    labels_angles = end_big_angle - big_angle + 2*small_angle + 2*small_angle * np.arange(0, len(labels_texts))
    xr = 130 * np.cos(np.array(labels_angles))
    yr = 130 * np.sin(np.array(labels_angles))
    labels_angles[labels_angles < -np.pi/2] += np.pi # easier to read labels on the left side
    for k, labl in enumerate(labels_texts):
        # ax.add_patch(Wedge((xr[k], yr[k]), 1, 0, 360, color="black", lw=0))
        ax.text(xr[k], yr[k], labl[:-6], rotation=deg(labels_angles[k]), fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

    # minor radial axes
    start = end_big_angle - big_angle + 2*small_angle
    for i in range(6):
        s2 = start + small_angle
        ax.add_patch(Wedge((0, 0), outer_radius+3, deg(start), deg(start), width=w+6, color="white", lw=0.05))
        ax.add_patch(Wedge((0, 0), outer_radius, deg(s2), deg(s2), width=w, color=color_big, lw=0.3))
        start += (2 * small_angle)


starting_end_angle = np.pi/2 - big_angle/2
for i, tp in enumerate(["V_D", "V_R"]):
    flags = ["noflag", "yellow", "red"] if tp == "V_D" else ["red", "yellow", "noflag"]
    # flags = ["noflag"]
    for j, flag in enumerate(flags):
        mapped_nonsurv_values = rmap(100 * np.array(data["nonsurvivors"][tp][flag]))
        mapped_surv_values = rmap(100 * np.array(data["survivors"][tp][flag]))
        big_wedge_idx = i * 3 + j
        end_big_angle1 = starting_end_angle - big_wedge_idx * big_angle
        wedge_hist(end_big_angle1, mapped_surv_values, mapped_nonsurv_values, color_big=flag_color[flag], ax=ax)

# last radial axis leftout and the divider vertical one
ax.add_patch(Wedge((0, 0), outer_radius+10, deg(np.pi/2 + big_angle/2), deg(np.pi/2 + big_angle/2), width=w+15, color="black", lw=0.5))
ax.add_patch(Wedge((0, 0), outer_radius+25, -90, -90, width=w+45, color="black", lw=1.5))

# circular axes and lables
labels = np.arange(0, 101, 20)
radii = rmap(labels)
for i, r in enumerate(radii):
    ax.add_patch(Wedge((0, 0), r, 0, 360, width=0, color="white", lw=0.05))
    ax.text(0, r+4, str(labels[i]) + r"%", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

# terminal legends (surv/nonsurv)
ax.add_patch(Wedge((-psize+7, psize-8), 4, 0, 360, color=terminal_color["survivor"], lw=0))
ax.text(-psize+15, psize-8, r"Survivors", fontsize=fontsize, horizontalalignment='left', verticalalignment='center')
ax.add_patch(Wedge((-psize+7, psize-20), 4, 0, 360, color=terminal_color["nonsurvivor"], lw=0))
ax.text(-psize+15, psize-20, r"Nonsurvivors", fontsize=fontsize, horizontalalignment='left', verticalalignment='center')
# ax.add_patch(Wedge((-psize+75, psize-10), 3, 0, 360, color=terminal_color["nonsurvivor"], lw=0))
# ax.text(-psize+80, psize-10, r"Nonsurvivors", fontsize=fontsize, horizontalalignment='left', verticalalignment='center')

# flag legends (red, yellow, noflag)
ax.add_patch(Rectangle((-30, 8), 16, 8, color=flag_color["noflag"], lw=0))
ax.text(-10, 8, r"No-flag", fontsize=fontsize, horizontalalignment='left', verticalalignment='bottom')
ax.add_patch(Rectangle((-30, -4), 16, 8, color=flag_color["yellow"], lw=0))
ax.text(-10, -4, r"Yellow", fontsize=fontsize, horizontalalignment='left', verticalalignment='bottom')
ax.add_patch(Rectangle((-30, -16), 16, 8, color=flag_color["red"], lw=0))
ax.text(-10, -16, r"Red", fontsize=fontsize, horizontalalignment='left', verticalalignment='bottom')

# V legends
ax.text(-psize+15, -psize+15, r"R-Network", fontsize=fontsize, horizontalalignment='left', verticalalignment='center')
ax.text(psize-65, -psize+15, r"D-Network", fontsize=fontsize, horizontalalignment='left', verticalalignment='center')

# time arch
m = np.pi/2 - big_angle/2
ax.add_patch(Wedge((0, 0), outer_radius+21, deg(m - 6*small_angle), deg(m - 2*small_angle), color="gray", width=0, lw=2))
ax.text(psize-63, psize-27, r"Time (Hours)", fontsize=fontsize, horizontalalignment='left', verticalalignment='center', color="black")

# fig.tight_layout()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(r"./plots/circular_hist.pdf", dpi=300)
