"""
Utilities
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def plot(data={}, loc="visualization.pdf", x_label="", y_label="", title="", kind='line',
         legend=True, index_col=None, clip=None, moving_average=False):
    if all([len(data[key]) > 1 for key in data]):
        if moving_average:
            smoothed_data = {}
            for key in data:
                smooth_scores = [np.mean(data[key][max(0, i - 10):i + 1]) for i in range(len(data[key]))]
                smoothed_data['smoothed_' + key] = smooth_scores
                smoothed_data[key] = data[key]
            data = smoothed_data
        df = pd.DataFrame(data=data)
        ax = df.plot(kind=kind, legend=legend, ylim=clip)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(loc)
        plt.close()


def write_to_csv(data={}, loc="data.csv"):
    if all([len(data[key]) > 1 for key in data]):
        df = pd.DataFrame(data=data)
        df.to_csv(loc)


class Font:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bgblue = '\033[44m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'
