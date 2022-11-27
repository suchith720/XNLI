import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.interpolate import make_interp_spline

def display_metrics(metrics):
    disp_metrics = {}
    for k, v in metrics.items():
        m = {}
        for l, acc in v.items():
            t = tuple(l.split(maxsplit=1))
            m[t] = acc
        disp_metrics[k] = m
    return disp_metrics

def pretty_epoch_metrics(epoch, train_history, valid_history, train_time, valid_time):
    epoch_metric = { epoch : {"Training Loss": train_history["loss"][-1],
                                    "Training Micro f1": train_history["accuracy"][-1],
                                    "Training Macro f1": train_history["macro_f1"][-1],
                                    "Training Average f1": train_history["average_f1"][-1],
                                    "Training Time (secs)": train_time[-1],
                                    "Validation Loss": valid_history["loss"][-1],
                                    "Validation Micro f1":valid_history["accuracy"][-1],
                                    "Validation Macro f1": valid_history["macro_f1"][-1],
                                    "Validation Average f1": valid_history["average_f1"][-1],
                                    "Validation Time (secs)": valid_time[-1]
                                   }}
    return epoch_metric


def pretty_metrics(epoch, train_history, valid_history):
    metric = []
    for i, name in enumerate(train_history):
        metric.append({})
        metric[-1][name] = {}
        for lang_str in train_history[name]:
            metric[-1][name][f'Training {lang_str}'] = train_history[name][lang_str][-1]
            metric[-1][name][f'Validation {lang_str}'] = valid_history[name][lang_str][-1]
    return metric


def get_smooth_curve(x, y):
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)
    return X_, Y_


def plot_metrics(train_metrics, valid_metrics, figsize=(16, 16), pc=0.2, save_file=None):
    assert len(train_metrics) == len(valid_metrics), "Training and Validation metric mismatch"

    palette = iter(sns.husl_palette(len(train_metrics)*2))

    fig, axes = plt.subplots(len(train_metrics), 2, figsize=figsize)
    for i, name in enumerate(train_metrics):

        x, y = np.arange(len(train_metrics[name])), np.array(train_metrics[name])
        sns.lineplot(ax=axes[i, 0], y=y, x=x, color=next(palette))

        x = np.arange(len(x), step=int(len(x)*pc))
        y = y[x]
        X, Y = get_smooth_curve(x, y)
        sns.lineplot(ax=axes[i, 0], y=Y, x=X)
        axes[i, 0].set(xlabel='iterations', ylabel=name, title=f'TRAINING {name.upper()}')

        x, y = np.arange(len(valid_metrics[name])), np.array(valid_metrics[name])
        sns.lineplot(ax=axes[i, 1], y=y, x=x, color=next(palette))

        x = np.arange(len(x), step=int(len(x)*pc))
        y = y[x]
        X, Y = get_smooth_curve(x, y)
        sns.lineplot(ax=axes[i, 1], y=Y, x=X)
        axes[i, 1].set(xlabel='iterations', ylabel=name, title=f'VALIDATION {name.upper()}')

        if save_file:
            if not os.path.exists(os.path.dirname(save_file)):
                raise Exception(f"{os.path.dirname(save_file)} does not exist.")
            plt.savefig(save_file)


def plot_avg_metrics(train_metrics, valid_metrics, figsize=(16, 4), save_file=None):
    assert len(train_metrics) == len(valid_metrics), "Training and Validation metric mismatch"

    palette = iter(sns.husl_palette(len(train_metrics)*2))

    fig, axes = plt.subplots(1, len(train_metrics), figsize=figsize)
    for i, name in enumerate(train_metrics):

        x = np.arange(len(train_metrics[name]))

        y = np.array(train_metrics[name])
        sns.lineplot(ax=axes[i], y=y, x=x, color=next(palette))

        y = np.array(valid_metrics[name])
        sns.lineplot(ax=axes[i], y=y, x=x, color=next(palette))

        axes[i].set(xlabel='epochs', ylabel=name, title=f'TRAIN/VALID {name.upper()}')
        axes[i].legend(['train', 'valid'])

    if save_file:
        if not os.path.exists(os.path.dirname(save_file)):
            raise Exception(f"{os.path.dirname(save_file)} does not exist.")
        plt.savefig(save_file)

