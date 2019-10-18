import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import gizeh as gz
import math
import gizeh as gz
import moviepy.editor as mpy

from .dataset import read_dataset
from .utils import plot_confusion_matrix


def increase_linewidth(ax):
    lines = ax.get_lines()
    for line in lines:
        line.set_linewidth(3)

    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 0:
        leg = ax.legend()
        leg_lines = leg.get_lines()
        plt.setp(leg_lines, linewidth=5)

    plt.tight_layout()


def plot_lineplot_with_paper_style(save_plot_path, **lineplot_kwargs):
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.rcParams.update({'axes.labelsize': '22',
                         'xtick.labelsize':'18',
                         'ytick.labelsize': '18',
                         'legend.fontsize': '18',
                         'figure.figsize': (8, 8)})

    if "ax" not in lineplot_kwargs:
        plt.clf()

    ax = sns.lineplot(**lineplot_kwargs)
    increase_linewidth(ax)

    if save_plot_path is not None:
        plt.savefig(save_plot_path)
    return ax


def plot_confusion_matrix_given_predicted_and_test_loader(predicted, test_loader, class_columns,
                                                          save_plot_path=None, multiclass_index=None):
    """ Plots the confusion matrix given the model's prediction and test_loader".
    Args:
        predicted: the model's predictions for the test_loader.
        test_loader: the dataset's loader the model has been evaluated on.
        class_columns: the names of the classes as they will appear in the confusion matrix.
        save_plot_path: If not None, the confusion matrix will be saved to this path.
        multiclass_index: If not None, it indicates the position of the branch that is being
                          evaluated. This index will be used to extract the corresponding columns
                          from the test_loader.
    """

    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams.update({'axes.labelsize': '22',
                         'xtick.labelsize':'18',
                         'ytick.labelsize': '18',
                         'legend.fontsize': '18',
                         'figure.figsize': (8, 8),
                         'font.size': 22})

    predicted = [pred.cpu() for pred in predicted]

    if multiclass_index is None:
        Y_test = np.concatenate([y.cpu().numpy() for x, y in test_loader])
    else:
        Y_test = np.concatenate([y[:, multiclass_index].cpu().numpy() for x, y in test_loader])

    ax = plot_confusion_matrix(Y_test, predicted, classes=class_columns, normalize=True)

    if save_plot_path is not None:
        plt.savefig(save_plot_path)

    return ax


def smooth_out_rl_stats(dataframe, columns_to_smooth, window_size=500):
    rolling_stats = []

    for seed, df in dataframe.groupby("seed"):
        rolling_df = pd.DataFrame(columns=columns_to_smooth)
        for column in columns_to_smooth:
            rolling_df[column] = df[column].rolling(window=window_size).mean()
        rolling_stats.append(rolling_df)
    rolling_stats = pd.concat(rolling_stats)

    rolling_stats["Episode"] = rolling_stats.index
    return rolling_stats


def make_frame_curried(this_data, n_bodies=4):
    def make_frame(t):
        labels = ['A', 'B', '', '']
        centers = np.array(['o1','o2','o3','o4'][:n_bodies])
        colors = [(1,0,0),(0,1,0),(0,0,1),(0,0,1)]
        RATIO = 100
        RAD = 25
        W = 600
        H = 400
        # H_outer = 500
        N_OBJ = 4

        frame = int(math.floor(t*60))
        # Essentially pauses the action if there are no more frames and but more clip duration
        # if frame >= len(this_data["co"]):
        #    frame = len(this_data["co"])-1

        # White background
        surface = gz.Surface(W,H, bg_color=(1,1,1))

        # Walls
        wt = gz.rectangle(lx=W, ly=20, xy=(W/2,10), fill=(0,0,0))#, angle=Pi/8
        wb = gz.rectangle(lx=W, ly=20, xy=(W/2,H-10), fill=(0,0,0))
        wl = gz.rectangle(lx=20, ly=H, xy=(10,H/2), fill=(0,0,0))
        wr = gz.rectangle(lx=20, ly=H, xy=(W-10,H/2), fill=(0,0,0))
        wt.draw(surface)
        wb.draw(surface)
        wl.draw(surface)
        wr.draw(surface)

        # Pucks
        for label, color, center in zip(labels, colors, centers):

            xy = np.array([this_data[center+'.x'].iloc[frame]*RATIO, this_data[center+'.y'].iloc[frame]*RATIO])

            ball = gz.circle(r=RAD, fill=color).translate(xy)
            ball.draw(surface)

            # Letters
            text = gz.text(label, fontfamily="Helvetica",  fontsize=25, fontweight='bold', fill=(0,0,0), xy=xy) #, angle=Pi/12
            text.draw(surface)

        # Mouse cursor
        if 'mouseY' in this_data.columns:
            cursor_xy = np.array([this_data['mouseX'].iloc[frame]*RATIO, this_data['mouseY'].iloc[frame]*RATIO])
        else:
            cursor_xy = np.array([0, 0])

        cursor = gz.text('+', fontfamily="Helvetica",  fontsize=25, fill=(0,0,0), xy=cursor_xy) #, angle=Pi/12
        cursor.draw(surface)

        # Control
        if "idControlledObject" in this_data.columns and this_data['idControlledObject'].iloc[frame] != "none":
            if this_data['idControlledObject'].iloc[frame]=="object1":
                xy = np.array([this_data['o1.x'].iloc[frame]*RATIO, this_data['o1.y'].iloc[frame]*RATIO])
            elif this_data['idControlledObject'].iloc[frame]=="object2":
                xy = np.array([this_data['o2.x'].iloc[frame]*RATIO, this_data['o2.y'].iloc[frame]*RATIO])
            elif this_data['idControlledObject'].iloc[frame]=="object3":
                xy = np.array([this_data['o3.x'].iloc[frame]*RATIO, this_data['o3.y'].iloc[frame]*RATIO])
            elif this_data['idControlledObject'].iloc[frame]=="object4":
                xy = np.array([this_data['o4.x'].iloc[frame]*RATIO, this_data['o4.y'].iloc[frame]*RATIO])

            # control_border = gz.arc(r=RAD, a1=0, a2=np.pi, fill=(0,0,0)).translate(xy)
            control_border = gz.circle(r=RAD, stroke_width=2).translate(xy)
            control_border.draw(surface)

        return surface.get_npimage()

    return make_frame


if __name__ == "__main__":

    random_trial = np.random.randint(0, 1600)
    random_trial = 0
    data = pd.read_hdf("../data/passive_trials.h5", key="trial_"+str(random_trial))

    print(data.head())
    make_frame = make_frame_curried(data)
    duration = data.shape[0]/60
    clip = mpy.VideoClip(make_frame, duration=duration)

    # Create the filename (adding 0s to ensure things are in a nice alphabetical order now)
    writename = "trial_"+str(random_trial)+".mp4"

    # Write the clip to file
    clip.write_videofile(writename, fps=24)

