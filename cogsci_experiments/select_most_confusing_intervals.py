import isaac.constants
isaac.constants.TQDM_DISABLE = True

from torch import nn
from torch.nn import Softmax
from isaac.utils import get_cuda_device_if_available
import joblib

from isaac.dataset import read_dataset, prepare_dataset
from isaac.models import MultiBranchModel, ComplexRNNModel
from isaac.constants import BASIC_TRAINING_COLS, MASS_CLASS_COLS, FORCE_CLASS_COLS, RTHETA_COLS, XY_RTHETA_COLS, XY_VXVY_RTHETA_COLS
from isaac.evaluation import predict_with_a_group_of_saved_models, evaluate_saved_model

import torch
import glob
from torch.autograd import Variable
import numpy as np
import pandas as pd
from tqdm import tqdm

SECONDS_PER_WINDOW = 5 # seconds
FPS = 60
STEP_SIZE = 3
# PD_STEP_SIZE = 10
SEQ_END = 2700

device = get_cuda_device_if_available()
print(device)

HIDDEN_DIM = 25  # hidden layer dimension
OUTPUT_DIM = 3   # output dimension
DROPOUT = 0.5

normalise_data = True
model_name = "xy_vxvy"
d = {"rtheta": RTHETA_COLS, "xy_vxvy": BASIC_TRAINING_COLS,
     "xy_rtheta": XY_RTHETA_COLS, "xy_vxvy_rtheta": XY_VXVY_RTHETA_COLS}
training_columns = d[model_name]

dataset_path = "data/test_passive_trials.h5"
multiclass = True
DATASET = read_dataset(dataset_path)


def get_question_predictions_for_group_of_models(question_type):

    if not multiclass:
        class_columns = FORCE_CLASS_COLS if question_type == "force" else MASS_CLASS_COLS
        models = sorted(glob.glob("models/GRU_singlebranch/"+model_name+"/best_"+question_type+"_model_seed_*.pt"))
        scaler_path = "scalers/GRU_singlebranch/"+question_type+"_"+model_name+"_scaler.sk"
        model_arch = ComplexRNNModel
        network_dims = (len(training_columns), HIDDEN_DIM, N_LAYERS, OUTPUT_DIM, DROPOUT)

    else:
        class_columns = [list(MASS_CLASS_COLS), list(FORCE_CLASS_COLS)]
        models = sorted(glob.glob("models/"+model_name+"/best_"+question_type+"_model_seed_*.pt"))
        scaler_path = "scalers/passive_"+model_name+"_scaler.sk"
        model_arch = MultiBranchModel
        network_dims = (len(training_columns), HIDDEN_DIM, OUTPUT_DIM, DROPOUT)


    predictions = predict_with_a_group_of_saved_models(tqdm(models), network_dims, None,
                                                       training_columns=training_columns,
                                                       class_columns=class_columns, step_size=STEP_SIZE,
                                                       seq_end=SEQ_END, scaler_path=scaler_path,
                                                       arch=model_arch, multiclass=multiclass, trials=DATASET,
                                                       predict_rolling_windows=True, seconds_per_window=SECONDS_PER_WINDOW)
    predictions = torch.stack(predictions)
    if multiclass:
        if question_type == "mass":
            predictions = predictions[:, :, :, 0]
        else:
            predictions = predictions[:, :, :, 1]

    return predictions

print("MASS")
question_type = "mass"
group_mass_seq_prediction = get_question_predictions_for_group_of_models(question_type)

print("\nFORCE")
question_type = "force"
group_force_seq_prediction = get_question_predictions_for_group_of_models(question_type)

mass_solutions = [trial[list(MASS_CLASS_COLS)].idxmax(axis=1).unique()[0] for trial in DATASET]
force_solutions = [trial[list(FORCE_CLASS_COLS)].idxmax(axis=1).unique()[0] for trial in DATASET]

s = Softmax(dim=-1)
group_force_seq_prediction = s(group_force_seq_prediction)
group_mass_seq_prediction = s(group_mass_seq_prediction)

avg_force_seq_prediction = torch.mean(group_force_seq_prediction, dim=0)
avg_mass_seq_prediction = torch.mean(group_mass_seq_prediction, dim=0)

n_trials = avg_force_seq_prediction.shape[0]
n_windows = avg_force_seq_prediction.shape[1]
window_second_start = [i for _ in range(n_trials) for i in range(1, n_windows+1)]
trial_number = [i for i in range(n_trials) for _ in range(1, n_windows+1)]

mass_df = pd.DataFrame(data=avg_mass_seq_prediction.reshape(n_trials*n_windows, 3).numpy(),
                       columns=["rnn_%s" % cl for cl in MASS_CLASS_COLS])
mass_df["window_second_start"] = window_second_start
mass_df["trial_number"] = trial_number
mass_df["solution"] = [mass_solutions[trial_id] for trial_id in trial_number]

force_df = pd.DataFrame(data=avg_force_seq_prediction.reshape(n_trials*n_windows, 3).numpy(),
                       columns=["rnn_%s" % cl for cl in FORCE_CLASS_COLS])
force_df["window_second_start"] = window_second_start
force_df["trial_number"] = trial_number
force_df["solution"] = [force_solutions[trial_id] for trial_id in trial_number]

def entropy_rnn_responses(responses):
    responses = responses.tolist()
    return -np.sum(responses * np.log2(responses))

def get_probabilities_df(df, task):
    rows = []

    class_columns = FORCE_CLASS_COLS if task == "force" else MASS_CLASS_COLS
    rnn_columns = ["rnn_" + cl for cl in class_columns]


    for i in range(df.shape[0]):
        row = df.iloc[i]
        window_second_start = row.window_second_start
        trial_number = row.trial_number
        solution = row.solution

        rnn_solution_probability = row["rnn_"+solution]
        rnn_preferred_option = class_columns[np.argmax(row[rnn_columns].values)]
        rnn_preferred_option_probability = max(row[rnn_columns])
        rnn_entropy = entropy_rnn_responses(row[rnn_columns].values)

        wrong_options = ["rnn_" + cl for cl in class_columns if cl != solution]
        max_probability_for_a_wrong_option = max(row[wrong_options])


        rows.append([window_second_start, trial_number, solution,
                    rnn_solution_probability, rnn_preferred_option, rnn_preferred_option_probability,
                    rnn_entropy, max_probability_for_a_wrong_option])


    df = pd.DataFrame(data=rows, columns=["window_second_start", "trial_number", "solution", "rnn_solution_probability",
                                         "rnn_preferred_option", "rnn_preferred_option_probability", "entropy",
                                         "max_probability_for_a_wrong_option"])
    return df

confused_mass_dfs = get_probabilities_df(mass_df, "mass")
confused_force_dfs = get_probabilities_df(force_df, "force")

confused_mass_dfs = confused_mass_dfs.sort_values(by="max_probability_for_a_wrong_option", ascending=False)
confused_force_dfs = confused_force_dfs.sort_values(by="max_probability_for_a_wrong_option", ascending=False)

from isaac.visualization import make_frame_curried
import moviepy.editor as mpy
from scipy import misc


def make_clip(trial_data, window_second_start, solution, rnn_thinks_this_is, rnn_confidence):

    trial_data = trial_data.iloc[window_second_start*FPS:(window_second_start + SECONDS_PER_WINDOW)*FPS]
    duration = len(trial_data)
    n_bodies = sum(["o"+str(i)+".x" in list(trial_data.columns) for i in range(1, 5)])

    while (len(trial_data) + 1) % 60 != 0:
        trial_data = trial_data.append(trial_data.iloc[-1], ignore_index=True)

    make_frame = make_frame_curried(trial_data, n_bodies, None, None)
    clip = mpy.VideoClip(make_frame, duration=duration / 60)
    return clip, trial_data

import os
import json

CONFUSING_DATA_PATH = "cogsci_images/confusing_videos/confusing_%s" % model_name
os.makedirs(CONFUSING_DATA_PATH, exist_ok=True)
FILENAME = "confusing_%s_interval_in_trial_%d_sec_%d_to_%d_rnn_thinks_%s__prob_%.4f_while_solution_is_%s_prob_%.4f"
VIDEO_PATH = os.path.join(CONFUSING_DATA_PATH, FILENAME + ".mp4")
JSON_PATH = os.path.join(CONFUSING_DATA_PATH, "confusing_"+model_name+"_%s_physics_data.json")
JSON_DATA = []
CSV_DATA = [["trial_number", "window_start", "window_end", "solution", "rnn_preferred_option",
             "rnn_preferred_option_probability", "rnn_solution_probability"]]
CSV_PATH = os.path.join(CONFUSING_DATA_PATH, "confusing_"+model_name+"_%s_interval_description.csv")

def write_confused_intervals(confused_df, question_type, solution):

    replays = read_dataset("data/test_passive_trials.h5")
    print(len(replays))
    written_replays = {}
    number_of_written_replays = 0


    confused_df = confused_df.query("solution == '%s'" % solution).copy()

    for row_i in range(confused_df.shape[0]):

        window_second_start, trial_number, solution, rnn_preferred_option, rnn_preferred_option_probability, rnn_solution_probability = (
            confused_df.iloc[row_i][["window_second_start", "trial_number", "solution", "rnn_preferred_option",
                                     "rnn_preferred_option_probability", "rnn_solution_probability"]])

        window_second_end = window_second_start + SECONDS_PER_WINDOW

        print("RNN thinks the interval (%d, %d) in trial %d is %s with %.4f confidence. In reality, it is %s (%.4f)." % (
            window_second_start, window_second_end, trial_number,
            rnn_preferred_option, rnn_preferred_option_probability, solution, rnn_solution_probability))


        # DON'T SAVE AN INTERVAL IF IT REFRESHES
        window_frame_start = window_second_start * 60
        window_frame_end = window_second_end * 60
        frames_in_which_the_replay_refreshes = replays[trial_number].refreshes[0] if\
                                               "refreshes" in replays[trial_number].columns\
                                                else []
        frames_in_which_the_replay_refreshes = [frames_in_which_the_replay_refreshes] if\
                                                    type(frames_in_which_the_replay_refreshes) == np.int64\
                                                else frames_in_which_the_replay_refreshes

        interval_refreshes = False
        for frame in frames_in_which_the_replay_refreshes:
            if window_frame_end > frame > window_frame_start:
                interval_refreshes = True
                break
        if interval_refreshes:
            print("Skipping trial %d and start %d because of interval refreshes" % (trial_number, window_second_start))
            continue

        # DON'T SAVE AN INTERVAL IF THERE'S AN OVERLAPPING INTERVAL ALREADY SAVED
        if trial_number in written_replays:
            overlapping_replay = False
            for already_written_window_start in written_replays[trial_number]:
                if abs(already_written_window_start - window_second_start) <= 5:
                    overlapping_replay = True
                    break
            if overlapping_replay:
                print("Skipping trial %d and start %d because of interval overlap" % (trial_number, window_second_start))
                continue

        # SAVE THE INTERVAL
        written_replays[trial_number] = written_replays.get(trial_number, []) + [window_second_start]
        clip, trial_data = make_clip(replays[trial_number], window_second_start, solution, rnn_preferred_option, rnn_preferred_option_probability)
        clip.ipython_display(fps=60)
        clip.write_videofile(VIDEO_PATH % (
            question_type, trial_number, window_second_start, window_second_end,
            rnn_preferred_option, rnn_preferred_option_probability, solution, rnn_solution_probability), fps=60)

        trial_data = trial_data.to_dict(orient='list')
        # Simplify attributes whose values are unique throughout the list
        for key in ["trial_type", "condition_world_variant", "tM", "tR", "world_id"]:
            trial_data[key] = trial_data[key][0]
        JSON_DATA.append(trial_data)
        CSV_DATA.append([trial_number, window_second_start, window_second_end, solution, rnn_preferred_option,
                         rnn_preferred_option_probability, rnn_solution_probability])

        number_of_written_replays += 1
        if number_of_written_replays == 5:
            break

    print(number_of_written_replays)

write_confused_intervals(confused_force_dfs, "force", "attract")
write_confused_intervals(confused_force_dfs, "force", "repel")
write_confused_intervals(confused_force_dfs, "force", "none")

with open(JSON_PATH % "force", "w+") as f: 
    json.dump(JSON_DATA, f)
    
interval_descriptions = pd.DataFrame(data=CSV_DATA[1:], columns=CSV_DATA[0])
interval_descriptions.to_csv(CSV_PATH % "force", index=False)


JSON_DATA = []
CSV_DATA = [["trial_number", "window_start", "window_end", "solution", "rnn_preferred_option",
             "rnn_preferred_option_probability", "rnn_solution_probability"]]

write_confused_intervals(confused_mass_dfs, "mass", "A")
write_confused_intervals(confused_mass_dfs, "mass", "B")
write_confused_intervals(confused_mass_dfs, "mass", "same")

with open(JSON_PATH % "mass", "w+") as f:
    json.dump(JSON_DATA, f)

interval_descriptions = pd.DataFrame(data=CSV_DATA[1:], columns=CSV_DATA[0])
interval_descriptions.to_csv(CSV_PATH % "mass",index=False)
