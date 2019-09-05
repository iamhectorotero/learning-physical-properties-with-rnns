from .constants import YOKED_TRAINING_COLS, BASIC_TRAINING_COLS, MOUSE_COLS, MOUSE_POSITION_COLS, MOUSE_CONTROL_COLS, S_PER_FRAME
import numpy as np

def add_mouse_columns_to_passive_trials(trials):
    """ Adds the mouse position and control columns for each trial in the list by setting them to
    a constant value for each trial step.
    Args:
        trials: a list of Pandas DataFrames.
    """
    for trial in trials:
        for column in MOUSE_POSITION_COLS:
            trial[column] = 0.
        for column in MOUSE_CONTROL_COLS:
            if column == "C_none":
                trial[column] = True
            else:
                trial[column] = False


def set_mouse_velocities(trials):
    """Adds the mouse velocity (mouse.vx and mouse.vy) for each trial in the list. The velocity is
    calculated from the position shift in consecutive frames.
    Args:
        trials: a list of Pandas DataFrames.
    """
    for trial in trials:
        trial[list(MOUSE_POSITION_COLS)]  /= 100
        positions = trial[list(MOUSE_POSITION_COLS)]

        shifted_positions = positions.shift(1).fillna(positions.iloc[0])
        trial[["mouse.vx", "mouse.vy"]] = (positions - shifted_positions) / S_PER_FRAME


def transform_velocity_to_speed_and_angle(trials):
    OBJECTS = ["o1", "o2", "o3", "o4", "mouse"]
    
    for trial in trials:
        for i, obj in enumerate(OBJECTS):
            trial[obj+".speed"] = np.sqrt(trial[obj+".vx"]**2 + trial[obj+".vy"]**2)        
            trial[obj+".angle"] = np.arctan(trial[obj+".vy"] / trial[obj+".vx"])
            trial[obj+".angle"].fillna(0, inplace=True)
"""


# TODO: Change method name to "square distances" or change functionality to match name
def get_distances_between_objects(trials):
    """Finds the square distance between every pair of pucks. To do so, applies the Pythagoras'
    theorem between the points defined by the coordinates of the pucks e.g. (o1.x, o1.y).
    Args:
        trials: a list of Pandas DataFrames."""

    OBJECTS = ["o1", "o2", "o3", "o4"]

    for trial in trials:
        for i, obj_one in enumerate(OBJECTS):
            for obj_two in OBJECTS[i+1:]:
                dist_x = trial[obj_one+".x"] - trial[obj_two+".x"]
                dist_y = trial[obj_one+".y"] - trial[obj_two+".y"]
                dist_mag = dist_x**2 + dist_y**2
                trial["d2_"+obj_one+"_"+obj_two] = dist_mag

def get_angle_between_objects_features(trials):
    OBJECTS = ["o1", "o2", "o3", "o4"]

    for trial in trials:
        for i, obj_one in enumerate(OBJECTS):
            for obj_two in OBJECTS[i+1:]:
                dist_x = np.abs(trial[obj_one+".x"] - trial[obj_two+".x"])
                dist_y = np.abs(trial[obj_one+".y"] - trial[obj_two+".y"])
                hyp = np.sqrt(dist_x**2 + dist_y**2)
                trial["cos_"+obj_one+"_"+obj_two] = dist_x / hyp
                trial["sin_"+obj_one+"_"+obj_two] = dist_y / hyp
