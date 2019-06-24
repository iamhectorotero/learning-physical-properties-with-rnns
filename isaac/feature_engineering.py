from isaac.constants import YOKED_TRAINING_COLS, BASIC_TRAINING_COLS, MOUSE_COLS, MOUSE_POSITION_COLS, MOUSE_CONTROL_COLS, S_PER_FRAME
import numpy as np

def add_mouse_columns_to_passive_trials(trials):
    
    for trial in trials:
        for column in MOUSE_POSITION_COLS:
            trial[column] = 0.
        for column in MOUSE_CONTROL_COLS:
            if column == "C_none":
                trial[column] = True
            else:
                trial[column] = False
            
def set_mouse_velocities(trials):
    
    for trial in trials:
        trial[MOUSE_POSITION_COLS]  /= 100
        positions = trial[MOUSE_POSITION_COLS]
        
        shifted_positions = positions.shift(1).fillna(positions.iloc[0])
        trial[["mouse.vx", "mouse.vy"]] = np.abs((shifted_positions - positions) / S_PER_FRAME)

def transform_velocity_to_speed_and_angle(trials):
    OBJECTS = ["o1", "o2", "o3", "o4", "mouse"]
    
    for trial in trials:
        for i, obj in enumerate(OBJECTS):
            trial[obj+".speed"] = np.sqrt(trial[obj+".vx"]**2 + trial[obj+".vy"]**2)        
            trial[obj+".angle"] = np.arctan(trial[obj+".vy"] / trial[obj+".vx"])
            trial[obj+".angle"].fillna(0, inplace=True)
        
def get_distances_between_objects(trials):
    OBJECTS = ["o1", "o2", "o3", "o4"]

    for trial in trials:
        for i, obj_one in enumerate(OBJECTS):
            for obj_two in OBJECTS[i+1:]:
                dist_x = trial[obj_one+".x"] - trial[obj_two+".x"]
                dist_y = trial[obj_one+".x"] - trial[obj_two+".x"]
                dist_mag = np.sqrt(dist_x**2 + dist_y**2)
                trial["d_"+obj_one+"_"+obj_two] = dist_mag
                