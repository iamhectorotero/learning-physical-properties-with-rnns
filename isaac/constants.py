BASIC_TRAINING_COLS = [o+"."+attr for o in ["o1", "o2", "o3", "o4"] for attr in ["x", "y", "vx", "vy"]]
MOUSE_CONTROL_COLS = ["C_none", "C_O1", "C_O2", "C_O3", "C_O4"]
MOUSE_POSITION_COLS = ["mouseX", "mouseY"]
MOUSE_COLS = MOUSE_CONTROL_COLS + MOUSE_POSITION_COLS
YOKED_TRAINING_COLS = BASIC_TRAINING_COLS + MOUSE_COLS
FORCE_CLASS_COLS = ["attract", "none", "repel"]
MASS_CLASS_COLS = ["A", "B", "same"]
S_PER_FRAME = 1./60