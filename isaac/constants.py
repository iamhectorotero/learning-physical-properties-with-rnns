BASIC_TRAINING_COLS = [o+"."+attr for o in ["o1", "o2", "o3", "o4"] for attr in ["x", "y", "vx", "vy"]]
YOKED_TRAINING_COLS = BASIC_TRAINING_COLS + ["C_none", "C_O1", "C_O2", "C_O3", "C_O4", "mouseX", "mouseY"]
FORCE_CLASS_COLS = ["attract", "none", "repel"]
MASS_CLASS_COLS = ["A", "B", "same"]