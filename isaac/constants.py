BASIC_TRAINING_COLS = [o+"."+attr for o in ["o1", "o2", "o3", "o4"] for attr in ["x", "y", "vx", "vy"]]
YOKED_TRAINING_COLS = BASIC_TRAINING_COLS + ["idControlledObject", "mouseX", "mouseY"]
FORCE_CLASS_COLS = ["attract", "none", "repel"]
MASS_CLASS_COLS = ["A", "B", "same"]