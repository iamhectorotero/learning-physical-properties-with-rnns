POSITION_COLS = tuple([o+"."+attr for o in ["o1", "o2", "o3", "o4"] for attr in ["x", "y"]])
BASIC_TRAINING_COLS = tuple([o+"."+attr for o in ["o1", "o2", "o3", "o4"] for attr in ["x", "y", "vx", "vy"]])
ALT_BASIC_TRAINING_COLS = tuple([o+"."+attr for o in ["o1", "o2", "o3", "o4"] for attr in ["x", "y", "speed", "angle"]])
PUCK_DISTANCES = ('d_o1_o2', 'd_o1_o3', 'd_o1_o4', 'd_o2_o3', 'd_o2_o4', 'd_o3_o4')
PUCK_SQUARE_DISTANCES = ('d2_o1_o2', 'd2_o1_o3', 'd2_o1_o4', 'd2_o2_o3', 'd2_o2_o4', 'd2_o3_o4')
PUCK_ANGLE_FEATURES = ('cos_o1_o2', 'sin_o1_o2', 'cos_o1_o3', 'sin_o1_o3', 'cos_o1_o4', 'sin_o1_o4', 
                       'cos_o2_o3', 'sin_o2_o3', 'cos_o2_o4', 'sin_o2_o4', 'cos_o3_o4', 'sin_o3_o4')
MOUSE_CONTROL_COLS = ("C_none", "C_O1", "C_O2", "C_O3", "C_O4")
MOUSE_POSITION_COLS = ("mouseX", "mouseY")
MOUSE_COLS = MOUSE_CONTROL_COLS + MOUSE_POSITION_COLS
MOUSE_VEL_COLS = ("mouse.vx", "mouse.vy")
MOUSE_SPEED_AND_ANGLE = ("mouse.speed", "mouse.angle")
MOUSE_ADV_COLS = MOUSE_COLS + MOUSE_VEL_COLS
YOKED_TRAINING_COLS = BASIC_TRAINING_COLS + MOUSE_COLS
FORCE_CLASS_COLS = ("attract", "none", "repel")
MASS_CLASS_COLS = ("A", "B", "same")
S_PER_FRAME = 1./60