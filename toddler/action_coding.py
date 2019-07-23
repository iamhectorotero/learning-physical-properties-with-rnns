import numpy as np

NO_OP = 0
ACCELERATE_IN_X = 1
ACCELERATE_IN_Y = 2
DECELERATE_IN_X = 3
DECELERATE_IN_Y = 4
CLICK = 5
FIRST_ANSWER = 6
SECOND_ANSWER = 7
THIRD_ANSWER = 8
BORDER = 0.25
MAX_X = 6.0 - BORDER
MAX_Y = 4.0 - BORDER
MIN_X = MIN_Y = BORDER
TIME_CONSTANT = 1./60

mass_answers = {FIRST_ANSWER:"A is heavier", SECOND_ANSWER:"B is heavier", THIRD_ANSWER:"same"}
mass_answers_idx = {"A is heavier": FIRST_ANSWER, "B is heavier": SECOND_ANSWER, "same": THIRD_ANSWER}
force_answers = {FIRST_ANSWER:"attract", SECOND_ANSWER:"repel", THIRD_ANSWER:"none"}
force_answers_idx = {"attract": FIRST_ANSWER, "repel": SECOND_ANSWER, "none": THIRD_ANSWER}

def update_velocity(vx, vy, action):
    velocity_decrease = 2.
    velocity_increment = 0.25

    if action == NO_OP or action == CLICK:
        vx /= velocity_decrease
        vy /= velocity_decrease
    elif action == ACCELERATE_IN_X:
        vx += velocity_increment
        #vy /= velocity_decrease
    elif action == ACCELERATE_IN_Y:
        vy += velocity_increment
        # vx /= velocity_decrease
    elif action == DECELERATE_IN_X:
        vx -= velocity_increment
        # vy /= velocity_decrease
    elif action == DECELERATE_IN_Y:
        vy -= velocity_increment
        # vx /= velocity_decrease
    return vx, vy

def update_position(pos, vel, max_pos, min_pos):

    pos += vel * TIME_CONSTANT
    if pos > max_pos:
        pos = max_pos
        vel = 0.
    if pos < min_pos:
        pos = min_pos
        vel = 0.

    return pos, vel

def get_mouse_action(action, x_pos, y_pos, vx, vy, mouse_click, object_in_control):
    vx, vy = update_velocity(vx, vy, action)
    x_pos, vx = update_position(x_pos, vx, MAX_X, MIN_X)
    y_pos, vy = update_position(y_pos, vy, MAX_Y, MIN_Y)

    click = (action == CLICK)
    if object_in_control != 0:
        click = not click

    return click, x_pos, y_pos, vx, vy
