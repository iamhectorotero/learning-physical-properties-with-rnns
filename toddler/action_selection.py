import torch
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
from .action_coding import MIN_X, MAX_X, MIN_Y, MAX_Y, ACCELERATE_IN_X, ACCELERATE_IN_Y, DECELERATE_IN_X, DECELERATE_IN_Y, ANSWER_QUESTION

import gc


def e_greedy_action(state, valueNetwork, epsilon, t, current_pos=(None, None),
                    yoked_network=None, episode=None, device="cuda:0",
                    possible_actions=np.arange(0, 6), mouse_exploration_frames=None,
                    force_answer_at_t=None):

    assert len(possible_actions) == valueNetwork.output_dim
    assert force_answer_at_t is None or mouse_exploration_frames is None or force_answer_at_t > mouse_exploration_frames

    valueNetwork = valueNetwork.eval()
    action_values = valueNetwork(state.to(device=device))[0][0]
    greedy_action = torch.argmax(action_values).item()

    policy = []
    for i, a in enumerate(possible_actions):
        if i == greedy_action:
            policy.append(1 - epsilon + epsilon/len(possible_actions))
        else:
            policy.append(epsilon/len(possible_actions))

    x_pos, y_pos = current_pos
    if x_pos == MAX_X:
        policy[ACCELERATE_IN_X] = 0.
    elif x_pos == MIN_X:
        policy[DECELERATE_IN_X] = 0.

    if y_pos == MAX_Y:
        policy[ACCELERATE_IN_Y] = 0.
    elif y_pos == MIN_Y:
        policy[DECELERATE_IN_Y] = 0.

    if mouse_exploration_frames is not None and t < mouse_exploration_frames:
        if yoked_network is not None:
            policy[ANSWER_QUESTION] = 0.
        else:
            policy[-3:] = [0., 0., 0.]

    if force_answer_at_t is not None and t >= force_answer_at_t:
        policy[:6] = [0., 0., 0., 0., 0., 0.]
        if yoked_network is not None:
            policy[-1] = 1.
        else:
            answers_values = action_values[-3:].cpu().detach().numpy()
            policy[6:] = (answers_values == np.max(answers_values))

    policy = np.array(policy) / sum(policy)

    selected_action = np.random.choice(possible_actions, p=policy)

    if yoked_network is None or selected_action != ANSWER_QUESTION:
        return selected_action

    # Two network setting
    if len(episode[0]) > 0:
        states = episode[0]
        states = torch.cat(states, dim=1).to(device=device)
    else:
        states = deepcopy(state)

    features = [0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15]
    states = states[:, :, features]
    question_logits = yoked_network(states)[0]
    answer = ANSWER_QUESTION + torch.argmax(question_logits).detach().cpu().numpy()
    return answer

