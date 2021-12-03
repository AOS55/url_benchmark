import learn2learn as l2l
from copy import deepcopy


def meta_surrogate_loss(replay_dict, agent_dict, replay_valid_dict,
                        policy, baseline, tau, gamma, adapt_lr):
    mean_loss = 0.0
    mean_kl = 0.0
    for replay, agent in zip(replay_dict, agent_dict):
        new_policy = deepcopy(policy)
        # for
    return None