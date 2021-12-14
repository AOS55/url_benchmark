import learn2learn as l2l
from copy import deepcopy
import utils
import torch
from torch.distributions.kl import kl_divergence
from adaption import compute_advantage, fast_adapt_a2c
import cherry as ch
from cherry.algorithms import trpo


def replay_list(replay_loader):
    data_list = []
    for data in replay_loader:
        print(len(data))
        data_list.append(data)
    return data_list


def meta_surrogate_loss(replay_dict, agent_dict, encode_dict, replay_valid_dict,
                        policy, baseline, tau, gamma, step, device):
    mean_loss = 0.0
    mean_kl = 0.0
    for task in agent_dict:
        task_replay = replay_dict[task]
        task_replay_valid = replay_valid_dict[task]
        task_agent = agent_dict[task]
        aug_and_encode = encode_dict[task]
        new_policy = deepcopy(policy)

        print(f'task_replay.dataset: {task_replay.dataset}')
        print(f'task_replay_valid.dataset: {task_replay_valid.dataset}')
        task_replay_list = replay_list(task_replay)
        print(task_replay_list)
        batch = next(iter(task_replay.dataset))
        states, actions, extr_reward, discount, next_states, dones, skills = utils.to_torch(batch, device)
        with torch.no_grad():
            next_states = aug_and_encode(next_states)

        valid_batch = next(iter(task_replay_valid))
        valid_states, valid_actions, valid_extr_reward, valid_discount, valid_next_states, valid_dones, valid_skills =\
            utils.to_torch(valid_batch, device)
        with torch.no_grad():
            valid_next_states = aug_and_encode(valid_next_states)

        states = torch.cat([states, skills], dim=1)
        next_states = torch.cat([next_states, skills], dim=1)

        valid_states = torch.cat([valid_states, valid_skills], dim=1)
        valid_next_states = torch.cat([valid_next_states, valid_next_states], dim=1)

        stddev = utils.schedule(new_policy.stddev_schedule, step)  # TODO: Needs to match an appropriate schedule

        new_dist = new_policy(valid_states.detach(), stddev)
        old_dist = task_agent(states.detach(), stddev)
        kl = kl_divergence(new_dist, old_dist)
        mean_kl += kl

        # Compute surrogate loss
        advantages = compute_advantage(baseline, tau, gamma, valid_dones, valid_extr_reward,
                                       valid_states, valid_next_states)
        advantages = ch.normalize(advantages).detach()
        old_log_probs = old_dist.log_prob(valid_actions).mean(dim=1, keepdim=True).detach()
        new_log_probs = new_dist.log_prob(valid_actions).mean(dim=1, keepdim=True)
        mean_loss += trpo.policy_loss(new_log_probs, old_log_probs, advantages)
    mean_kl /= len(replay_dict)
    mean_loss /= len(replay_dict)
    return mean_loss, mean_kl
