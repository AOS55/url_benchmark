import learn2learn as l2l
from copy import deepcopy
import utils
import torch
from torch.distributions.kl import kl_divergence
from adaption import compute_advantage
import cherry as ch
import numpy as np
from cherry.algorithms import trpo


def unpack_replay_lists(time_step_list, meta_list, device):
    states = torch.tensor(np.array([item['observation'] for item in time_step_list[:-1]]),
                          device=device, dtype=torch.float64)
    next_states = torch.tensor(np.array([item['observation'] for item in time_step_list[1:]]),
                               device=device, dtype=torch.float64)
    actions = torch.tensor(np.array([item['action'] for item in time_step_list[:-1]]),
                           device=device, dtype=torch.float64)
    extr_reward = torch.tensor(np.array([item['reward'] for item in time_step_list[:-1]]),
                               device=device, dtype=torch.float64)
    skills = torch.tensor(np.array([item['skill'] for item in meta_list[:-1]]),
                          device=device, dtype=torch.float64)
    dones = torch.tensor(np.array([item['done'] for item in time_step_list[:-1]]),
                         device=device, dtype=torch.float64)
    return states, next_states, actions, extr_reward, skills, dones


def meta_surrogate_loss(time_step_dict, meta_dict, agent_dict, encode_dict, time_step_vaid_dict, meta_valid_dict,
                        policy, baseline, tau, gamma, step, device, schedule):
    mean_loss = 0.0
    mean_kl = 0.0
    for task in agent_dict:
        task_time_step_list = time_step_dict[task]
        task_meta_list = meta_dict[task]
        task_time_step_valid_list = time_step_vaid_dict[task]
        task_meta_valid_list = meta_valid_dict[task]
        task_agent = agent_dict[task]
        aug_and_encode = encode_dict[task]
        new_policy = l2l.clone_module(policy)

        states, next_states, actions, extr_reward, skills, dones = \
            unpack_replay_lists(task_time_step_list, task_meta_list, device)
        with torch.no_grad():
            next_states = aug_and_encode(next_states)

        valid_states, valid_next_states, valid_actions, valid_extr_reward, valid_skills, valid_dones = \
            unpack_replay_lists(task_time_step_valid_list, task_meta_valid_list, device)
        with torch.no_grad():
            valid_next_states = aug_and_encode(valid_next_states)

        states = torch.cat([states, skills], dim=1)
        next_states = torch.cat([next_states, skills], dim=1)

        valid_states = torch.cat([valid_states, valid_skills], dim=1)
        valid_next_states = torch.cat([valid_next_states, valid_skills], dim=1)

        stddev = utils.schedule(schedule, step)  # TODO: Needs to match an appropriate schedule

        stddev = 1.0

        new_dist = new_policy(valid_states.detach().to(torch.float32), stddev)
        old_dist = task_agent(states.detach().to(torch.float32), stddev)
        kl = kl_divergence(new_dist, old_dist).mean()
        mean_kl += kl

        # Compute surrogate loss
        advantages = compute_advantage(baseline, tau, gamma, valid_extr_reward, valid_dones,
                                       valid_states, valid_next_states)
        advantages = ch.normalize(advantages).detach()
        old_log_probs = old_dist.log_prob(valid_actions).mean(dim=1, keepdim=True).detach()
        new_log_probs = new_dist.log_prob(valid_actions).mean(dim=1, keepdim=True)
        mean_loss += trpo.policy_loss(new_log_probs, old_log_probs, advantages)
    mean_kl /= len(agent_dict)
    mean_loss /= len(agent_dict)
    return mean_loss, mean_kl
