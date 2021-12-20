import cherry as ch
import numpy as np
import torch
from cherry.algorithms import a2c
from torch import autograd
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm
import utils
import learn2learn as l2l


def compute_advantage(baseline, tau, gamma, rewards, dones, states, next_states):
    # Update baseline
    rewards = torch.reshape(rewards, [len(rewards), 1])
    dones = torch.reshape(dones, [len(dones), 1])
    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return ch.pg.generalized_advantage(tau=tau,
                                       gamma=gamma,
                                       rewards=rewards,
                                       dones=dones,
                                       values=bootstraps,
                                       next_value=next_value)


def maml_a2c_loss(time_step_list, meta_list, learner, baseline, gamma, tau, step):
    # print(utils.to_torch(batch, learner.device))

    states = torch.tensor(np.array([item['observation'] for item in time_step_list[:-1]]),
                          device=learner.device, dtype=torch.float64)
    next_states = torch.tensor(np.array([item['observation'] for item in time_step_list[1:]]),
                               device=learner.device, dtype=torch.float64)
    actions = torch.tensor(np.array([item['action'] for item in time_step_list[:-1]]),
                           device=learner.device, dtype=torch.float64)
    extr_reward = torch.tensor(np.array([item['reward'] for item in time_step_list[:-1]]),
                               device=learner.device, dtype=torch.float64)
    skills = torch.tensor(np.array([item['skill'] for item in meta_list[:-1]]),
                          device=learner.device, dtype=torch.float64)
    dones = torch.tensor(np.array([item['done'] for item in time_step_list[:-1]]),
                         device=learner.device, dtype=torch.float64)
    with torch.no_grad():
        next_states = learner.aug_and_encode(next_states)

    states = torch.cat([states, skills], dim=1)
    next_states = torch.cat([next_states, skills], dim=1)
    stddev = utils.schedule(learner.stddev_schedule, step)

    stddev = 1.0

    dist = learner.actor(states.detach().to(torch.float32), stddev)
    log_probs = dist.log_prob(actions).mean(dim=1, keepdim=True)
    advantages = compute_advantage(baseline, tau, gamma, extr_reward, dones, states, next_states)
    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)


def fast_adapt_a2c(clone, time_step_list, meta_list, adapt_lr, baseline, gamma, tau, step, first_order=False):
    second_order = not first_order
    loss = maml_a2c_loss(time_step_list, meta_list, clone, baseline, gamma, tau, step)
    gradients = autograd.grad(loss,
                              clone.actor.parameters(),
                              retain_graph=second_order,
                              create_graph=second_order)
    return l2l.algorithms.maml.maml_update(clone.actor, adapt_lr, gradients)
