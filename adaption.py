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
    # print(f'Length of dones: {len(dones)}')
    # dones = dones[-len(rewards):]
    # dones = torch.as_tensor(dones, device=rewards.device)
    # dones = dones.unsqueeze(1)
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


def maml_a2c_loss(replay_iter, learner, baseline, gamma, tau, meta, step):
    batch = next(replay_iter)
    # print(utils.to_torch(batch, learner.device))
    states, actions, extr_reward, discount, next_states, dones, skills = utils.to_torch(batch, learner.device)
    # print(f'states: {states}, length: {len(states)}')
    # print(f'actions: {actions}, length: {len(actions)}')
    # print(f'extr_reward: {extr_reward}, length: {len(extr_reward)}')
    # print(f'discount: {discount}, length: {len(discount)}')
    # print(f'next_states: {next_states}, length: {len(next_states)}')
    # print(f'dones: {dones}, length: {len(dones)}')
    # print(f'skills: {skills}, length: {len(skills)}')
    with torch.no_grad():
        next_states = learner.aug_and_encode(next_states)

    states = torch.cat([states, skills], dim=1)
    next_states = torch.cat([next_states, skills], dim=1)
    stddev = utils.schedule(learner.stddev_schedule, step)

    dist = learner.actor(states.detach(), stddev)
    log_probs = dist.log_prob(actions).mean(dim=1, keepdim=True)
    advantages = compute_advantage(baseline, tau, gamma, extr_reward, dones, states, next_states)
    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)


def fast_adapt_a2c(clone, replay_iter, adapt_lr, baseline, gamma, tau, step, meta, first_order=False):
    second_order = not first_order
    loss = maml_a2c_loss(replay_iter, clone, baseline, gamma, tau, meta, step)
    gradients = autograd.grad(loss,
                              clone.actor.parameters(),
                              retain_graph=second_order,
                              create_graph=second_order)
    return l2l.algorithms.maml.maml_update(clone.actor, adapt_lr, gradients)
