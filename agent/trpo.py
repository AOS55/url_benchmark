import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils

from ddpg import Encoder, Actor
import cherry as ch


class TRPOAgent:
    def __init__(self, name, reward_free, obs_type, obs_shape, action_shape,
                 device, lr, feature_dim, hidden_dim, num_expl_steps, update_every_steps,
                 stddev_schedule, nstep, batch_size, stddev_clip,
                 adv_gamma, adv_tau,
                 use_tb, use_wandb, meta_dim=0):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.update_every_step = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.feature_dim = feature_dim
        self.solved_meta = None

        # advantage values
        self.adv_gamma = adv_gamma
        self.adv_tau = adv_tau

        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim

        self.policy = Actor(obs_type, self.obs_dim, self.action_dim,
                            feature_dim, hidden_dim).to(device)

        # optimizers
        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        else:
            self.encoder_opt = None
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.policy.train(training)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        try:
            utils.hard_update_params(other.actor, self.policy)
        except AttributeError:
            utils.hard_update_params(other.policy, self.policy)

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def get_advantage(self, rewards, dones, values, next_values):
        advantage = ch.pg.generalized_advantage(self.adv_gamma, self.adv_tau,
                                                rewards, dones, states, next_states)
        return advantage

    def act(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(input, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpe().numpy()[0]

    def update_policy(self, obs, step):
        metrics = dict()

        advantage = self.get_advantage(obs.rewards, obs.dones, obs.states, obs.next_states)


        if self.use_tb or self.use_wandb:
            metrics['policy_loss'] = policy_loss.item()
            metrics['policy_logprob'] = log_prob.mean().item()
            metrics['policy_ent'] = dist.entropy().sum(dim=-1).mean().item()
        self.metrics = None
        return self.metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_step != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update policy
        metrics.update(self.update_policy(obs.detach(), step))

        return metrics