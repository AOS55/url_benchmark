import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
from mbrl.planning.trajectory_opt import MPPIOptimizer

import utils
from agent.ddpg import DDPGAgent


class DADS(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim):
        super().__init__()
        input_dim = obs_dim + skill_dim
        self.pred_net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, obs_dim))
        
        self.apply(utils.weight_init)

    def forward(self, obs, skill):
        input = torch.cat((obs, skill), 1)
        pred_net = self.pred_net(input)
        return pred_net


class DADSAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, dads_scale, update_encoder, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.dads_scale = dads_scale
        self.update_encoder = update_encoder
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim

        # create actor and critic
        super().__init__(**kwargs)

        self.obs_dim = 2
        # TODO: Check input network dimensions
        # create dads
        self.dads = DADS(self.obs_dim, self.skill_dim,
                         kwargs['hidden_dim']).to(kwargs['device'])

        # loss criterion
        self.dads_criterion = nn.KLDivLoss()
        # optimizers
        self.dads_opt = torch.optim.Adam(self.dads.parameters(), lr=self.lr)

        # MPPI Optimizer
        # self.planner = MPPIOptimizer(num_iterations=200,
        #                              population_size=40,
        #                              gamma=10.0,
        #                              sigma=1.0,
        #                              beta=0.9,
        #                              lower_bound=[],
        #                              upper_bound=[],
        #                              device=self.device)

        self.dads.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def update_dads(self, skill, obs, next_obs, step):
        metrics = dict()

        loss, df_accuracy = self.compute_dads_loss(obs, next_obs, skill)
        self.dads_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.dads_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        
        if self.use_tb or self.use_wandb:
            metrics['dads_loss'] = loss.item()
            metrics['dads_acc'] = df_accuracy

        return metrics

    def compute_intr_reward(self, skill, obs, next_obs, step):
        d_pred = self.dads(obs, skill)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        d_pred_softmax = F.softmax(d_pred, dim=1)
        pred_sum = torch.sum(d_pred_softmax, dim=0)
        reward = d_pred_log_softmax - torch.log(pred_sum) + np.log(len(d_pred))
        reward = torch.sum(reward, dim=1)
        reward = reward.reshape(-1, 1)
        return reward * self.dads_scale

    def compute_dads_loss(self, state, next_state, skill):
        d_pred = self.dads(state, skill)
        d_pred_log_softmax = F.log_softmax(d_pred)
        target = next_state - state  # target state change to zero mean
        d_loss = self.dads_criterion(d_pred_log_softmax, target)
        df_accuracy = torch.sum(d_pred - target)
        return d_loss, df_accuracy
                        
    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device
        )

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)
        
        if self.reward_free:
            # TODO: Look at urlb state obs vs dads obs
            metrics.update(self.update_dads(skill, obs[:, -3:-1], next_obs[:, -3:-1], step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, obs[:, -3:-1], next_obs[:, -3:-1], step)
            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        if self.reward_free:

            # extend observations with skill
            obs = torch.cat([obs, skill], dim=1)
            next_obs = torch.cat([next_obs, skill], dim=1)
            # update critic
            metrics.update(
                self.update_critic(obs.detach(), action, reward, discount, next_obs.detach(), step)
            )
            # update actor
            metrics.update(self.update_actor(obs.detach(), step))
            # update critic target
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        
        else:
            return None
            # action = self.planner.optimize(self., )
        return metrics
