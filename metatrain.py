import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
from copy import deepcopy

import hydra
import numpy as np
import torch
from dm_env import specs
from collections import namedtuple

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from cherry.algorithms import trpo
from cherry.models.robotics import LinearValue

from torch import autograd
from torch.nn.utils import parameters_to_vector, vector_to_parameters

torch.backends.cudnn.benchmark = True
from adaption import fast_adapt_a2c
from meta_trpo import meta_surrogate_loss


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # MAML specific hyperparameters
        self.adapt_lr = cfg.adapt_lr
        self.meta_lr = cfg.meta_lr
        self.gamma = cfg.gamma
        self.tau = cfg.tau

        # get a group of tasks based on domain
        self.available_tasks = []
        if cfg.domain == 'walker':
            self.available_tasks = ['walker_stand', 'walker_walk', 'walker_run', 'walker_flip']
        elif cfg.domain == 'quadruped':
            self.available_tasks = ['quadruped_walk', 'quadruped_run', 'quadruped_stand', 'quadruped_jump']
        elif cfg.domain == 'jaco':
            self.available_tasks = ['jaco_reach_top_left', 'jaco_reach_top_right', 'jaco_reach_bottom_left',
                                    'jaco_reach_bottom_right']
        else:
            print(f'Domain: {cfg.domain} not recognized!')

        self.logger_dict = {}
        self.train_envs = {}
        self.eval_envs = {}
        self.task_episodes = {}
        for task in self.available_tasks:
            train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)
            eval_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                cfg.action_repeat, cfg.seed)
            self.train_envs[task] = train_env
            self.eval_envs[task] = eval_env
            self.logger_dict[task] = Logger(self.work_dir,
                                            use_tb=cfg.use_tb,
                                            task=task,
                                            use_wandb=cfg.use_wandb)
            self.logger_outer = Logger(self.work_dir,
                                       use_tb=cfg.use_tb,
                                       task='meta',
                                       use_wandb=cfg.use_wandb)
            self.task_episodes[task] = 0

        print(self.train_envs.items())
        # ensure that the obs and action spec are the same for all tasks on the environment
        for train_env in self.train_envs.items():
            assert train_env[1].observation_spec() == self.train_envs[self.cfg.task].observation_spec()
            assert train_env[1].action_spec() == self.train_envs[self.cfg.task].action_spec()

        self.agent = make_agent(cfg.obs_type,
                                self.train_envs[self.cfg.task].observation_spec(),
                                self.train_envs[self.cfg.task].action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # initialize from a pretrained agent
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)

        # initialize baseline
        self.baseline = LinearValue(self.train_envs[self.cfg.task].observation_spec().shape[0],
                                    self.train_envs[self.cfg.task].action_spec().shape[0])

        # get meta specs
        self.meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        self.data_specs = (self.train_envs[self.cfg.task].observation_spec(),
                           self.train_envs[self.cfg.task].action_spec(),
                           specs.Array((1,), np.float32, 'reward'),
                           specs.Array((1,), np.float32, 'discount'),
                           specs.Array((1,), np.float32, 'done'))
        # create data storage
        self.replay_storage = ReplayBufferStorage(self.data_specs, self.meta_specs,
                                                  self.work_dir / 'buffer')
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self.task_replay_storage = None
        self.task_replay_loader = None
        self._task_replay_iter = None
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None
        )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.meta_steps = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def task_replay_iter(self):
        if self._task_replay_iter is None:
            self._task_replay_iter = iter(self.task_replay_loader)
        return self._task_replay_iter

    def adaption_train(self, task, train, policy=None):
        time_step_list = []  # list of steps in episode
        meta_list = []  # list of unsupervised meta values
        l2l_agent = None  # set to none in case the agent is not updated
        adapt_until_episode = utils.Until(self.cfg.num_adapt_episodes,
                                          self.cfg.action_repeat)  # how many adaption episodes to train on
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        self._task_replay_iter = None
        train_env = self.train_envs[task]
        eval_env = self.eval_envs[task]
        # setup env
        agent_clone = deepcopy(self.agent)  # policy to use for task rollouts
        video_clone = deepcopy(self.train_video_recorder)  # train_video_recorder clone
        episode_step, episode_reward, num_episodes = 0, 0, 0
        time_step = train_env.reset()
        meta = agent_clone.init_meta()
        time_step_list.append(time_step)
        meta_list.append(meta)
        video_clone.init(time_step.observation)
        metrics = None
        adapt_steps = 0
        task_step = 0

        if policy:
            agent_clone.actor = policy

        # complete adaption steps for each step
        while True:
            if time_step.done == 1.0:
                self._global_episode += 1
                self.task_episodes[task] += 1
                video_clone.save(f'{task + str(self.global_frame)}.mp4')
                # wait until all the metrics schema is populated
                # log stats
                elapsed_time, total_time = self.timer.reset()
                episode_frame = episode_step * self.cfg.action_repeat
                with self.logger_dict[task].log_and_dump_ctx(self.task_episodes[task], ty='train') as log:
                    log('fps', episode_frame / elapsed_time)
                    log('total_time', total_time)
                    log('episode_reward', episode_reward)
                    log('episode_length', episode_frame)
                    log('episode', self.global_episode)
                    log('step', self.global_step)
                if train:
                    l2l_agent = fast_adapt_a2c(agent_clone, time_step_list, meta_list, self.adapt_lr,
                                               self.baseline, self.gamma, self.tau, self.global_step)
                return time_step_list, meta_list, l2l_agent, agent_clone.aug_and_encode

            # try to evaluate, don't if seed frames needed
            if eval_every_step(task_step):
                self.logger_dict[task].log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval_task(eval_env, agent_clone, video_clone, task)

            meta = agent_clone.update_meta(meta, self.global_step, time_step)

            # get the action
            with torch.no_grad(), utils.eval_mode(agent_clone):
                action = agent_clone.act(time_step.observation,
                                         meta,
                                         self.global_step,
                                         eval_mode=False)
            # take env step
            time_step = train_env.step(action)
            episode_reward += time_step.reward
            time_step_list.append(time_step)
            meta_list.append(meta)
            video_clone.record(time_step.observation)
            episode_step += 1
            self._global_step += 1
            task_step += 1

    def eval_task(self, eval_env, agent, video_recorder, task):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = agent.init_meta()
        while eval_until_episode(episode):
            time_step = eval_env.reset()
            video_recorder.init(eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(time_step.observation,
                                       meta,
                                       self.global_step,
                                       eval_mode=True)
                time_step = eval_env.step(action)
                video_recorder.record(eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger_dict[task].log_and_dump_ctx(self.task_episodes[task], ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.task_episodes[task])
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)  # how long to train on all tasks

        while train_until_step(self.global_step):
            # Select a sample of tasks
            time_step_dict = {}
            meta_dict = {}
            time_step_valid_dict = {}
            meta_valid_dict = {}
            agent_dict = {}
            encode_dict = {}
            reward_list = []
            for task in self.available_tasks:
                print(f'Task is: {task}')
                time_step_list, meta_list, l2l_agent, aug_and_encode = self.adaption_train(task, True)
                time_step_dict[task] = time_step_list
                meta_dict[task] = meta_list
                agent_dict[task] = l2l_agent
                encode_dict[task] = aug_and_encode
                time_step_list, meta_list, _, _ = self.adaption_train(task, False, l2l_agent)
                time_step_valid_dict[task] = time_step_list
                meta_valid_dict[task] = meta_list
                # total_reward for logging
                reward_list.append(np.array(([item['reward'] for item in time_step_list[:-1]])).sum())
            # edit from here for meta_trpo
            # Update with MAML outer loop
            meta_policy = deepcopy(self.agent.actor)
            backtrack_factor = 0.5
            ls_max_steps = 40
            max_kl = 0.5
            self.logger_outer.log('train_adaption_reward', np.array(reward_list).mean(), self.meta_steps)

            old_loss, old_kl = meta_surrogate_loss(time_step_dict, meta_dict,
                                                   agent_dict, encode_dict,
                                                   time_step_valid_dict, meta_valid_dict, meta_policy,
                                                   self.baseline, self.tau, self.gamma, self.global_step,
                                                   self.device, self.agent.stddev_schedule)

            grad = autograd.grad(old_loss,
                                 meta_policy.parameters(),
                                 retain_graph=True)
            grad = parameters_to_vector([g.detach() for g in grad])
            Fvp = trpo.hessian_vector_product(old_kl, meta_policy.parameters())
            step = trpo.conjugate_gradient(Fvp, grad)
            shs = 0.5 * torch.dot(step, Fvp(step))
            lagrange_multiplier = torch.sqrt(shs / max_kl)
            step = step / lagrange_multiplier
            step_ = [torch.zeros_like(p.data) for p in meta_policy.parameters()]
            vector_to_parameters(step, step_)
            step = step_
            del old_kl, Fvp, grad
            old_loss.detach_()

            # Line Search
            for ls_step in range(ls_max_steps):
                stepsize = backtrack_factor ** ls_step * self.meta_lr
                clone = deepcopy(meta_policy)
                for p, u in zip(clone.parameters(), step):
                    p.data.add_(-stepsize, u.data)
                new_loss, kl = meta_surrogate_loss(time_step_dict, meta_dict,
                                                   agent_dict, encode_dict,
                                                   time_step_valid_dict, meta_valid_dict, clone,
                                                   self.baseline, self.tau, self.gamma, self.global_step,
                                                   self.device, self.agent.stddev_schedule)
                if new_loss < old_loss and kl < max_kl:
                    self.logger_outer.log('train_loss_new', new_loss, self.meta_steps)
                    print(f'new_loss: {new_loss} < old_loss: {old_loss} and kl: {kl} < {max_kl}')
                    for p, u in zip(meta_policy.parameters(), step):
                        p.data.add_(-stepsize, u.data)
                    break
            print(f'meta_steps: {self.meta_steps}, loss: {old_loss}')
            self.logger_outer.log('train_loss', old_loss, self.meta_steps)
            self.meta_steps += 1
            self.agent.actor = meta_policy
            print(torch.cuda.memory_summary(self.device))

    def load_snapshot(self):
        root_dir = os.path.dirname(os.path.realpath(__file__))
        snapshot_base_dir = Path(os.path.join(root_dir, self.cfg.snapshot_base_dir))
        domain, _ = self.cfg.task.split('_', 1)
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name

        def try_load(seed):
            snapshot = snapshot_dir / str(
                seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                payload = torch.load(f)
            return payload

        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try random seed
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
        return None


@hydra.main(config_path='.', config_name='metatrain')
def main(cfg):
    from metatrain import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
