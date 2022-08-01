from curses import meta
from random import sample
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark=True

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        self.logger = Logger(self.work_dir, 
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)

        # create env
        self.train_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        # self.eval_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
        #                          cfg.action_repeat, cfg.seed)
        self.sample_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
    
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
    
    def sample(self):
        sample_until_step =  utils.Until(self.cfg.num_sample_episodes)
        step, episode, total_reward = 0, 0, 0
        meta = self.agent.init_meta()
        while sample_until_step(episode):
            meta = self.agent.init_meta()
            print(f"meta: {np.where(meta['skill']==1)[0][0]}")
            print(f"number of skills: {len(meta['skill'])}")
            time_step = self.sample_env.reset()
            self.video_recorder.init(self.sample_env, enabled=True)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.sample_env.step(action)
                self.video_recorder.record(self.sample_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            skill_index = np.where(meta['skill']==1)[0][0]
            self.video_recorder.save(f'{episode}_{skill_index}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
    
    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        domain, _ = self.cfg.task.split('_', 1)
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name

        def try_load(seed):
            snapshot = snapshot_dir / str(
                seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            print(f'snapshot exists: {snapshot.exists()}')
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
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

@hydra.main(config_path='.', config_name='sampling')
def main(cfg):
    from sampling import Workspace as W
    # root_dir = Path.cwd()
    workspace = W(cfg)
    # snapshot = root_dir / 'snapshot.pt'
    # print(f'snapshot is: {snapshot}')
    # if snapshot.exists():
    #     print(f'resuming: {snapshot}')
    #     workspace.load_snapshot()
    workspace.sample()

if __name__=='__main__':
    main()
