#!/usr/bin/env python3

import copy
import math
import os
import pickle as pkl
import sys
import time

import numpy as np

import dmc
import hydra
import torch
import utils
from logger import Logger
from replay_buffer import ReplayBuffer, MetaReplayBuffer
from video import VideoRecorder

torch.backends.cudnn.benchmark = True


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg


        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = dmc.make_meta(cfg.env, cfg.num_tasks, cfg.seed)
        self.eval_env = dmc.make_meta(cfg.env, cfg.num_tasks, cfg.seed + 1)

        obs_spec = self.env.observation_spec()['features']
        action_spec = self.env.action_spec()

        cfg.agent.params.obs_shape = obs_spec.shape
        cfg.agent.params.action_shape = action_spec.shape
        cfg.agent.params.action_range = [
            float(action_spec.minimum.min()),
            float(action_spec.maximum.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        num_buffers = 1 if cfg.agent.name == 'ddpg' else len(cfg.train_tasks)
        self.replay_buffer = MetaReplayBuffer(num_buffers,
                                          obs_spec.shape, action_spec.shape,
                                          cfg.replay_buffer_capacity,
                                          self.device)

        self.eval_video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_total_reward = 0
        for task_id in self.cfg.eval_tasks:
            # adaptation phase
            state = self.agent.reset() # reset agent once, so the memory persists acros episodes
            for episode in range(self.cfg.num_adapt_episodes):
                time_step = self.eval_env.reset(task_id)
                while not time_step.last():
                    with utils.eval_mode(self.agent):
                        obs = time_step.observation['features']
                        action = self.agent.act(obs, state, sample=False)
                    time_step = self.eval_env.step(action)
                    next_obs = time_step.observation['features']
                    # update agent's memory
                    state = self.agent.step(state, obs, action, time_step.reward, next_obs)
            
            # evaluation phase
            # agent's memory should be initialized by now
            average_episode_reward = 0
            for episode in range(self.cfg.num_eval_episodes):
                time_step = self.eval_env.reset(task_id)
                self.eval_video_recorder.init(enabled=(episode == 0))
                episode_reward = 0
                episode_success = 0
                episode_step = 0
                while not time_step.last():
                    with utils.eval_mode(self.agent):
                        obs = time_step.observation['features']
                        action = self.agent.act(obs, state, sample=False)
                    time_step = self.eval_env.step(action)
                    next_obs = time_step.observation['features']
                    # update agent's memory
                    state = self.agent.step(state, obs, action, time_step.reward, next_obs)
                    self.eval_video_recorder.record(self.eval_env)
                    episode_reward += time_step.reward
                    episode_step += 1

                average_episode_reward += episode_reward
                self.eval_video_recorder.save(f'task_{task_id}_step_{self.step}.mp4')
            average_episode_reward /= self.cfg.num_eval_episodes
            average_total_reward += average_episode_reward
            self.logger.log(f'eval/task_{task_id}_episode_reward', average_episode_reward,
                            self.step)
        average_total_reward /= self.eval_env.num_tasks()
        self.logger.log('eval/episode_reward', average_total_reward,
                            self.step)
        self.logger.dump(self.step, ty='eval')

    def run(self):
        episode, episode_reward, episode_step = 0, 0, 0
        start_time = time.time()
        done = True
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    fps = episode_step / (time.time() - start_time)
                    self.logger.log('train/fps', fps, self.step)
                    start_time = time.time()

                    self.logger.log('train/episode_reward', episode_reward,
                                    self.step)
                    self.logger.log('train/episode', episode, self.step)
                    self.logger.dump(
                        self.step,
                        save=(self.step > self.cfg.num_seed_steps),
                        ty='train')

                # initially try each task
                if episode < len(self.cfg.train_tasks):
                    task_id = self.cfg.train_tasks[episode]
                else:
                    task_id = np.random.choice(self.cfg.train_tasks)
                state = self.agent.reset()
                time_step = self.env.reset(task_id)
                obs = time_step.observation['features']
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # evaluate agent periodically
            if self.step % self.cfg.eval_frequency == 0:
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate()

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                spec = self.env.action_spec()
                action = np.random.uniform(spec.minimum, spec.maximum,
                                           spec.shape)
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, state, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            time_step = self.env.step(action)
            next_obs = time_step.observation['features']

            # allow infinite bootstrap
            done = time_step.last()
            episode_reward += time_step.reward

            buffer_id = 0 if self.cfg.agent.name == 'ddpg' else task_id
            self.replay_buffer.add(buffer_id, obs, action, time_step.reward, next_obs,
                                   done)
            # update agent's memory
            state = self.agent.step(state, obs, action, time_step.reward, next_obs)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    from train import Workspace as W
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
