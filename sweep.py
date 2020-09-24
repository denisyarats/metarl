#!/usr/bin/env python3

import numpy as np
import numpy.random as npr

import time
import os
import argparse

from subprocess import Popen, DEVNULL


class Overrides(object):
    def __init__(self):
        self.kvs = dict()

    def add(self, key, values):
        processed_values = []
        for v in values:
            if type(v) == str:
                processed_values.append(v)
            else:
                processed_values.append(str(v))
        value = ','.join(processed_values)
        assert key not in self.kvs
        self.kvs[key] = value

    def cmd(self):
        cmd = []
        for k, v in self.kvs.items():
            cmd.append(f'{k}={v}')
        return cmd


ENVS_SMALL = [
    'cheetah_run', 'walker_walk', 'cartpole_swingup', 'reacher_easy',
    'ball_in_cup_catch', 'finger_spin'
]

ENVS_HARD = [
    'acrobot_swingup',
    #'fish_swim',
    'hopper_hop',
    'humanoid_stand',
    'humanoid_walk',
    'humanoid_run',
    'quadruped_run',
    #'swimmer_swimmer6',
    #'swimmer_swimmer15'
]

ENVS = [
    'cartpole_swingup_sparse', 'finger_spin', 'walker_run', 'fish_upright',
    'humanoid_run', 'fish_swim', 'quadruped_walk', 'cheetah_run',
    'quadruped_run', 'humanoid_walk', 'walker_stand', 'hopper_hop',
    'reacher_hard', 'finger_turn_easy', 'walker_walk', 'reacher_easy',
    'finger_turn_hard', 'hopper_stand', 'humanoid_stand', 'pendulum_swingup',
    'cartpole_balance_sparse', 'cartpole_balance', 'ball_in_cup_catch',
    'cartpole_swingup'
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    overrides = Overrides()
    overrides.add('hydra/launcher', ['submitit'])
    overrides.add(key='experiment', values=[args.experiment])
    overrides.add(key='log_save_tb', values=['false'])
    overrides.add(key='save_video', values=['false'])
    # envs
    overrides.add(key='env', values=ENVS_SMALL)
    overrides.add(key='num_train_steps', values=[1000000])
    overrides.add(key='eval_frequency', values=[50000])
    overrides.add(key='agent', values=['ddpg'])
    overrides.add(key='agent.params.actor_update_frequency', values=[1])
    overrides.add(key='agent.params.critic_target_update_frequency',
                  values=[1])
    overrides.add(key='obs_type', values=['features'])
    overrides.add(key='agent.params.use_ln', values=[True])
    #overrides.add(key='lr', values=[1e-3, 5e-4, 1e-4])
    overrides.add(key='hidden_depth', values=[2, 3, 4, 5, 7])
    #overrides.add(key='parameterization', values=['squashed'])
    #overrides.add(key='parameterization', values=['clipped'])
    # seeds
    overrides.add(key='seed', values=list(range(1, 6)))

    cmd = ['python', 'train.py', '-m']
    cmd += overrides.cmd()

    if args.dry:
        print(cmd)
    else:
        env = os.environ.copy()
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == '__main__':
    main()
