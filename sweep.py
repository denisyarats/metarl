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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    overrides = Overrides()
    overrides.add('hydra/launcher', ['submitit'])
    overrides.add(key='experiment', values=[args.experiment])
    overrides.add(key='log_save_tb', values=['false'])
    # envs
    overrides.add(key='env', values=['cartpole_balance'])
    overrides.add(key='num_train_steps', values=[500000])
    overrides.add(key='eval_frequency', values=[10000])
    overrides.add(key='agent', values=['meta_ddpg'])
    
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
