#!/usr/bin/env python3
import numpy as np
import numpy.random as npr

import time
import os
import sys
import argparse
from subprocess import Popen, DEVNULL
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_folder', type=str)
    args = parser.parse_args()

    slurm_pattern = os.path.join(args.sweep_folder, '.slurm', '*.pkl')
    total_canceled = 0
    for i, f in enumerate(glob.glob(slurm_pattern)):
        file_name = f.strip().split('/')[-1].split('.')[0]
        if file_name.endswith('_submitted'):
            idx = file_name[:-len('_submitted')]
            print(f'canceling {idx}')
            cmd = ['scancel', idx]
            env = os.environ.copy()
            p = Popen(cmd, env=env)
            p.communicate()
            total_canceled += 1

    print(f'total canceled {total_canceled}')


if __name__ == '__main__':
    main()
