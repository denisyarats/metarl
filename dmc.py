import numpy as np
from collections import OrderedDict, deque

import dm_env
from dm_env import specs
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels

from cartpoles import multi_task_cartpole


class FlattenObservationWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._obs_spec = OrderedDict()

        wrapped_obs_spec = env.observation_spec().copy()
        for key, spec in wrapped_obs_spec.items():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array
        dim = np.sum(
            np.fromiter((np.int(np.prod(spec.shape))
                         for spec in wrapped_obs_spec.values()), np.int32))

        self._obs_spec['features'] = specs.Array(shape=(dim,),
                                                 dtype=np.float32,
                                                 name='features')

        self._obs_spec['state'] = specs.Array(
            shape=self._env.physics.get_state().shape,
            dtype=np.float32,
            name='state')

    def _transform_observation(self, time_step):
        obs = OrderedDict()

        features = []
        for feature in time_step.observation.values():
            features.append(feature.ravel())
        obs['features'] = np.concatenate(features, axis=0)
        obs['state'] = self._env.physics.get_state().copy()
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    
    
class MetaEnv(dm_env.Environment):
    def __init__(self, envs):
        self._envs = envs
        self._task_id = 0
        
    def num_tasks(self):
        return len(self._envs)
    
    def _get_current_task(self):
        return self._envs[self._task_id]
    
    def get_current_task_id(self):
        return self._task_id
    
    def reset(self, task_id):
        assert 0 <= task_id < len(self._envs)
        self._task_id = task_id
        return self._get_current_task().reset()

    def step(self, action):
        return self._get_current_task().step(action)

    def observation_spec(self):
        return self._get_current_task().observation_spec()

    def action_spec(self):
        return self._get_current_task().action_spec()
    
    def render(self, *args, **kwargs):
        return self._get_current_task().physics.render(*args, **kwargs)
    
    def __getattr(self, name):
        env = self._get_current_task()
        return getattr(env, name)
    
    
    
class TaskIdWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._obs_spec = OrderedDict()
        self._num_tasks = self._env.num_tasks()
        
        wrapped_obs_spec = env.observation_spec().copy()
        self._obs_spec['state'] = wrapped_obs_spec['state']
        dim = wrapped_obs_spec['features'].shape[0]
        self._obs_spec['features'] = specs.Array(shape=(dim + self._num_tasks,),
                                                 dtype=np.float32,
                                                 name='features')

    def _transform_observation(self, time_step):
        task_features = np.zeros(self._num_tasks)
        task_features[self._env.get_current_task_id()] = 1.0
        obs = OrderedDict()
        obs['features'] = np.hstack([time_step.observation['features'], task_features])
        obs['state'] = time_step.observation['state'].copy()
        return time_step._replace(observation=obs)

    def reset(self, task_id):
        time_step = self._env.reset(task_id)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
        


def make(env_name, seed):

    if env_name == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = env_name.split('_')[0]
        task_name = '_'.join(env_name.split('_')[1:])

    env = suite.load(domain_name=domain_name,
                     task_name=task_name,
                     task_kwargs={'random': seed},
                     visualize_reward=False)

    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = FlattenObservationWrapper(env)


    action_spec = env.action_spec()
    assert np.all(action_spec.minimum >= -1.0)
    assert np.all(action_spec.maximum <= +1.0)

    return env


def make_meta(env_name, num_tasks, seed):
    assert num_tasks == 5
    assert env_name == 'cartpole_balance'
    envs = [
        multi_task_cartpole.balance_v1(random=seed),
        multi_task_cartpole.balance_v2(random=seed),
        multi_task_cartpole.balance_v3(random=seed),
        multi_task_cartpole.balance_v4(random=seed),
        multi_task_cartpole.balance_v5(random=seed),
    ]
    envs = [action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0) for env in envs]
    envs = [FlattenObservationWrapper(env) for env in envs]
        
    env = MetaEnv(envs)
    #env = TaskIdWrapper(env)
    return env
