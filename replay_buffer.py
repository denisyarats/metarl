import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        obs_dtype = np.uint8 if len(obs_shape) == 3 else np.float32
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.uint8)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], np.uint8(done))

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, discount):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        discounts = np.ones((idxs.shape[0], 1), dtype=np.float32) * discount
        discounts = torch.as_tensor(discounts, device=self.device)

        return obses, actions, rewards, next_obses, discounts
    
    def multi_sample(self, batch_size, T, discount):
        assert T < len(self)
        idxs = np.random.randint(0, len(self) - T, size=batch_size)
        
        obses, actions, rewards, next_obses, discounts = [], [], [], [], []
        for t in range(T):
            obses.append(self.obses[idxs + t])
            actions.append(self.actions[idxs + t])
            rewards.append(self.rewards[idxs + t])
            next_obses.append(self.next_obses[idxs + t])
            discounts.append(np.ones((idxs.shape[0], 1), dtype=np.float32) * discount)
            
        obses = torch.as_tensor(np.stack(obses), device=self.device).float()
        next_obses = torch.as_tensor(np.stack(next_obses), device=self.device).float()
        actions = torch.as_tensor(np.stack(actions), device=self.device)
        rewards = torch.as_tensor(np.stack(rewards), device=self.device)
        discounts = torch.as_tensor(np.stack(discounts), device=self.device)
        
        return obses, actions, rewards, next_obses, discounts

    
class MetaReplayBuffer(object):
    def __init__(self, num_tasks, obs_shape, action_shape, capacity, device):
        self.buffers = []
        for task_id in range(num_tasks):
            buffer = ReplayBuffer(obs_shape, action_shape, capacity, device)
            self.buffers.append(buffer)
            
    def num_tasks(self):
        return len(self.buffers)
            
    def add(self, task_id, obs, action, reward, next_obs, done):
        buffer = self.buffers[task_id]
        buffer.add(obs, action, reward, next_obs, done)
        
    def sample(self, task_id, batch_size, discount):
        buffer = self.buffers[task_id]
        return buffer.sample(batch_size, discount)
    
    def multi_sample(self, task_id, batch_size, T, discount):
        buffer = self.buffers[task_id]
        return buffer.multi_sample(batch_size, T, discount)