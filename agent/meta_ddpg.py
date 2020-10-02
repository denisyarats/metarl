import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
import hydra
import kornia


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_shape, action_shape, state_dim, hidden_dim, hidden_depth,
                 stddev, parameterization):
        super().__init__()

        assert parameterization in ['clipped', 'squashed']
        self.stddev = stddev
        self.dist_type = utils.SquashedNormal if parameterization == 'squashed' else utils.ClippedNormal

        self.trunk = utils.mlp(obs_shape[0] + state_dim,
                               hidden_dim,
                               action_shape[0],
                               hidden_depth,
                               use_ln=True)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, state):
        z = torch.cat([obs, state], dim=-1)
        mu = self.trunk(z)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * self.stddev

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = self.dist_type(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_shape, action_shape, state_dim, hidden_dim, hidden_depth):
        super().__init__()
        
        self.Q1 = utils.mlp(obs_shape[0] + state_dim + action_shape[0],
                            hidden_dim,
                            1,
                            hidden_depth,
                            use_ln=True)
        self.Q2 = utils.mlp(obs_shape[0] + state_dim + action_shape[0],
                            hidden_dim,
                            1,
                            hidden_depth,
                            use_ln=True)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, state):
        assert obs.size(0) == action.size(0)

        z = torch.cat([obs, action, state], dim=-1)
        q1 = self.Q1(z)
        q2 = self.Q2(z)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)
                
                
class StateModel(nn.Module):
    def __init__(self, obs_shape, action_shape, state_dim):
        super().__init__()
        
        self.state_dim = state_dim
        self.rnn = nn.GRUCell(2 * obs_shape[0] + action_shape[0] + 1, state_dim)
        
    def forward(self, state, obs, action, reward, next_obs):
        x = torch.cat([obs, action, reward, next_obs], dim=1)
        return self.rnn(x, state)


class MetaDDPGAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, obs_shape, action_shape, action_range, device,
                 critic_cfg, actor_cfg, state_model_cfg, discount, lr,
                 actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size, multi_step):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.multi_step = multi_step

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.state_model = hydra.utils.instantiate(state_model_cfg).to(self.device)


        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr)
        self.state_model_optimizer = torch.optim.Adam(self.state_model.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.state_model.train(training)
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, state, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs, state)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])
    
    def reset(self):
        return torch.zeros(1, self.state_model.state_dim).to(self.device)
        
    def step(self, state, obs, action, reward, next_obs):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        action = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        reward = torch.FloatTensor([reward]).to(self.device).unsqueeze(0)
        next_obs = torch.FloatTensor(next_obs).to(self.device).unsqueeze(0)
        return self.state_model.forward(state, obs, action, reward, next_obs)

    def update_critic2(self, obses, actions, rewards, next_obses, discounts,
                      logger, step):
        
        #import ipdb; ipdb.set_trace()
        state = torch.zeros((obses.shape[1], self.state_model.state_dim)).to(self.device)
        T = obses.shape[0]
        critic_loss = 0
        for t in range(T):
            obs, action, reward, next_obs, discount = obses[t], actions[t], rewards[t], next_obses[t], discounts[t]
        
            with torch.no_grad():
                next_state = self.state_model(state, obs, action, reward, next_obs)
                dist = self.actor(next_obs, next_state)
                next_action = dist.rsample()
                target_Q1, target_Q2 = self.critic_target(
                    next_obs, next_action, next_state)
                target_V = torch.min(target_Q1, target_Q2)
                target_Q = reward + (discount * target_V)

            Q1, Q2 = self.critic(obs, action, state)
            critic_loss += F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
            state = self.state_model(state, obs, action, reward, next_obs)

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        self.state_model_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.state_model_optimizer.step()

        self.critic.log(logger, step)
        
    def update_critic(self, obses, actions, rewards, next_obses, discounts,
                      logger, step):
        
        #import ipdb; ipdb.set_trace()
        states = [torch.zeros((obses.shape[1], self.state_model.state_dim)).to(self.device)]
        T = obses.shape[0]
        for t in range(T):
            obs, action, reward, next_obs = obses[t], actions[t], rewards[t], next_obses[t]
            state = self.state_model(states[-1], obs, action, reward, next_obs)
            states.append(state)
            
        with torch.no_grad():
            dist = self.actor(next_obses[-1], states[-1])
            next_action = dist.rsample()
            target_Q1, target_Q2 = self.critic_target(
                next_obses[-1], next_action, states[-1])
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = rewards[-1] + (discounts[-1] * target_V)
            
        Q1, Q2 = self.critic(obses[-1], actions[-1], states[-2])
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        self.state_model_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.state_model_optimizer.step()

        self.critic.log(logger, step)

    def update_actor2(self, obses, actions, rewards, next_obses, logger, step):
        #import ipdb; ipdb.set_trace()
        state = torch.zeros((obses.shape[1], self.state_model.state_dim)).to(self.device)
        T = obses.shape[0]
        
        actor_loss = 0
        for t in range(T):
            obs, action, reward, next_obs = obses[t], actions[t], rewards[t], next_obses[t]
            dist = self.actor(obs, state)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            Q1, Q2 = self.critic(obs, action, state)
            Q = torch.min(Q1, Q2)

            actor_loss += -Q.mean()
            
            state = self.state_model(state, obs, action, reward, next_obs)

        logger.log('train_actor/loss', actor_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        self.state_model_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.state_model_optimizer.step()

        self.actor.log(logger, step)
        
    def update_actor(self, obses, actions, rewards, next_obses, logger, step):
        #import ipdb; ipdb.set_trace()
        state = torch.zeros((obses.shape[1], self.state_model.state_dim)).to(self.device)
        T = obses.shape[0]
        for t in range(T - 1):
            obs, action, reward, next_obs = obses[t], actions[t], rewards[t], next_obses[t]
            state = self.state_model(state, obs, action, reward, next_obs)
            
        dist = self.actor(obses[-1], state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obses[-1], action, state)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()
        logger.log('train_actor/loss', actor_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        self.state_model_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.state_model_optimizer.step()

        self.actor.log(logger, step)

    def update(self, replay_buffer, logger, step):
        obses, actions, rewards, next_obses, discounts = replay_buffer.multi_sample(
            self.batch_size, self.multi_step, self.discount)

        logger.log(f'train/batch_reward', rewards.mean(), step)

        self.update_critic(obses, actions, rewards, next_obses, discounts, logger,
                           step)

        if step % self.actor_update_frequency == 0:
            self.update_actor(obses, actions, rewards, next_obses, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
