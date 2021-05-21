import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
import autograd_hacks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        
        self.mu = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)


    def forward(self, state):
        x = self.l1(state)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)

        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1).unsqueeze(1)
        return action, log_prob


    def get_action(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample().to(device)
        action = torch.tanh(mu + e * std).cpu()
        return action[0]



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim+action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)





class GER_SAC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        discount=0.99,
        tau=0.005,
        policy_freq=1
    ):
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)

        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.target_entropy = -action_dim
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam(params=[self.log_alpha], lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state, test=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state)[0].cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        # Sample uniformly for the actor
        state0, action0, next_state0, reward0, not_done0  = replay_buffer.sample0(batch_size)
        # Sample according to priorities for the critics
        state1, action1, next_state1, reward1, not_done1, ind1, weights1 = replay_buffer.sample1(batch_size)
        state2, action2, next_state2, reward2, not_done2, ind2, weights2 = replay_buffer.sample2(batch_size)

        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action1, log_pis_next1 = self.actor.evaluate(next_state1)
            Q_target1_next1 = self.critic1_target(next_state1, next_action1)
            Q_target2_next1 = self.critic2_target(next_state1, next_action1)        
            # take the mean of both critics for updating
            Q_target_next1 = torch.min(Q_target1_next1, Q_target2_next1)
            # Compute Q targets for current states (y_i)
            Q_target_next1 = reward1 + not_done1 * self.discount * (Q_target_next1 - self.alpha * log_pis_next1)

        # Compute critic loss and per-sample gradient norms
        autograd_hacks.add_hooks(self.critic1)
        current_Q1 = self.critic1(state1, action1)
        critic1_loss = (weights1 * F.mse_loss(current_Q1, Q_target_next1, reduction="none")).mean()
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        autograd_hacks.compute_grad1(self.critic1)
        autograd_hacks.disable_hooks()
        autograd_hacks.clear_backprops(self.critic1)
        autograd_hacks.remove_hooks(self.critic1)
        
        grad_norms1 = np.zeros(batch_size)
        for layer in self.critic1.modules():
            if not autograd_hacks.is_supported(layer):
                continue
            for param in layer.parameters():
                for i in range(batch_size):
                    grad_norms1[i] += np.linalg.norm(param.grad1[i].numpy())**2
        grad_norms1 = np.sqrt(grad_norms1) / weights1.squeeze(1).numpy()

        replay_buffer.update_priority1(ind1, grad_norms1)
        self.critic1_optimizer.step()


        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action2, log_pis_next2 = self.actor.evaluate(next_state2)
            Q_target1_next2 = self.critic1_target(next_state2, next_action2)
            Q_target2_next2 = self.critic2_target(next_state2, next_action2)        
            # take the mean of both critics for updating
            Q_target_next2 = torch.min(Q_target1_next2, Q_target2_next2)
            # Compute Q targets for current states (y_i)
            Q_target_next2 = reward2 + not_done2 * self.discount * (Q_target_next2 - self.alpha * log_pis_next2)

        # Compute critic loss and per-sample gradient norms
        autograd_hacks.add_hooks(self.critic2)
        current_Q2 = self.critic2(state2, action2)
        critic2_loss = (weights2 * F.mse_loss(current_Q2, Q_target_next2, reduction="none")).mean()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward(retain_graph=True)
        autograd_hacks.compute_grad1(self.critic2)
        autograd_hacks.disable_hooks()
        autograd_hacks.clear_backprops(self.critic2)
        autograd_hacks.remove_hooks(self.critic2)

        grad_norms2 = np.zeros(batch_size)
        for layer in self.critic2.modules():
            if not autograd_hacks.is_supported(layer):
                continue
            for param in layer.parameters():
                for i in range(batch_size):
                    grad_norms2[i] += np.linalg.norm(param.grad1[i].numpy())**2
        grad_norms2 = np.sqrt(grad_norms2) / weights2.squeeze(1).numpy()

        replay_buffer.update_priority1(ind2, grad_norms2)
        self.critic2_optimizer.step()


        if self.total_it % self.policy_freq == 0:
            alpha = torch.exp(self.log_alpha)
            # Compute alpha loss
            actions_pred, log_pis = self.actor.evaluate(state0)
            alpha_loss = - (self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = alpha
            # Compute actor loss
            actor_loss = (alpha * log_pis.squeeze(0) - self.critic1(state0, actions_pred.squeeze(0))).mean()

            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

