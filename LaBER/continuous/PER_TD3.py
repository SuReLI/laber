import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


    def act(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        return self.max_action * torch.tanh(a), a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q



class PER_TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=0.6,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=1e-3)

        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=1e-3)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0


    def select_action(self, state, test=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        # Sample uniformly for the actor
        state0, action0, next_state0, reward0, not_done0  = replay_buffer.sample0(batch_size)
        # Sample according to priorities for the critics
        state1, action1, next_state1, reward1, not_done1, ind1, weights1 = replay_buffer.sample1(batch_size)
        state2, action2, next_state2, reward2, not_done2, ind2, weights2 = replay_buffer.sample2(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action1) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action1 = (
                self.actor_target(next_state1) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            Q_target1_next1 = self.critic1_target(next_state1, next_action1)
            Q_target2_next1 = self.critic2_target(next_state1, next_action1)
            Q_target_next1 = torch.min(Q_target1_next1, Q_target2_next1)
            Q_target_next1 = reward1 + not_done1 * self.discount * Q_target_next1


        # Get current Q estimates
        current_Q1 = self.critic1(state1, action1)
        # Update priorities
        td_loss1 = (current_Q1 - Q_target_next1).abs()
        priority1 = td_loss1.pow(self.alpha).cpu().data.numpy().flatten()
        replay_buffer.update_priority1(ind1, priority1)
        
        critic1_loss = (weights1 * F.mse_loss(current_Q1, Q_target_next1, reduction='none')).mean()

        # Optimize the critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()


        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action2) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

            next_action2 = (
                    self.actor_target(next_state2) + noise
                    ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            Q_target1_next2 = self.critic1_target(next_state2, next_action2)
            Q_target2_next2 = self.critic2_target(next_state2, next_action2)
            Q_target_next2 = torch.min(Q_target1_next2, Q_target2_next2)
            Q_target_next2 = reward2 + not_done2 * self.discount * Q_target_next2


        # Get current Q estimates
        current_Q2 = self.critic2(state2, action2)
        # Update priorities
        td_loss2 = (current_Q2 - Q_target_next2).abs()
        priority2 = td_loss2.pow(self.alpha).cpu().data.numpy().flatten()
        replay_buffer.update_priority2(ind2, priority2)
        
        critic2_loss = (weights2 * F.mse_loss(current_Q2, Q_target_next2, reduction='none')).mean()

        # Optimize the critic
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()


        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            actor_loss = -self.critic1(state0, self.actor(state0)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

