import copy
import numpy as np
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




class LABER_TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
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

        self.total_it = 0


    def select_action(self, state, test=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=100, m_factor=4):
        self.total_it += 1

        # Sample uniformly a large batch from the replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(int(m_factor*batch_size))

        # Select with no_grad the mini-batch from the large batch
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            Q_target1_next = self.critic1_target(next_state, next_action)
            Q_target2_next = self.critic2_target(next_state, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)
            Q_target_next_large = reward + not_done * self.discount * Q_target_next

            # Select uniformly a subset for the actor
            indices_actor = np.random.randint(int(m_factor*batch_size), size=batch_size)
            states_selected_for_actor = state[indices_actor]

            ## Select a subset of transitions for the first critic
            # Compute surrogate distribution
            Q_1_large = self.critic1(state, action)
            td_errors1 = (Q_1_large - Q_target_next_large).abs().detach().cpu().data.numpy().flatten()
            probs1 = td_errors1/td_errors1.sum()
            # Select the samples
            indices1 = np.random.choice(int(m_factor*batch_size), batch_size, p=probs1)
            td_errors_for_selected_indices1 = td_errors1[indices1]
            states_selected1 = state[indices1]
            actions_selected1 = action[indices1]
            Q_targets1 = Q_target_next_large[indices1]
            # Compute the weights for SGD update
            # LaBER:
            loss_weights1 = (1.0 / td_errors_for_selected_indices1) * td_errors1.mean()
            # LaBER-lazy: loss_weights1 = (1.0 / td_errors_for_selected_indices1)
            loss_weights1 = torch.from_numpy(loss_weights1).unsqueeze(1)

            Q_2_large = self.critic2(state, action)
            td_errors2 = (Q_2_large - Q_target_next_large).abs().detach().cpu().data.numpy().flatten()
            probs2 = td_errors2/td_errors2.sum()
            # Select the samples
            indices2 = np.random.choice(int(m_factor*batch_size), batch_size, p=probs2)
            td_errors_for_selected_indices2 = td_errors2[indices2]
            states_selected2 = state[indices2]
            actions_selected2 = action[indices2]
            Q_targets2 = Q_target_next_large[indices2]
            # Compute the weights for SGD update
            # LaBER:
            loss_weights2 = (1.0 / td_errors_for_selected_indices2) * td_errors2.mean()
            # LaBER-lazy: loss_weights2 = (1.0 / td_errors_for_selected_indices2)
            loss_weights2 = torch.from_numpy(loss_weights2).unsqueeze(1)


        Q_1 = self.critic1(states_selected1, actions_selected1)
        critic1_loss = (0.5 * F.mse_loss(Q_1, Q_targets1, reduction="none").cpu() * loss_weights1).mean()
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        Q_2 = self.critic2(states_selected2, actions_selected2)
        critic2_loss = (0.5 * F.mse_loss(Q_2, Q_targets2, reduction="none").cpu() * loss_weights2).mean()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()



        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic1(states_selected_for_actor, self.actor(states_selected_for_actor)).mean()

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

