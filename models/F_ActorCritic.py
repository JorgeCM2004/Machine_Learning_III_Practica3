import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .F_Network import Network


class ActorCritic:
	def __init__(self, env, lr=1e-3, gamma=0.99):
		"""
		Initialize the ActorCritic model.

		Args:
			env (gym.Env): The environment to train on.
			lr (float): Learning rate for the optimizer.
			gamma (float): Discount factor for the future rewards.
		"""
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.gamma = gamma
		self.lr = lr

		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.n

		self.actor = Network(state_dim, action_dim).to(self.device)
		self.critic = Network(state_dim, action_dim).to(self.device)

		self.critic_target = copy.deepcopy(self.critic)
		self.critic_target.eval()

		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

		self.loss_fn = nn.MSELoss()

		self.target_update_freq = 10
		self.update_count = 0

	def act(self, state, eval_mode=False):
		"""
		Select an action based on the current state.

		Args:
			state (np.ndarray): The current state of the environment.
			eval_mode (bool): Whether to use evaluation mode.

		Returns:
			int: The selected action.
		"""
		state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
		logits = self.actor(state)

		if eval_mode:
			return torch.argmax(logits).item()

		dist = Categorical(logits=logits)
		action = dist.sample()
		return action.item()

	def update(self, episode_exp):
		"""
		Update the actor and critic networks based on the episode experience.

		Args:
			episode_exp (list): List of tuples containing the episode experience.

		Returns:
			tuple: Tuple containing the actor loss and critic loss.
		"""

		states, actions, rewards, next_states, dones = zip(*episode_exp)

		states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
		actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
		rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
		next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(
			self.device
		)
		dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

		loss_critic = self._update_critic(states, actions, next_states, rewards, dones)
		loss_actor = self._update_actor(states, actions, next_states, rewards, dones)

		self.update_count += 1
		if self.update_count % self.target_update_freq == 0:
			self.critic_target.load_state_dict(self.critic.state_dict())

		return loss_actor, loss_critic

	def _update_critic(self, state, actions, next_states, rewards, dones):
		"""
		Update the critic network based on the episode experience.

		Args:
			state (torch.Tensor): The current state of the environment.
			actions (torch.Tensor): The actions taken in the environment.
			next_states (torch.Tensor): The next states of the environment.
			rewards (torch.Tensor): The rewards received in the environment.
			dones (torch.Tensor): The done flags for the environment.

		Returns:
			float: The critic loss.
		"""
		q_values = self.critic(state)
		q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

		with torch.no_grad():
			next_q_values = self.critic_target(next_states)
			max_next_q = next_q_values.max(1)[0]
			target = rewards + self.gamma * max_next_q * (1 - dones)

		loss = self.loss_fn(q_value, target)

		self.critic_optimizer.zero_grad()
		loss.backward()
		self.critic_optimizer.step()

		return loss.item()

	def _update_actor(self, state, actions, next_states, rewards, dones):
		"""
		Update the actor network based on the episode experience.

		Args:
			state (torch.Tensor): The current state of the environment.
			actions (torch.Tensor): The actions taken in the environment.
			next_states (torch.Tensor): The next states of the environment.
			rewards (torch.Tensor): The rewards received in the environment.
			dones (torch.Tensor): The done flags for the environment.

		Returns:
			float: The actor loss.
		"""
		logits = self.actor(state)
		dist = Categorical(logits=logits)
		log_probs = dist.log_prob(actions)

		with torch.no_grad():
			q_values = self.critic(state)
			q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

			q_values_next = self.critic_target(next_states)
			q_next_max = q_values_next.max(1)[0]

			target_q = rewards + self.gamma * q_next_max * (1 - dones)
			advantage = target_q - q_current

		advantage = advantage.detach()

		actor_loss = -(log_probs * advantage).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		return actor_loss.item()
