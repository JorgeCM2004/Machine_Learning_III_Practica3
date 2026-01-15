import torch
import torch.optim as optim
from torch.distributions import Categorical

from .F_Network import Network


class Reinforce:
	def __init__(self, env, lr=1e-3, gamma=0.99):
		self.gamma = gamma
		self.lr = lr
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.obs_dim = env.observation_space.shape[0]
		self.n_actions = env.action_space.n

		self.actor = Network(self.obs_dim, self.n_actions).to(self.device)

		self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)

	def act(self, state, eval_mode=False):
		state_t = torch.from_numpy(state).float().to(self.device)
		logits = self.actor(state_t)

		if eval_mode:
			return torch.argmax(logits).item(), 0.0

		dist = Categorical(logits=logits)
		action = dist.sample()
		return action.item(), dist.log_prob(action)

	def update(self, log_probs, rewards):
		returns = []
		G = 0
		for r in reversed(rewards):
			G = r + self.gamma * G
			returns.insert(0, G)

		returns = torch.tensor(returns).to(self.device)
		if len(returns) > 1:
			returns = (returns - returns.mean()) / (returns.std() + 1e-8)

		policy_loss = 0
		for log_prob, G_t in zip(log_probs, returns):
			policy_loss += -log_prob * G_t

		self.optimizer.zero_grad()
		policy_loss.backward()
		self.optimizer.step()

		return policy_loss.item()
