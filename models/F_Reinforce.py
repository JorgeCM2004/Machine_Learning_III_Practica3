import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Reinforce:
	def __init__(self, env, lr=1e-3, gamma=0.99):
		self.gamma = gamma
		self.lr = lr
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.obs_dim = env.observation_space.shape[0]
		self.n_actions = env.action_space.n

		self.policy_net = nn.Sequential(
			nn.Linear(self.obs_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, self.n_actions),
		).to(self.device)

		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

	def act(self, state, eval_mode=False):
		state_t = torch.from_numpy(state).float().to(self.device)
		logits = self.policy_net(state_t)

		if eval_mode:
			return torch.argmax(logits).item(), None

		dist = Categorical(logits=logits)
		action = dist.sample()
		return action.item(), (dist.log_prob(action), dist.entropy())

	def update(self, trajectory_data, rewards):
		"""
		trajectory_data: Lista de tuplas (log_prob, entropy)
		rewards: Lista de recompensas
		"""
		log_probs = [item[0] for item in trajectory_data]
		entropies = [item[1] for item in trajectory_data]

		returns = []
		G = 0
		for r in reversed(rewards):
			G = r + self.gamma * G
			returns.insert(0, G)

		returns = torch.tensor(returns).to(self.device)

		if len(returns) > 1:
			returns = (returns - returns.mean()) / (returns.std() + 1e-8)

		policy_loss = 0
		entropy_loss = 0

		ent_coef = 0.01

		for log_prob, entropy, G_t in zip(log_probs, entropies, returns):
			policy_loss += -log_prob * G_t
			entropy_loss += -entropy

		total_loss = policy_loss + (ent_coef * entropy_loss)

		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		return total_loss.item()
