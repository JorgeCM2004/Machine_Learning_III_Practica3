import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Reinforce:
	def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.99):
		"""
		1. Constructor: Define e inicializa atributos y la red neuronal.
		"""
		self.gamma = gamma
		self.lr = lr
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.policy_net = nn.Sequential(
			nn.Linear(obs_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, n_actions),
		).to(self.device)

		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

	def act(self, state):
		"""
		2. Método act: Recibe un estado y selecciona una acción.
		"""
		state_t = torch.from_numpy(state).float().to(self.device)
		logits = self.policy_net(state_t)
		dist = Categorical(logits=logits)
		action = dist.sample()
		log_prob = dist.log_prob(action)

		return action.item(), log_prob

	def update(self, log_probs, rewards):
		"""
		3. Método update: Realiza la optimización de la red.

		Args:
		    log_probs: Lista de log_probs guardados durante el episodio.
		    rewards: Lista de recompensas obtenidas en cada paso del episodio.
		"""
		returns = []
		G = 0
		for r in reversed(rewards):
			G = r + self.gamma * G
			returns.insert(0, G)

		returns = torch.tensor(returns).to(self.device)
		returns = (returns - returns.mean()) / (returns.std() + 1e-9)

		loss = 0
		for log_prob, G_t in zip(log_probs, returns):
			loss += -log_prob * G_t

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()
