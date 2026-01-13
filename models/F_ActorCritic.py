import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic:
	def __init__(self, env, lr=1e-3, gamma=0.99):
		self.gamma = gamma
		self.lr = lr
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"

		# 1. EL ACTOR (Política): Decide qué hacer
		self.obs_dim = env.observation_space.shape[0]
		self.n_actions = env.action_space.n

		self.actor_net = nn.Sequential(
			nn.Linear(self.obs_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, self.n_actions),
		).to(self.device)

		# 2. EL CRÍTICO (Valor): Juzga qué tan bueno es el estado
		self.critic_net = nn.Sequential(
			nn.Linear(self.obs_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 1),  # Salida escalar (Valor)
		).to(self.device)

		# Optimizamos ambas redes
		all_params = list(self.actor_net.parameters()) + list(
			self.critic_net.parameters()
		)
		self.optimizer = optim.Adam(all_params, lr=lr)

		# Referencia para el guardado del modelo
		self.policy_net = self.actor_net

	def act(self, state, eval_mode=False):
		"""
		Devuelve: acción, log_prob, valor
		"""
		state_t = torch.from_numpy(state).float().to(self.device)

		# 1. El Actor calcula los logits
		logits = self.actor_net(state_t)

		# 2. El Crítico estima el valor
		value = self.critic_net(state_t)

		if eval_mode:
			# Modo evaluación (Determinista)
			action = torch.argmax(logits).item()
			return action, 0.0, 0.0
		else:
			# Modo entrenamiento (Estocástico)
			dist = Categorical(logits=logits)
			action = dist.sample()
			log_prob = dist.log_prob(action)

			return action.item(), log_prob, value

	def update(self, log_probs, rewards, values):
		"""
		Calcula las pérdidas de Actor y Crítico y actualiza la red.
		"""
		# 1. Calcular Retornos Reales (G_t)
		returns = []
		G = 0
		for r in reversed(rewards):
			G = r + self.gamma * G
			returns.insert(0, G)

		# --- CORRECCIÓN CRÍTICA ---
		# Convertimos a tensor especificando float32 para evitar conflictos de tipo (Double vs Float)
		returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

		# Normalizamos retornos (Estabilidad numérica)
		if len(returns) > 1:
			returns = (returns - returns.mean()) / (returns.std() + 1e-9)

		# Procesamos los valores estimados por el crítico
		# 'values' es una lista de tensores, los concatenamos y ajustamos dimensiones
		values = torch.cat(values).squeeze()

		# 2. Calcular la VENTAJA (Advantage)
		# A(s,a) = G_t - V(s)
		# Usamos detach() porque el error de predicción del valor es solo para el crítico
		advantage = returns - values.detach()

		# 3. Cálculo de Pérdidas (Losses)

		# ACTOR LOSS: - log_prob * Ventaja
		actor_loss = 0
		for log_prob, adv in zip(log_probs, advantage):
			actor_loss += -log_prob * adv

		# CRITIC LOSS: MSE entre Valor Estimado y Retorno Real
		# Al haber convertido 'returns' a float32, esto ya es compatible
		critic_loss = F.mse_loss(values, returns)

		# Loss Total
		loss = actor_loss + critic_loss

		# 4. Optimización
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return actor_loss.item(), critic_loss.item()
