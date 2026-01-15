from torch import nn


class Network(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=128):
		super(Network, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, action_dim),
		)

	def forward(self, x):
		return self.net(x)
