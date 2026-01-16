import os
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models import ActorCritic, Reinforce

EPISODES = 1000
RESULTS_DIR = "results_plots"
MODELS_DIR = "saved_models"

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)


def moving_average(data, window_size=50):
	return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def train_agent(env, model, episodes, agent_name, env_name):
	history_rewards = []

	print(f"\nEntrenando {model.__class__.__name__} en {env.spec.id}...")

	for episode in range(1, episodes + 1):
		obs, _ = env.reset()
		done = False
		truncated = False

		episode_rewards = []
		log_probs = []
		episode_exp = []

		while not (done or truncated):
			if isinstance(model, Reinforce):
				action, log_prob = model.act(obs, eval_mode=False)
				log_probs.append(log_prob)
			else:
				action = model.act(obs, eval_mode=False)

			next_obs, reward, done, truncated, _ = env.step(action)

			if isinstance(model, ActorCritic):
				mask = 1 if (done or truncated) else 0
				episode_exp.append((obs, action, reward, next_obs, mask))

			episode_rewards.append(reward)
			obs = next_obs

		if isinstance(model, Reinforce):
			model.update(log_probs, episode_rewards)
		else:
			model.update(episode_exp)

		total_reward = sum(episode_rewards)
		history_rewards.append(total_reward)

		if episode % 100 == 0:
			print(f"Episodio {episode}/{episodes} | Retorno: {total_reward:.2f}")

	save_path = f"{MODELS_DIR}/{agent_name}_{env_name}.pth"
	torch.save(model.actor.state_dict(), save_path)
	print(f"Modelo guardado en: {save_path}")

	return history_rewards


def run_comparison(env_name, episodes):
	env = gym.make(env_name)

	agent_reinforce = Reinforce(env, lr=1e-3, gamma=0.99)
	rewards_reinforce = train_agent(
		env, agent_reinforce, episodes, "reinforce", env_name
	)

	agent_ac = ActorCritic(env, lr=1e-3, gamma=0.99)
	rewards_ac = train_agent(env, agent_ac, episodes, "actorcritic", env_name)

	env.close()

	plt.figure(figsize=(10, 6))

	plt.plot(rewards_reinforce, alpha=0.3, color="blue", label="Reinforce (Raw)")
	plt.plot(rewards_ac, alpha=0.3, color="orange", label="ActorCritic (Raw)")

	ma_reinforce = moving_average(rewards_reinforce)
	ma_ac = moving_average(rewards_ac)

	plt.plot(ma_reinforce, color="blue", linewidth=2, label="Reinforce (MA)")
	plt.plot(ma_ac, color="orange", linewidth=2, label="ActorCritic (MA)")

	plt.title(f"Comparativa de Aprendizaje: {env_name}")
	plt.xlabel("Episodios")
	plt.ylabel("Retorno Acumulado")
	plt.legend()
	plt.grid(True, alpha=0.3)

	filename = f"{RESULTS_DIR}/comparativa_{env_name}.png"
	plt.savefig(filename)
	print(f"Gráfica guardada en: {filename}")
	plt.close()


def test_compare_cartpole():
	run_comparison("CartPole-v1", episodes=EPISODES)


def test_compare_lunarlander():
	run_comparison("LunarLander-v3", episodes=EPISODES)
