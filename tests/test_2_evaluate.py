import os
import sys

import gymnasium as gym
import pytest
import torch
from gymnasium.wrappers import RecordVideo

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models import ActorCritic, Reinforce


def run_evaluation(env_name, agent_name, num_episodes=5):
	model_path = f"saved_models/{agent_name}_{env_name}.pth"
	if not os.path.exists(model_path):
		pytest.skip(
			f"Modelo no encontrado en {model_path}. Ejecuta primero el entrenamiento."
		)
		return

	video_folder = f"videos/{agent_name}_{env_name}"
	env = gym.make(env_name, render_mode="rgb_array")

	env = RecordVideo(
		env,
		video_folder=video_folder,
		episode_trigger=lambda x: True,
		name_prefix="eval",
		disable_logger=True,
	)

	if agent_name == "reinforce":
		model = Reinforce(env)
	elif agent_name == "actorcritic":
		model = ActorCritic(env)
	else:
		raise ValueError("Agente desconocido")

	try:
		model.actor.load_state_dict(torch.load(model_path))
		model.actor.eval()
	except Exception as e:
		pytest.fail(f"Error al cargar los pesos del modelo: {e}")
	total_rewards = []

	for i in range(num_episodes):
		obs, _ = env.reset()
		done = False
		truncated = False
		episode_reward = 0

		while not (done or truncated):
			if agent_name == "reinforce":
				action, _ = model.act(obs, eval_mode=True)
			else:
				action = model.act(obs, eval_mode=True)

			obs, reward, done, truncated, _ = env.step(action)
			episode_reward += reward

		total_rewards.append(episode_reward)
	with open(f"{video_folder}/total_rewards.txt", "w") as f:
		for reward in total_rewards:
			f.write(f"{reward}\n")
	env.close()


def test_evaluate_cartpole_reinforce():
	run_evaluation("CartPole-v1", "reinforce")


def test_evaluate_cartpole_actorcritic():
	run_evaluation("CartPole-v1", "actorcritic")


def test_evaluate_lunarlander_reinforce():
	run_evaluation("LunarLander-v3", "reinforce")


def test_evaluate_lunarlander_actorcritic():
	run_evaluation("LunarLander-v3", "actorcritic")
