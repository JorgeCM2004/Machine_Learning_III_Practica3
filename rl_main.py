import argparse
from pathlib import Path
from typing import Union

import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from models import ActorCritic, Reinforce


def train(
	env: gym.Env,
	model: Union[ActorCritic, Reinforce],
	args: argparse.Namespace,
	log_interval=10,
):
	Path("saved_models").mkdir(parents=True, exist_ok=True)

	try:
		for episode in range(1, args.episodes + 1):
			obs, _ = env.reset()
			done = False
			truncated = False

			episode_exp = []
			log_probs = []
			rewards = []

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

				rewards.append(reward)
				obs = next_obs

			actor_loss = 0.0
			critic_loss = 0.0

			if isinstance(model, Reinforce):
				actor_loss = model.update(log_probs, rewards)
			else:
				actor_loss, critic_loss = model.update(episode_exp)

			if episode % log_interval == 0:
				print(
					f"Episodio {episode} | "
					f"Reward: {sum(rewards):.2f} | "
					f"Actor Loss: {actor_loss:.4f} | "
					f"Critic Loss: {critic_loss:.4f}"
				)

			if episode % 100 == 0:
				save_path = f"saved_models/{args.agent}_{args.env}.pth"
				torch.save(model.actor.state_dict(), save_path)

	except KeyboardInterrupt:
		print("\nEntrenamiento interrumpido por el usuario.")
	finally:
		save_path = f"saved_models/{args.agent}_{args.env}.pth"
		torch.save(model.actor.state_dict(), save_path)


def evaluate(model, args, num_episodes=5):
	video_folder = f"videos/{args.agent}_{args.env}"
	Path(video_folder).mkdir(parents=True, exist_ok=True)

	eval_env = gym.make(args.env, render_mode="rgb_array")

	eval_env = RecordVideo(
		eval_env,
		video_folder=video_folder,
		episode_trigger=lambda x: True,
		name_prefix="eval",
		disable_logger=True,
	)

	model.actor.eval()

	for i in range(num_episodes):
		obs, _ = eval_env.reset()
		done = False
		truncated = False
		total_reward = 0

		while not (done or truncated):
			if isinstance(model, Reinforce):
				action, _ = model.act(obs, eval_mode=True)
			else:
				action = model.act(obs, eval_mode=True)

			obs, reward, done, truncated, _ = eval_env.step(action)
			total_reward += reward

		print(f"Evaluación Episodio {i + 1}: Retorno {total_reward:.2f}")

	eval_env.close()


def parse_args():
	parser = argparse.ArgumentParser(description="Entrenamiento RL.")
	parser.add_argument(
		"env",
		type=str,
		nargs="?",
		help="Nombre del entorno.",
		default="LunarLander-v3",
	)
	parser.add_argument(
		"agent",
		type=str,
		nargs="?",
		help="Algoritmo a usar.",
		choices=["reinforce", "actorcritic"],
		default="actorcritic",
	)
	parser.add_argument(
		"--episodes",
		type=int,
		default=500,
		help="Número total de episodios de entrenamiento",
	)

	return parser.parse_args()


def main():
	args = parse_args()

	try:
		env = gym.make(args.env, render_mode=None)
	except gym.error.Error as e:
		print(f"Error al crear el entorno '{args.env}': {e}")
		return

	if args.agent == "reinforce":
		model = Reinforce(env, lr=1e-3, gamma=0.99)

	elif args.agent == "actorcritic":
		model = ActorCritic(env, lr=1e-3, gamma=0.99)

	else:
		raise ValueError(f"Agente desconocido: {args.agent}")

	train(env, model, args)
	evaluate(model, args)

	env.close()


if __name__ == "__main__":
	main()
