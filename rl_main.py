from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from torch import save

from models import ActorCritic, Reinforce


def train(
	env: gym.Env,
	model: Union[ActorCritic, Reinforce],
	episodes=500,
	log_interval=10,
):
	global args
	try:
		for episode in range(1, episodes + 1):
			obs, _ = env.reset()
			done = False
			truncated = False

			log_probs = []
			rewards = []
			values = []

			while not (done or truncated):
				if isinstance(model, Reinforce):
					action, log_prob = model.act(obs, eval_mode=False)
				else:
					action, log_prob, value = model.act(obs, eval_mode=False)
					values.append(value)

				obs, reward, done, truncated, _ = env.step(action)

				log_probs.append(log_prob)
				rewards.append(reward)

			actor_loss = 0
			critic_loss = 0

			if isinstance(model, Reinforce):
				actor_loss = model.update(log_probs, rewards)
			else:
				actor_loss, critic_loss = model.update(log_probs, rewards, values)

			if episode % log_interval == 0:
				print(
					f"Episodio {episode} | Retorno: {sum(rewards):.2f} | "
					f"Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}"
				)
	except KeyboardInterrupt:
		print("Entrenamiento interrumpido por el usuario.")
	finally:
		Path("saved_models").mkdir(exist_ok=True)
		save_path = f"saved_models/{args.agent}_{args.env}.pth"
		save(model.policy_net.state_dict(), save_path)


def evaluate(model, num_episodes=5):
	video_folder = f"videos/{args.agent}_{args.env}"

	env = gym.make(args.env, render_mode="rgb_array")
	env = RecordVideo(
		env,
		video_folder=video_folder,
		episode_trigger=lambda x: True,
		name_prefix="eval",
	)

	for i in range(num_episodes):
		obs, _ = env.reset()
		done = False
		truncated = False
		total_reward = 0

		while not (done or truncated):
			if isinstance(model, Reinforce):
				action, _ = model.act(obs, eval_mode=True)
			else:
				action, _, _ = model.act(obs, eval_mode=True)

			obs, reward, done, truncated, _ = env.step(action)
			total_reward += reward

		print(f"Evaluación Episodio {i + 1}: Retorno {total_reward:.2f}")

	env.close()


def parse_args():
	parser = ArgumentParser()
	parser.add_argument(
		"env",
		type=str,
		nargs="?",
		help="Selecciona el entorno.",
		default="LunarLander-v3",
	)
	parser.add_argument(
		"agent",
		type=str,
		nargs="?",
		help="Selecciona el agente.",
		choices=["reinforce", "actorcritic"],
		default="reinforce",
	)

	return parser.parse_args()


def main():
	global args
	args = parse_args()
	try:
		env = gym.make(args.env, render_mode=None)
	except gym.error.UnregisteredEnv:
		raise ValueError(f"Entorno no reconocido: {args.env}")
	match args.agent:
		case "reinforce":
			model = Reinforce(env)
		case "actorcritic":
			model = ActorCritic(env)
		case _:
			raise ValueError(f"Agente no reconocido: {args.agent}")

	train(env, model)
	evaluate(model)


if __name__ == "__main__":
	main()
