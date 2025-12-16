from argparse import ArgumentParser


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


if __name__ == "__main__":
	args = parse_args()
	print(f"Entorno seleccionado: {args.env}")
	print(f"Agente seleccionado: {args.agent}")
