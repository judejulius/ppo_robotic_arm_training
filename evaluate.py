import tensorflow as tf
import numpy as np

import argparse
import json
from pathlib import Path

from train import RobotAgent, configure, evaluate, write_json, VALIDATION_SEEDS, FINAL_SEEDS


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved Tensor Optix robot arm checkpoint.')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--split', choices=['validation', 'final'], default='validation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new result path; existing evaluation results are preserved')
    configure()
    agent = RobotAgent()
    agent.load_weights(args.checkpoint)
    config_path = args.checkpoint.parent.parent/'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    seeds = config.get('validation_seeds', VALIDATION_SEEDS) if args.split == 'validation' else config.get('final_seeds', FINAL_SEEDS)
    result = evaluate(agent, seeds)
    result.update(split=args.split, steps=agent.steps, checkpoint=str(args.checkpoint.resolve()))
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
