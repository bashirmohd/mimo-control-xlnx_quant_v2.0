import argparse
import logging
import os

from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import gym

# from stable_baselines3.common.env_utils import make_vec_env


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-i", "--trained_agent", help="Path to a pretrained agent to continue training",
                   default="runs/best_model.zip", type=str)
    p.add_argument("-m", type=int, default=3, choices=[3, 9], help="MxM beam shape")
    p.add_argument("-p", "--perturb", type=float, default=180, help="start phase perturb range [deg]")
    p.add_argument("-t", "--threshold", type=float, default=0.9, help="target norm. efficiency")
    args = p.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    config = {"M": args.m, "done_threshold": args.threshold, "perturb_range": args.perturb}
    env = gym.make("laser_cbc:mimocontrol-v0", **config)

    check_env(env)
    print("Env check passed.")

    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"
    print("Load the saved agent from {}".format(args.trained_agent))
    model = DDPG.load(args.trained_agent)
    print(model.policy)

    print("Evaluate the agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    for ix in range(10):
        done = False
        print("Testing agent...{}".format(ix))
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
        print("Done...steps: {}".format(env.steps))
        env.render()
