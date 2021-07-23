import argparse
import logging
import os

import numpy as np
import stable_baselines3.common.results_plotter as plotter
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

import gym


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = plotter.ts2xy(plotter.load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                self.logger.record("mean_reward", mean_reward)
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
        return self.best_mean_reward < -4.5


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-i", "--trained_agent", help="Path to a pretrained agent to continue training", default="", type=str)
    p.add_argument("-m", type=int, default=3, choices=[3, 9], help="MxM beam shape")
    p.add_argument("-p", "--perturb", type=float, default=180, help="start phase perturb range")
    p.add_argument("-t", "--threshold", type=float, default=0.8, help="target norm. efficiency")
    p.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate")
    p.add_argument("--batch_size", type=int, default=256, help="batch size")
    args = p.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    log_dir = "runs/"
    # Create log dir
    os.makedirs(log_dir, exist_ok=True)
    config = {"M": args.m, "done_threshold": args.threshold, "perturb_range": args.perturb}
    env = gym.make("laser_cbc:mimocontrol-v0", **config)
    # Create and wrap the environment
    env = Monitor(env, log_dir)
    # Add some action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.001 * np.ones(n_actions))

    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"
    continue_training = args.trained_agent.endswith(".zip") and os.path.isfile(args.trained_agent)

    if continue_training:
        print("Load the saved agent...")
        model = DDPG.load(
            args.trained_agent, env=env,
            tensorboard_log=log_dir,
            action_noise=action_noise,
            verbose=args.verbose,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        print(model.policy)
    else:
        print("Training new agent...")
        # Because we use parameter noise, we should use a MlpPolicy with layer normalization
        model = DDPG(
            "MlpPolicy", env=env,
            create_eval_env=True,
            tensorboard_log=log_dir,
            action_noise=action_noise,
            verbose=args.verbose,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            # policy_kwargs=dict(net_arch=[400, 300])
        )
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # Train the agent
    timesteps = 4.5e5
    model.learn(total_timesteps=int(timesteps), callback=callback)

    # del model
    print("Load the saved agent...")
    model = DDPG.load("runs/best_model.zip")
    print(model.policy)

    print("Evaluate the agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    print("Testing agent...")
    for ix in range(10):
        done = False
        print("Testing agent...{}".format(ix))
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
        print("Done...steps: {}".format(env.steps))
        env.render()
