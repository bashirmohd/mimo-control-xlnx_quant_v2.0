import argparse
import logging
import time

import numpy as np
import torch
from drllib import models, utils

import gym

# ================================
#    Environment HYPERPARAMETERS
# ================================
# RANDOM_SEED = 1234
# Reward function hyperparameters
ALPHA_R = 100.0
BETA_R = 4.5  # 1.25
GAMMA_R = 5.0  # 1.25
DELTA_R = 1.0


def pred_net(net, env, device="cpu"):

    buf = utils.ExperienceBuffer(env.obs_names, env.action_names)
    rewards = 0.0
    steps = 0
    obs = env.reset()
    while True:
        obs_v = utils.float32_preprocessor([obs]).to(device)
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.cpu().numpy()
        action = np.clip(action, -1, 1)

        obs, reward, done, _ = env.step(action)
        # action_scaled = env.scale_action(action)
        # obs_scaled = env.scale_obs(obs)
        action_scaled = action
        obs_scaled = obs
        buf.append(action_scaled, obs_scaled, reward)

        rewards += reward
        steps += 1
        if done:
            break
    actions_df = buf.action_data()
    obs_df = buf.obs_data()
    reward_df = buf.reward_data()

    actions_df.to_csv("saves/actions_df.csv", index=False)
    obs_df.to_csv("saves/obs_df.csv", index=False)
    reward_df.to_csv("saves/reward_df.csv", index=False)

    return rewards, steps


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-m", type=int, default=3, choices=[3, 9], help="MxM beam shape")
    p.add_argument("-p", "--perturb", type=float, default=180, help="start phase perturb range [deg]")
    p.add_argument("-t", "--threshold", type=float, default=0.7, help="target norm. efficiency")
    p.add_argument("-d", "--dat", help="Best reward model", default="saves/best_-0.029_106464.dat")
    args = p.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    config = {"M": args.m, "done_threshold": args.threshold, "perturb_range": args.perturb}
    env = gym.make("laser_cbc:mimocontrol-v0", **config)

    act_net = models.DDPGActor(env.observation_space.shape[-1], env.action_space.shape[-1]).to(device)

    best_model = torch.load(args.dat)
    act_net.load_state_dict(best_model)
    act_net.train(False)
    act_net.eval()

    frame_idx = 0
    best_reward = None

    ts = time.time()
    rewards, steps = pred_net(act_net, env, device=device)
    print("Test done in %.2f sec, reward %.3f, steps %d" % (time.time() - ts, rewards, steps))
    print(env.render())
