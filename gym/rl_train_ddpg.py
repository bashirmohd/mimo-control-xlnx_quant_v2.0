#!/usr/bin/env python
import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from drllib import models, utils
from tensorboardX import SummaryWriter

import gym

np.set_printoptions(precision=0, suppress=True, linewidth=150)
logger = logging.getLogger(__name__)

GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
TEST_ITERS = 5  # compute test evaluation every TEST_ITERS episodes
SAVE_ITERS = 1000  # save best models every SAVE_ITERS episodes
RunName = "Test5"


def test_net(net, env, writer, exp_idx, count=1, device="cpu"):
    acc_reward = 0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = utils.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, info = env.step(action)
            acc_reward += reward
            steps += 1
            if done:
                break
    return acc_reward / count, steps / count


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-m", type=int, default=3, choices=[3, 9], help="MxM beam shape")
    p.add_argument("-p", "--perturb", type=float, default=180, help="start phase perturb range [deg]")
    p.add_argument("-t", "--threshold", type=float, default=0.7, help="target norm. efficiency")
    args = p.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join("saves", "ddpg-" + RunName)

    # save all values in dataframe:
    advantage = []
    values = []
    loss_value = []
    episode_steps = []
    test_reward = []
    test_steps = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config = {"M": args.m, "done_threshold": args.threshold, "perturb_range": args.perturb}
    env = gym.make("laser_cbc:mimocontrol-v0", **config)
    test_env = gym.make("laser_cbc:mimocontrol-v0", **config)

    act_net = models.DDPGActor(env.observation_space.shape[-1], env.action_space.shape[-1]).to(device)
    crt_net = models.DDPGCritic(env.observation_space.shape[-1], env.action_space.shape[-1]).to(device)
    print("DDPG Actor  net:\n{}".format(act_net))
    print("DDPG Critic net:\n{}".format(crt_net))

    tgt_act_net = utils.TargetNet(act_net)
    tgt_crt_net = utils.TargetNet(crt_net)

    writer = SummaryWriter(comment="-ddpg_" + RunName)

    agent = models.AgentDDPG(act_net, device=device)
    exp_source = utils.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buf = utils.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = torch.optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = torch.optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    best_reward = -np.inf
    mean_reward = -np.inf
    exp_idx = 0
    episodes = 0
    with utils.RewardTracker(writer) as tracker:
        with utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                exp_idx += 1
                buf.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], exp_idx)
                    tracker.reward(rewards[0], exp_idx)
                    episodes += 1
                    print(" ====== episodes: %d ====== " % (episodes))

                if len(buf) < REPLAY_INITIAL:
                    continue

                batch = buf.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = utils.unpack_batch_ddqn(batch, device)

                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask] = False
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                critic_loss_v = torch.nn.functional.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, exp_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), exp_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, exp_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                # tests, avoid noisy results at beginning
                if exp_idx % TEST_ITERS == 0 and exp_idx > 100:
                    ts = time.time()
                    reward, steps = test_net(act_net, test_env, writer, exp_idx, device=device)
                    test_reward.append(reward)
                    test_steps.append(steps)
                    mean_reward = np.mean(test_reward[-100:])
                    print(
                        "Test done in {:5.1f}s, best_reward {:8.1f}, mean_reward {:8.1f}, steps {:8.0f}".format(
                            time.time() - ts, best_reward, mean_reward, steps
                        )
                    )
                    writer.add_scalar("test_reward", reward, exp_idx)
                    writer.add_scalar("test_steps", steps, exp_idx)

                    # find max mean rewards and save the best trained NN
                    if mean_reward > best_reward:
                        print("Best reward updated: %8.1f -> %8.1f" % (best_reward, mean_reward))
                        best_reward = mean_reward

                if exp_idx % SAVE_ITERS == 0 and exp_idx > 2000:
                    fname = os.path.join(save_path, "best_%+.1f_%d.dat" % (best_reward, exp_idx))
                    print(" ===== Best model saved as {}. ===== ".format(fname))
                    torch.save(act_net.state_dict(), fname)
                    # save rewards to file
                    df = pd.DataFrame(data={"test_reward": test_reward, "test_step": test_steps})
                    df.to_csv("saves/test_reward.csv", sep=",", index=False)
