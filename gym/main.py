import gym

from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common import make_vec_env
from stable_baselines3 import PPO2
# from stable_baselines3.common.env_checker import check_env
# from env.mimo_env.py import LaserControllerEnv

env = gym.make('laser_cbc:mimocontrol-v0')

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

# from stable_baselines3 import A2C

# env = gym.make('laser_cbc:mimocontrol-v0')
# model = A2C(MlpPolicy, env, verbose=1,  tensorboard_log="./tensorboardoutputs/")
# model.learn(total_timesteps=100000)
# model.save("a2c_laser")
# model = A2C.load("a2c_laser")

# from stable_baselines3 import DDPG

# env = gym.make('laser_cbc:mimocontrol-v0')
# model = DDPG(MlpPolicy, env, verbose=1,  tensorboard_log="./tensorboardoutputs/")
# model.learn(total_timesteps=100000)
# model.save("ddpg_laser")
# model = DDPG.load("ddpg_laser")

# # Evaluate the trained agent
# eval_env = LaserControllerEnv()
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
