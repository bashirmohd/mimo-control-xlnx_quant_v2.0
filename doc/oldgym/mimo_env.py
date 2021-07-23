# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import numpy as np

from gym import Env
from gym import spaces
from scipy import signal

"""
Simulate the Laser beam intensity control environment.

The sim8 function will give us our output and we generate our own input (beam_phs_deg) to collect the training data

"""

INPUT_SIZE = 9
OUTPUT_SIZE = 25

# convolution algorithem
# DOE phase response, measured
_doe_phs_deg = np.array([
    0,  90,  0,
    0,   0, -90,
    180, 0,  0]).reshape(3, 3)
_doe_amp = np.ones_like(_doe_phs_deg)
_doe_amp[1, 1] = 0

D_MATRIX = _doe_amp * np.exp(1j * np.deg2rad(_doe_phs_deg))
BEAM_AMP = _doe_amp  # laser beam amplitude
BEAM_PHS_IDEAL = -np.rot90(np.rot90(_doe_phs_deg))  # laser beam working phase


class LaserControllerEnv(Env):

    """
    Define LaserController environment

    The enviromentment for Laser beam intensity control environment for OpenAI gym.

    State space: Observation space looking at the 25 cells.
    State/observation vector is shape (9,) and action vector is  also of (9,).
    But at the time of giving our reward, we will look to the 25 cells.
    If middle cell no[12] higher intensity than surrounding cells reward =+1
    if other cells has higher intensity reward =-1, all other states =0
    Reward function: only need one. camera will be saturated in middle  contrast ratio,
    Reward to be changed to contrast ratio :> combined efficiency
    number of middle one divide by sum of total elements so middle one will be high compared to others
    Objective : To keep the center beam intensity  highest


    Observation:
        Type: Box()
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

    """
    metadata = {'render.modes': ['human']}

    def __init__(
            self, laser00=None, laser01=None, laser02=None, laser10=None,
            laser11=None, laser12=None, laser20=None, laser21=None, laser22=None, step_size=None):
        # declare all variables
        self.laser00 = laser00
        self.laser01 = laser01
        self.laser02 = laser02
        self.laser10 = laser10
        self.laser11 = laser11
        self.laser12 = laser12
        self.laser20 = laser20
        self.laser21 = laser21
        self.laser22 = laser22

        self._shape = (5, 5)
        self.start_time = 0
        self.final_time = 1000
        self.step_size = step_size
        self.time_steps = np.arange(self.start_time, self.final_time, self.step_size)
        self.n_steps = len(self.time_steps)

        self.episode_idx = -1

        # MK: What is episode fail point
        # Action space is controlling 9 beams

        indx = [0, 1, 2, 3, 5, 6, 7, 8]
        self.min_action = np.ones(8) * -180 + np.reshape(BEAM_PHS_IDEAL, (-1))[indx]
        self.max_action = np.ones(8) * 180 + np.reshape(BEAM_PHS_IDEAL, (-1))[indx]

        self.action_init = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            dtype=np.float32
        )

        # Observation space
        self.min_observation_space = np.zeros(25)
        self.max_observation_space = np.ones(25)*64
        self.observation_space = spaces.Box(
            low=self.min_observation_space, high=self.max_observation_space, dtype=np.float32)
        # self.observation_space = spaces.Box(5, 5, dtype=np.float32)
        print("in gym")



        # initialize the state

        self.episode_rewards = []  #used for plotting
        self.seed()


    def sim8(self, action):
        a_list = action.tolist()
        beam_phs_deg = np.array(a_list[:4] + [0] + a_list[4:8]).reshape(3, 3)
        # beam_phs_deg[0][0] = 0
        # Force DOE center output to 0 by design
        b = BEAM_AMP * np.exp(1j * np.deg2rad(beam_phs_deg))
        s = signal.convolve2d(b, D_MATRIX)
        return np.abs(s * s.conj())

    def angle_normalize(x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def seed(self, seed=None):
        ini_action = np.random.uniform(low=self.min_action, high=self.max_action)
        ini_a_list = ini_action.tolist()
        beam_phs_deg = np.array(ini_a_list[:4] + [0] + ini_a_list[4:8]).reshape(3, 3)
        seed = self.sim8(ini_action)
 #       self.np_random, seed = seeding.np_random(seed)
 #       print(seed)
        ##### np.random.uniform(low=self.min_observation_space, high=self.max_observation_space)
        return [seed]

    def reset(self):
        self.time_step_idx = 0
        self.episode_idx =+ 1
        self.reward = 0.0
        self.epi_reward = -10
        self.time_step = 0
        self.state = np.random.uniform(low=self.min_observation_space, high=self.max_observation_space)
        return self.state

#    def _get_obs(self):
#        max_S = np.amax(self.state)
#        S_center = self.state[2][2]
#        S_contract = current_center/(np.sum(self.state)-current_center)
#        return np.array([max_S, S_center,S_contract])


    def step(self, action):
        S_previous = self.state  # th := theta
        #print(S_previous)
        S_new = self.sim8(action)
        #print(action)
        self.state = S_new  # 5 by 5 output
        S_center = self.state[2][2]
        #print(S_new)
        S_contract = S_center/(np.sum(self.state)-S_center)

        costs = (64 - S_center) ** 1
        # + .001 * (self.angle_normalize(action)) ** 2
        rewards = -costs

        self.epi_reward = rewards
        self.time_step +=1
        # print(rewards)

        if self.time_step == 50:
            plot_flag = True
            self.episode_rewards.append(self.epi_reward)

        return self.state, rewards, False, {}


    def render(self, mode='human', close=False):
        pass







class ExperienceBuffer:
    def __init__(self, obs_names, action_names):
        # initialize model observation
        self.obs_names = obs_names
        self.action_names = action_names

        obs_dict = dict()
        for obs_i in obs_names:
            obs_dict[obs_i] = [0.0]
        self.obs_df = pd.DataFrame(obs_dict)

        # initialize actions
        action_dict = dict()
        for action_i in self.action_names:
            action_dict[action_i] = [0.0]
        self.actions_df = pd.DataFrame(action_dict)

        # initialize rewards
        self.rewards_df = pd.DataFrame({"reward": [0.0]})

              


    def append(
        self, action, obs, 
        reward):

        action_dict = dict()
        for i in range(len(self.action_names)):
            action_i = self.action_names[i]
            action_dict[action_i] = [action[i]]
        action_df_0 = pd.DataFrame(action_dict)
        self.actions_df = self.actions_df.append(action_df_0, ignore_index=True)

        obs_dict = dict()
        for i in range(len(self.obs_names)):
            obs_i = self.obs_names[i]
            obs_dict[obs_i] = [obs[i]]
        obs_df_0 = pd.DataFrame(obs_dict)
        self.obs_df = self.obs_df.append(obs_df_0, ignore_index=True)

        reward_df_0 = pd.DataFrame({"reward": [reward]})
        self.rewards_df = self.rewards_df.append(reward_df_0, ignore_index=True)

       

    def last_action(self):
        return self.actions_df.iloc[len(self.actions_df) - 1]

    def action_data(self):
        return self.actions_df

    def obs_data(self):
        return self.obs_df

    def reward_data(self):
        return self.rewards_df

    
