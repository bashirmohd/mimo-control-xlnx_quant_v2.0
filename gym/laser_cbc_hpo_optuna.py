import optuna
import gym

from stable_baseline3 import DDPG
from stable_baseline3.common.evaluation import evaluate_policy
from stable_baseline3.common.cmd_util import make_vec_env


n_cpu = 4


def optimize_ddpg(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_ddpg(trial)
    # env = DummyVecEnv([lambda: gym.make("laser_cbc:mimocontrol-v0") for i in range(n_cpu)])
    env = make_vec_env(lambda: gym.make('laser_cbc:mimocontrol-v0'), n_envs=16, seed=0)
    model = DDPG('MlpPolicy', env, verbose=0, nminibatches=1, **model_params)
    model.learn(10000)
    mean_reward, _ = evaluate_policy(model, gym.make('laser_cbc:mimocontrol-v0'), n_eval_episodes=10)

    return -1 * mean_reward


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(optimize_agent, n_trials=100, n_jobs=4)
