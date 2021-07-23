from gym.envs.registration import register

register(
    id="mimocontrol-v0", entry_point="laser_cbc.envs:LaserControllerEnv",
)
__all__ = ["envs"]
