import gymnasium as gym


def make_env(env_name="CartPole-v1"):
    return gym.make(env_name)