import gym_rubik
def make_env_Rubik(**kwargs):
    env = gym_rubik.envs.rubik_env.RubikEnv(**kwargs)

    return env