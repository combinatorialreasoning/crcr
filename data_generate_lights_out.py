import time
import joblib
import numpy as np
import torch
import tqdm
from generate_lo_env import LightsOut

def step_wrapped(sa, env):
    state = sa[:-2]
    action = sa[-2:]
    return env.step(state.reshape((env.n, env.n)), action)[0]

def gen_trajectory_batch(batch_size, n, length):
    env = LightsOut(n)
    states = np.zeros(n ** 2).reshape(1, -1).repeat(batch_size, axis=0).astype(int)


    trajectory = np.zeros((batch_size,length,  n ** 2), dtype=int)
    for i in tqdm.tqdm(range(length)):
        trajectory[:, length - i - 1] = states
        actions = np.random.randint(n, size=(batch_size, 2))
        sa = np.concatenate([states, actions], axis=1)
        states = np.apply_along_axis(step_wrapped, 1, sa, env)
        states = states.reshape(states.shape[0], -1)
    return trajectory

def generate_problems(num_problems, shuffles):
    env = LightsOut(shuffles=shuffles)
    problems = []
    for _ in range(num_problems):
        state = env.reset()
        problems.append(state)
    return problems

def get_goal_state(problem):
    return torch.zeros_like(problem)

def main():
    import os
    board_size = 7
    length = 49
    env = LightsOut(7)
    save_path = 'lights_out_data'
    os.mkdir(save_path)
    for traj in range(1000):
        trajectory = gen_trajectory_batch(10000, board_size, length)
        # import pdb; pdb.set_trace()
        joblib.dump(torch.tensor(trajectory).to(torch.float32), f'{save_path}/lights_out_trajectories_{traj}.pkl') 


if __name__ == "__main__":
    main()
