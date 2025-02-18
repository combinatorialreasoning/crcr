import time
import joblib
import numpy as np
import torch

from npuzzle.env import NPuzzle

def step_wrapped(sa, env):
    state = sa[:-1]
    action = sa[-1]
    return env.step(torch.from_numpy(state), action)[0]

def gen_trajectory_batch(batch_size, n, length):
    env = NPuzzle(n, -1)
    states = np.arange(n ** 2).reshape(1, -1).repeat(batch_size, axis=0)


    trajectory = np.zeros((batch_size,length,  n ** 2), dtype=int)
    reverse = np.array([1, 0, 3, 2])
    import tqdm
    for i in tqdm.tqdm(range(length)):
        # if i == 0:
        trajectory[:, length - i - 1] = states
        if i == 0:
          actions = np.random.randint(0, 4, size=batch_size)
        if i > 0:
          actions = np.random.randint(0, 3, size=batch_size)
          mask = actions >= rev_actions
          actions[mask] += 1
        sa = np.concatenate([states, actions[:, None]], axis=1)

        states = np.apply_along_axis(step_wrapped, 1, sa, env)
        rev_actions = reverse[actions]
    
    return trajectory

def generate_problems(num_problems, shuffles=-1):
    env = NPuzzle(shuffles=shuffles)
    problems = []
    for _ in range(num_problems):
        state = env.reset()
        problems.append(state)
    return problems

def get_goal_state(problem):
    return torch.arange(len(problem), dtype=problem.dtype, device=problem.device)

def main():
    import os
    save_path = 'npuzzle_data'
    os.mkdir(save_path)
    board_size = 4
    length = 150
    for traj in range(100):
        trajectory = gen_trajectory_batch(10000, board_size, length)
        # import pdb; pdb.set_trace()
        joblib.dump(torch.tensor(trajectory).to(torch.float32), f'{save_path}/npuzzle_trajectories_{traj}.pkl') 


if __name__ == "__main__":
    main()
