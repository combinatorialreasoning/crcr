import torch
from sokoban_env import CustomSokobanEnv, CustomSokobanGenerator
from search.gen_sokoban import from_jumanji, get_solved_state
import jax
import numpy as np
from jax_rand import next_key
import gin

@gin.configurable()
def generate_problems_sokoban(n_problems, *unused_args):
    problems = []
    env = CustomSokobanEnv(grid_size=12, generator=CustomSokobanGenerator)

    for _ in range(n_problems):
        state, timestep = env.reset()
        state = from_jumanji(state.fixed_grid, state.variable_grid)
        solved_state = get_solved_state(state)
        problems.append((state.astype(np.float32).flatten(), solved_state.astype(np.float32).flatten()))

    return problems
