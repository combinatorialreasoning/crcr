import random
import gin
import jax
import joblib
import numpy as np

from jax_rand import next_key
from jumanji_utils.sokoban_env import Sokoban
from jumanji.environments.routing.sokoban.generator import Generator
import jax.numpy as jnp
from jumanji.environments.routing.sokoban.generator import convert_level_to_array
from jumanji.environments.routing.sokoban.types import State
import torch

from search.gen_sokoban import convert_to_board, convert_to_state, solved, to_jumanji

@gin.configurable()
class CustomSokobanEnv(Sokoban):
    def __init__(self, grid_size, generator, **unused_kwargs):
        super().__init__(generator=generator())
        
        self.grid_size = grid_size
        self.num_cols = grid_size
        self.num_rows = grid_size
        self.shape = (self.num_rows, self.num_cols)

    def in_grid(self, coordinates):
        return jnp.all((0 <= coordinates) & (coordinates < self.grid_size))
    
    def reset(self):
        return super().reset(next_key())
    
    def step(self, state, action):
        state_obj = convert_to_state(state.reshape((int(state.size**0.5), int(state.size**0.5))))
        new_state, timestep = super().step(state_obj, action)
        new_obs_str = convert_to_board(new_state).flatten()
        done = solved(timestep)

        return new_obs_str, None, done, None
    
    def get_all_actions(self):
        return [0, 1, 2, 3]
        
    
@gin.configurable()
class CustomSokobanGenerator(Generator):
    def __init__(self, boards_path):
        self.boards = joblib.load(boards_path)
        self.boards = np.array(self.boards).argmax(axis=-1)
        self.i = 0

    def __call__(self, key):
        self.i += 1
        assert self.i <= len(self.boards)

        board = self.boards[self.i - 1]

        fixed, variable = to_jumanji(torch.from_numpy(board))
        initial_agent_location = self.get_agent_coordinates(variable)

        return State(
            key=key,
            fixed_grid=fixed,
            variable_grid=variable,
            step_count=jnp.array(0, jnp.int32),
            agent_location=initial_agent_location
        )
    
    


def main():
    key = jax.random.key(0)
    env = CustomSokobanEnv(grid_size=12, generator=CustomSokobanGenerator())

    for i in range(10):
        key, new_key = jax.random.split(key)
        state, _= env.reset(new_key)
        
        for i in range(10):
            key, new_key = jax.random.split(key)
            state, done = env.step(state, random.randint(0, 3))

            env.render(state)
    
if __name__ == "__main__":
    main()