import numpy as np
# import matplotlib.pyplot as plt

class LightsOut:
    def __init__(self, n):
        self.n = n

    def step(self, state, action):
        assert state.shape == (self.n, self.n)
        assert action.shape == (2,)
        assert (0 <= state).all() and (state <= 1).all()
        assert (0 <= action).all() and (action < self.n).all()

        new_state = state.copy()
        new_state[action[0], action[1]] = 1 - new_state[action[0], action[1]]

        if action[0] > 0:
            new_state[action[0] - 1, action[1]] = 1 - new_state[action[0] - 1, action[1]]
        if action[0] < self.n - 1:
            new_state[action[0] + 1, action[1]] = 1 - new_state[action[0] + 1, action[1]]
        if action[1] > 0: 
            new_state[action[0], action[1] - 1] = 1 - new_state[action[0], action[1] - 1]
        if action[1] < self.n - 1:
            new_state[action[0], action[1] + 1] = 1 - new_state[action[0], action[1] + 1]

        return new_state, self.is_solved(new_state)

    def _is_solvable(self, state):
        # https://www.reddit.com/r/mathriddles/comments/14vb2eu/lights_out_on_a_2n_1_x_2n_1_grid_is_always/
        assert state.shape == (self.n, self.n)
        assert 2**(np.log2(self.n + 1).astype(int)).astype(int) == self.n + 1
        return True

    def reset(self, nshuffles):
        if nshuffles == -1:
            state = np.random.randint(2, size=(self.n, self.n))
            while not self._is_solvable(state):
                state = np.random.randint(2, size=(self.n, self.n))

            return state
        
        else:
            state = np.zeros((self.n, self.n))
            for _ in range(nshuffles):
                action = np.random.randint(self.n, size=2)
                state, _ = self.step(state, action)

            return state
        
    def is_solved(self, state):
        return np.sum(state) == 0
    
    def render(self, state):
        plt.imshow(state, cmap='gray')
        plt.show()


