from copy import deepcopy

from search.utils import rubik_solver_utils, gen_rubik_data
from search.utils.gen_rubik_data import encode_policy_data
from search.utils.rubik_solver_utils import make_RubikEnv, cube_to_string

import gin

@gin.configurable
class VanillaPolicyRubik:
    def __init__(self, shuffles, env, device=None, n_actions=None, num_beams=None, temperature=None, possible_actions = 12):
        self.model = None
        self.device = device
        self.env = env(shuffles=shuffles)
        self.env.reset()
        self.n_actions = n_actions
        self.num_beams = num_beams
        self.temperature = temperature

        assert possible_actions in [12, 18]
        self.possible_actions = possible_actions

    def construct_networks(self):
        pass

    def build_goals(self, state):
        actions = self.env.get_all_actions()
        goals = []

        for action in actions:
            new_obs, _, done, _ = self.env.step(state=state, action=action)

            if done:
                return goals, (new_obs, [action], done)

            goals.append((new_obs, [action], done))

        return goals, None

    def solve(self, proof_state, debugg_mode=False):
        path = []
        reaching_goal_debug_info = {}
        self.env.load_state(gen_rubik_data.cube_str_to_state(proof_state[1:-1]))
        subgoal_reached = False
        steps_taken = 0
        current_proof_state = deepcopy(proof_state)
        seen_states = {proof_state}
        while not subgoal_reached and steps_taken < 100:
            if debugg_mode:
                print(f'Step = {steps_taken} \n curr = {current_proof_state}')
            actions, debugg_info = self.predict_actions(current_proof_state, 32, 1)
            if len(actions) > 0:
                action = actions[0][0]
                path.append(action)
                if action is None:
                    reaching_goal_debug_info['None action'] = True
                    if debugg_mode:
                        print('None action')
                    return False
                else:
                    if debugg_mode:
                        print(f'{action} | ()')
                    new_obs, _, done, _ = self.env.step(action)

                    new_obs_str = cube_to_string(new_obs)
                    new_obs = new_obs_str

                    if new_obs_str in seen_states and not done:
                        print('Seen state')
                        debugg_info['seen_state'] = True
                        return False, debugg_info

                    seen_states.add(new_obs_str)
                    current_proof_state = deepcopy(new_obs)
                    if done:
                        return True, debugg_info
                steps_taken += 1
            else:
                steps_taken += 1
                break
        return False, {'limit': True}