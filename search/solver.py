from queue import PriorityQueue
import random

import numpy as np

from search.utils import gen_rubik_data
from search.utils.rubik_solver_utils import cube_to_string, make_RubikEnv
from search.value_function import ValueEstimatorRubik
import gin
import torch

GOAL = "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww"

class SolverNode:
    def __init__(self, state, parent, depth, child_num, path, done):
        self.state = state
        self.parent = parent
        self.depth = depth
        self.child_num = child_num
        self.path = path
        self.done = done
        self.children = []
        self.hash = state

    def add_child(self, child):
        self.children.append(child)

    def set_value(self, value):
        self.value = value


class GeneralSolver():
    def __init__(self):
        # self.core_env = make_RubikEnv()
        pass

@gin.configurable
class BestFSSolverRubik(GeneralSolver):
    def __init__(self,
                 metric,
                 shuffles,
                 n_actions=None,
                 sample_actions=False,
                 goal_builder_class=None,
                 max_tree_size=None,
                 max_tree_depth=None,
                 network=None,
                 checkpoint_path=None,
                 value_estimator_class=None,
                 include_actions=False,
                 sampling_temperature=1.
                 ):
        super().__init__()
        self.max_tree_size = max_tree_size
        self.max_tree_depth = max_tree_depth
        self.goal_builder_class = goal_builder_class
        self.goal_builder = self.goal_builder_class(shuffles=shuffles)
        self.value_estimator_class = value_estimator_class
        self.value_estimator = self.value_estimator_class(network, include_actions=include_actions, checkpoint_path=checkpoint_path, metric=metric)
        self.buffer = []
        self.n_actions = n_actions
        self.sample_actions = sample_actions
        self.sampling_temperature = sampling_temperature
        
        if sample_actions:
            assert n_actions is not None, "To sample actions, number of actions needs to be specified"
        
    def construct_networks(self):
        self.goal_builder.construct_networks()
        self.value_estimator.construct_networks()

    def solve(self, input, goal):
        solved = False
        solved_state=goal
        # self.goal_builder.env.unwrapped.set_solved_state(solved_state)
        
        root = SolverNode(input, None, 0, 0, [], False)
        nodes_queue = PriorityQueue()
        # To prevent situations where two exactly same states (thus with the same value),
        # cannot be compared, we add another dimension
        # with random number which are being compared in these rare situations.
        # TODO value estimator has to get a 2 by 2 cube
        
        # solved_state = gen_rubik_data.cube_bin_to_str(self.goal_builder.env.unwrapped.get_solved_state())
        root_value = self.value_estimator.get_solved_distance(root.state, solved_state)
        
        root.set_value(root_value)
        nodes_queue.put((root_value, random.random(), root))
        solution = []
        tree_size = 1
        expanded_nodes = 0
        all_goals_created = 0
        tree_depth = 0
        total_path_between_goals = 0
        seen_hashed_states = {str(root.hash)}
        finished_cause = 'Not determined'

        while True:
            if nodes_queue.empty():
                finished_cause = 'Finished cause queue is empty'
                break
            if tree_size >= self.max_tree_size:
                finished_cause = 'Finished cause tree too big'
                break
            if solved:
                finished_cause =  'Finished cause solved'
                break

            #pop node from queue to expand
            curr_val, _, current_node = nodes_queue.get()
            self.buffer.append(curr_val)
            if len(self.buffer) >= 10:
                self.buffer = self.buffer[1:]

            # self.loggers.log_scalar('value', 0, curr_val)
            # print(f'val = {sum(self.buffer) / len(self.buffer)} | {current_node.state}')
            expanded_nodes += 1

            # print(logic_statement_to_seq_string(current_node.state['observation']['objectives'][0]))

            if current_node.depth < self.max_tree_depth:
                goals, solving_subgoal = self.goal_builder.build_goals(current_node.state)

                # look for solution
                if solving_subgoal is not None:
                    solving_state, path, done = solving_subgoal
                    new_node = SolverNode(solving_state, current_node, current_node.depth + 1, 0,
                               path, True)
                    solution.append(new_node)
                    solved = True
                    finished_cause = 'Finished cause solved'
                    tree_size += 1
                    expanded_nodes += 1
                    break


                all_goals_created += len(goals)

                created_new = 0

                to_add_list = []
                for child_num, goal_proposition in enumerate(goals):
                    current_goal_state, current_path, _ = goal_proposition
                    current_goal_state_hash = str(current_goal_state)

                    if current_goal_state_hash not in seen_hashed_states:
                        new_node = SolverNode(current_goal_state, current_node, current_node.depth + 1, child_num, current_path, False)
                        node_val = self.value_estimator.get_solved_distance(new_node.state, solved_state)
                    
                        to_add_list.append((node_val, random.random(), goal_proposition))

                to_add_list = sorted(to_add_list)

                if self.n_actions is not None:
                    if self.sample_actions:
                        probabilities = torch.concat([elt[0] for elt in to_add_list])
                        probabilities = torch.nn.functional.softmax(-probabilities/self.sampling_temperature)
                        to_add_list = random.choices(to_add_list, weights=probabilities, k=self.n_actions)
                    else:
                        to_add_list = to_add_list[:self.n_actions]
                    
                for child_num, (node_val, _, goal_proposition) in enumerate(to_add_list):
                    current_goal_state, current_path, _ = goal_proposition

                    current_goal_state_hash = str(current_goal_state)
                    total_path_between_goals += len(current_path)

                    if current_goal_state_hash not in seen_hashed_states:
                        created_new += 1
                        seen_hashed_states.add(current_goal_state_hash)
                        new_node = SolverNode(current_goal_state, current_node, current_node.depth + 1, child_num, current_path, False)
                        current_node.add_child(new_node)
                        tree_depth = max(tree_depth, new_node.depth)
                        new_node.set_value(node_val)
                        nodes_queue.put((node_val, random.random(), new_node))
                        tree_size += 1



        tree_metrics = {'nodes' : tree_size,
                        'expanded_nodes': expanded_nodes,
                        'unexpanded_nodes': tree_size - expanded_nodes,
                        'max_depth' : tree_depth,
                        'avg_n_goals': all_goals_created/expanded_nodes if expanded_nodes > 0 else 0,
                        'avg_dist_between_goals' : total_path_between_goals/all_goals_created
                        if all_goals_created > 0 else 0
                        }

        additional_info = {'finished_cause': finished_cause}

        if solved:
            node = solution[0]
            while node.parent is not None:
                solution.append(node.parent)
                node = node.parent

            trajectory_actions = []
            for inter_goal in solution:
                trajectory_actions = list(inter_goal.path) + trajectory_actions

            inter_goals = [node for node in reversed(solution)]
            return (inter_goals, tree_metrics, root, trajectory_actions, additional_info)
        else:
            return (None, tree_metrics, root, None, additional_info)