from collections import defaultdict
import pickle
import time
from copy import deepcopy
import sys

from joblib import Parallel, delayed

# TODO 
import gin

def solve_problem(solver, problem):
    sys.setrecursionlimit(1000000)
    time_s = time.time()
    input_state, solved_state = problem
    solution, tree_metrics, root, trajectory_actions, additional_info = solver.solve(input_state, solved_state)
    time_solving = time.time() - time_s
    return dict(
        solution=solution,
        tree_metrics=tree_metrics,
        root=root,
        trajectory_actions=trajectory_actions,
        time_solving=time_solving,
        input_problem=deepcopy(input_state),
        additional_info=additional_info
    )

@gin.configurable
class JobSolveRubik():
    def __init__(self,
                 loggers,
                 solver_class,
                 n_jobs,
                 n_parallel_workers,
                 batch_size,
                 metric,
                 network,
                 generate_problems,
                 shuffles,
                 n_actions=None,
                 budget_checkpoints=None,
                 include_actions=False,
                 log_solutions_limit=100,
                 collect_solutions=None,
                 sample_actions=False
                 ):

        self.loggers = loggers
        self.solver_class = solver_class
        self.n_jobs = n_jobs
        self.n_parallel_workers = n_parallel_workers
        self.batch_size = batch_size
        self.budget_checkpoints = budget_checkpoints
        self.log_solutions_limit = log_solutions_limit
        self.collect_solution = collect_solutions
        self.shuffles = shuffles
        self.n_actions = n_actions
        self.include_actions = include_actions
        self.sample_actions = sample_actions
        self.generate_problems = generate_problems
        self.solution_lengths = []
        self.budget_solved = defaultdict(int)
        self.budget_exp_solved = defaultdict(int)
        self.prefix = f"{self.shuffles}_shuffles_{'search' if self.n_actions != 1 else 'argmax'}"

        self.logged_solutions = 0

        if self.collect_solution is not None:
            self.collection = {}

        self.solved_boards = 0
        self.all_boards = 0
        self.network = network
        self.metric = metric

    def execute(self):            
        solver = self.solver_class(network = self.network, include_actions=self.include_actions, metric=self.metric, n_actions=self.n_actions, sample_actions=self.sample_actions, shuffles=self.shuffles)
        solver.construct_networks()

        problems_to_solve = self.generate_problems(self.n_jobs, self.shuffles)
        

        jobs_done = 0
        jobs_to_do = self.n_jobs
        batch_num = 0
        all_batches = self.n_jobs // self.batch_size
        
        all_results = []

        total_time_start = time.time()
        while jobs_to_do > 0:
            jobs_in_batch = min(jobs_to_do, self.batch_size)
            problems_to_solve_in_batch = problems_to_solve[jobs_done:jobs_done+jobs_in_batch]
            print('============================ Batch {:>4}  out  of  {:>4} ============================'.
                  format(batch_num+1, all_batches))
            results = Parallel(n_jobs=self.n_parallel_workers, verbose=10000)(
                delayed(solve_problem)(solver, input_problem) for input_problem in problems_to_solve_in_batch
            )

            all_results += results

            print('===================================================================================')


            jobs_done += jobs_in_batch
            jobs_to_do -= jobs_in_batch
            batch_num += 1

        self.log_results(all_results, jobs_done)

       
    def log_results(self, results, step):        
        for log_num, result in enumerate(results):
            if self.logged_solutions < self.log_solutions_limit:
                self.log_solution(result['solution'], result['trajectory_actions'], result['input_problem'], step+log_num)
            solved = result['solution'] is not None
            if solved:
                self.solved_boards += 1
                self.solution_lengths.append(len(result['solution']))
                for budget in self.budget_checkpoints:
                    if result['tree_metrics']['nodes'] < budget:
                        self.budget_solved[budget] += 1
                    if result['tree_metrics']['expanded_nodes'] < budget:
                        self.budget_exp_solved[budget] += 1
                    
            self.all_boards += 1
            
        def get_name(name):
            if self.prefix is not None:
                return f'{self.prefix}/{name}'
            return name
            
        self.loggers.log_scalar(get_name('solved_rate'), 0, self.solved_boards / self.all_boards)
        
        avg_length = 0 if not self.solution_lengths else (sum(self.solution_lengths)/len(self.solution_lengths))
        self.loggers.log_scalar(get_name('avg_length'), 0, avg_length)
        
        for budget in self.budget_checkpoints:
            self.loggers.log_scalar(get_name(f'solved_rate_{budget}_nodes'), 0, self.budget_solved[budget] / self.all_boards)
            self.loggers.log_scalar(get_name(f'solved_rate_{budget}_exp_nodes'), 0, self.budget_exp_solved[budget] / self.all_boards)
           
           
    def log_solution(self, solution, trajectory_actions, input_problem, step):

        if solution is not None:
            solution_str = f'Problem {step} : {solution[0].hash} \n'
            for subgoal_num, node in enumerate(solution[1:]):
                solution_str += f'subgoal {subgoal_num} : {node.hash} \n'
            solution_str += '\n \n'
            solution_str += 'Actions: \n'
            for action_num, action in enumerate(trajectory_actions):
                solution_str += f'action {action_num}: ({action}, () ) \n'

        else:
            solution_str = f'Unsolved problem {step} : {input_problem} \n \n'


        if self.collect_solution is not None:
            if solution is not None:
                new_path = [(node.hash, -node.value) for node in solution[:-1]]
                new_path.append((solution[-1].hash, 0))
            else:
                new_path = None
            self.collection[len(self.collection)] = new_path
            print(new_path)
            pickle.dump(self.collection, open(self.collect_solution, "wb"))