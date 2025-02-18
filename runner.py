import gin

import argparse

import cloudpickle
import pickle
import os
import torch 

import metric_logging
from jobs import generating_job, generating_job_baseline
from search.utils.rubik_solver_utils import generate_problems_rubik 
import datasets.new_array_dataset
import datasets.new_array_dataset_len
import datasets.new_array_dataset_baseline_len
import datasets.new_array_dataset_baseline
from search import value_function_baseline, value_function
from search import gen_problems_sokoban
import sokoban_env
from losses import get_mrn_dist

from lights_out import env, data_generate
from npuzzle import env, data_generate


from networks import LNDenseNet
from search.solve_job import JobSolveRubik
from search.solver import BestFSSolverRubik
from search.goal_builder import VanillaPolicyRubik
from losses import vanilla_loss, log_lambda_loss, mrn_loss
from search.utils.rubik_solver_utils import make_RubikEnv

import random
import numpy as np
import jax_rand

gin.external_configurable(torch.utils.data.DataLoader, module='torch.utils.data')

def get_configuration_path(spec_path):
    """Get mrunner experiment specification and gin-config overrides."""
    try:
        with open(spec_path, 'rb') as f:
            specification = cloudpickle.load(f)
    except pickle.UnpicklingError:
        with open(spec_path) as f:
            vars_ = {'script': os.path.basename(spec_path)}
            exec(f.read(), vars_)  # pylint: disable=exec-used

            specification = vars_['experiments_list'][0].to_dict()
            # TODO I can write the datapath into an environmental variable

            print('NOTE: Only the first experiment from the list will be run!')

    parameters = specification['parameters']
    gin_bindings = []
    for key, value in parameters.items():
        if isinstance(value, str) and not (value[0] == '@' or value[0] == '%'):
            binding = f'{key} = "{value}"'
        else:
            binding = f'{key} = {value}'
        gin_bindings.append(binding)

    return specification, gin_bindings


@gin.configurable
def run(job_class, args, seed, specification=None):
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    jax_rand.set_seed(seed)

    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    
    loggers = metric_logging.Loggers()
    loggers.register_logger(metric_logging.StdoutLogger())

    if args.use_neptune:
        neptune_logger = metric_logging.NeptuneLogger(specification=specification)
        loggers.register_logger(neptune_logger)

    loggers.log_property('seed', seed)
    job = job_class(
        loggers,
    )

    job.execute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", action='append', default=[])
    parser.add_argument("--config_file", action='append', default=[])
    parser.add_argument("--use_neptune", action="store_true")

    args = parser.parse_args()

    gin_bindings = args.config
    path = gin_bindings.pop()

    if not args.config_file:
        gin_bindings = args.config

        specification, overrides = get_configuration_path(path)

        # if not run from mrunner, get the arguments from the script
        script = specification["script"]
        args2 = parser.parse_args(script.split()[2:])
        args.config_file = args2.config_file
        args.use_neptune = args2.use_neptune

        gin_bindings.extend(overrides)

    else:
        specification, overrides = get_configuration_path(path)
        gin_bindings.extend(overrides)

    params = {}
    with open(args.config_file[0], 'r') as handle:
        for line in handle.readlines():
            splitted = line.split()
            if len(splitted) != 3:
                continue

            k = splitted[0]
            v = splitted[2]

            params[k] = v
    
    params.update(specification["parameters"])
    specification["parameters"] = params

    gin.parse_config_files_and_bindings(args.config_file, gin_bindings)
    run(args=args, specification=specification)
