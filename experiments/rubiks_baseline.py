from mrunner.helpers.specification_helper import create_experiments_helper
import os

base_config = {
}


params_grid = {
    'idx': [0],
    'jobs.generating_job_baseline.TrainJob.train_steps': [500000000],
    'jobs.generating_job_baseline.TrainJob.batch_size': [512],
    'jobs.generating_job_baseline.TrainJob.chunk_size': [None],
    'jobs.generating_job_baseline.TrainJob.dataset_class': ["@new_array_dataset_baseline.GenDataset"],
    'jobs.generating_job_baseline.TrainJob.loss_fn': ["@losses.vanilla_loss"],
    'vanilla_loss.loss_type': ['symmetric'],
    'jobs.generating_job_baseline.TrainJob.metric': ['l22'],
    'jobs.generating_job_baseline.TrainJob.search_shuffles': [[8, 12, 16]],
    'jobs.generating_job_baseline.TrainJob.c': [10],
    'jobs.generating_job_baseline.TrainJob.lr': [3e-4],
    'jobs.generating_job_baseline.TrainJob.model_type': ["@LNDenseNet"],
    'jobs.generating_job_baseline.TrainJob.eval_job_class': ['@JobSolveRubik'],
    'new_array_dataset_baseline.GenDataset.gamma': [0.9],
    'new_array_dataset_baseline.GenDataset.path': ['rubik_dataset'],
    f'new_array_dataset_baseline.GenDataset.max_horizon': [200],
    'LNDenseNet.input_size': [108],
    'LNDenseNet.repr_dim': [21],
    'LNDenseNet.hidden_size': [1024],
    'LNDenseNet.depth': [8],
    'VanillaPolicyRubik.possible_actions': [12],
    'jobs.generating_job_baseline.TrainJob.solving_interval': [150000],
    'BestFSSolverRubik.value_estimator_class': ['@value_function_baseline.ValueEstimatorRubik'],
    'VanillaPolicyRubik.env': ['@make_RubikEnv'],
    'JobSolveRubik.generate_problems': ['@generate_problems_rubik'],
    'run.job_class': ['@jobs.generating_job_baseline.TrainJob'],
    'run.seed': [0]
}

experiments_list = create_experiments_helper(
    experiment_name='not removing positives from uniform loss',
    project_name='xxx',
    script='python3 runner.py --config_file configs/generating.gin',
    python_path='',
    tags=[],
    exclude=[],
    base_config=base_config, params_grid=params_grid
)
