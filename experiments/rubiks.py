from mrunner.helpers.specification_helper import create_experiments_helper
import os

base_config = {
}


params_grid = {
    'idx': [0],
    'jobs.generating_job.TrainJob.train_steps': [500000000],
    'jobs.generating_job.TrainJob.batch_size': [512],
    'jobs.generating_job.TrainJob.chunk_size': [None],
    'jobs.generating_job.TrainJob.dataset_class': ["@new_array_dataset.GenDataset"],
    'jobs.generating_job.TrainJob.loss_fn': ["@losses.vanilla_loss"],
    'vanilla_loss.loss_type': ['backward'],
    'jobs.generating_job.TrainJob.metric': ['l22'],
    'jobs.generating_job.TrainJob.search_shuffles': [[10, 15, 20]],
    'jobs.generating_job.TrainJob.c': [10],
    'jobs.generating_job.TrainJob.lr': [3e-4],
    'jobs.generating_job.TrainJob.model_type': ["@LNDenseNet"],
    'new_array_dataset.GenDataset.gamma': [0.9],
    'new_array_dataset.GenDataset.double_batch': [1, 2],
    'jobs.generating_job.TrainJob.eval_job_class': ['@JobSolveRubik'],
    'new_array_dataset.GenDataset.path': ['rubik_dataset'],
    f'new_array_dataset.GenDataset.max_horizon': [200],
    'BestFSSolverRubik.value_estimator_class': ['@value_function.ValueEstimatorRubik'],
    'LNDenseNet.repr_dim': [64],
    'LNDenseNet.hidden_size': [1024],
    'LNDenseNet.depth': [8],
    'VanillaPolicyRubik.possible_actions': [12],
    'run.job_class': ['@jobs.generating_job.TrainJob'],
    'jobs.generating_job.TrainJob.solving_interval': [150000],
    'VanillaPolicyRubik.env': ['@make_RubikEnv'],
    'run.seed': [0],
    'JobSolveRubik.generate_problems': ['@generate_problems_rubik']
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
