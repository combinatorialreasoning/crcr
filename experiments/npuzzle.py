from mrunner.helpers.specification_helper import create_experiments_helper
import os

base_config = {
}

params_grid = {
    'idx': [0, 1, 2],
    'jobs.generating_job.TrainJob.train_steps': [500000000],
    'jobs.generating_job.TrainJob.batch_size': [512],
    'jobs.generating_job.TrainJob.chunk_size': [None],
    'jobs.generating_job.TrainJob.dataset_class': ["@new_array_dataset.GenDataset"],
    'jobs.generating_job.TrainJob.loss_fn': ["@vanilla_loss"],
    'jobs.generating_job.TrainJob.use_log_lambda': [False],
    'jobs.generating_job.TrainJob.metric': ['l22'],
    'jobs.generating_job.TrainJob.search_shuffles': [[-1]],
    'npuzzle.env.NPuzzle.n': [4],
    'jobs.generating_job.TrainJob.c': [10],
    'jobs.generating_job.TrainJob.lr': [3e-4],
    'jobs.generating_job.TrainJob.solving_interval': [150000],
    'jobs.generating_job.TrainJob.model_type': ["@LNDenseNet"],
    'new_array_dataset.GenDataset.gamma': [0.9],
    'new_array_dataset.GenDataset.path': ['npuzzle_data'],
    f'new_array_dataset.GenDataset.max_horizon': [200],
    'new_array_dataset.GenDataset.sampling_probability___': ['uniform'],
    'new_array_dataset.GenDataset.double_batch': [1, 2],
    'losses.vanilla_loss.loss_type': ['backward'],
    'JobSolveRubik.generate_problems': ['@npuzzle.data_generate.generate_problems'],
    'VanillaPolicyRubik.env': ['@npuzzle.env.NPuzzle'],

    'LNDenseNet.input_size': [16],
    'LNDenseNet.repr_dim': [64],
    'LNDenseNet.hidden_size': [1024],
    'LNDenseNet.depth': [8],
    'VanillaPolicyRubik.possible_actions': [12],
    'run.job_class': ['@jobs.generating_job.TrainJob'],
    'BestFSSolverRubik.value_estimator_class': ['@value_function.ValueEstimatorRubik'],
    'jobs.generating_job.TrainJob.eval_job_class': ['@JobSolveRubik'],
    'run.seed': [0, 1, 2],
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
