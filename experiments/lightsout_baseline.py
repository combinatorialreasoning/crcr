from mrunner.helpers.specification_helper import create_experiments_helper
import os

base_config = {
}

params_grid = {
    'idx': [0,1],
    'jobs.generating_job_baseline.TrainJob.train_steps': [500000000],
    'jobs.generating_job_baseline.TrainJob.batch_size': [512],
    'jobs.generating_job_baseline.TrainJob.chunk_size': [None],
    'jobs.generating_job_baseline.TrainJob.dataset_class': ["@new_array_dataset_baseline.GenDataset"],
    'jobs.generating_job_baseline.TrainJob.loss_fn': ["@vanilla_loss"],
    'jobs.generating_job_baseline.TrainJob.use_log_lambda': [False],
    'jobs.generating_job_baseline.TrainJob.metric': ['l22'],
    'jobs.generating_job_baseline.TrainJob.search_shuffles': [[-1]],
    'lights_out.env.LightsOut.n': [7],
    'jobs.generating_job_baseline.TrainJob.c': [10],
    'jobs.generating_job_baseline.TrainJob.lr': [3e-4],
    'jobs.generating_job_baseline.TrainJob.model_type': ["@LNDenseNet"],
    'new_array_dataset_baseline.GenDataset.gamma': [0.9],
    'new_array_dataset_baseline.GenDataset.path': ['lights_out_data'],
    f'new_array_dataset_baseline.GenDataset.max_horizon': [200],
    'new_array_dataset_baseline.GenDataset.sampling_probability___': ['uniform'],
    'JobSolveRubik.n_actions': [10],
    'JobSolveRubik.generate_problems': ['@lights_out.data_generate.generate_problems'],
    'VanillaPolicyRubik.env': ['@lights_out.env.LightsOut'],

    'LNDenseNet.input_size': [49 * 2],
    'LNDenseNet.repr_dim': [49],
    'LNDenseNet.hidden_size': [1024],
    'LNDenseNet.depth': [8],
    'VanillaPolicyRubik.possible_actions': [12],
    'run.seed': [0, 1, 2],
    'BestFSSolverRubik.value_estimator_class': ['@search.value_function_baseline.ValueEstimatorRubik'],
    'run.job_class': ['@jobs.generating_job_baseline.TrainJob'],
    'jobs.generating_job_baseline.TrainJob.eval_job_class': ['@JobSolveRubik'],
    'jobs.generating_job_baseline.TrainJob.solving_interval': [150000]
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
