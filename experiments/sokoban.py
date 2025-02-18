from mrunner.helpers.specification_helper import create_experiments_helper
import os

base_config = {
}

params_grid = {
    'idx': [0, 1, 2],
    'jobs.generating_job.TrainJob.train_steps': [500000000],
    'jobs.generating_job.TrainJob.batch_size': [500],
    'jobs.generating_job.TrainJob.chunk_size': [None],
    'jobs.generating_job.TrainJob.dataset_class': ["@new_array_dataset_len.GenDataset"],
    'jobs.generating_job.TrainJob.loss_fn': ["@vanilla_loss"],
    'jobs.generating_job.TrainJob.metric': ['l2'],

    'jobs.generating_job.TrainJob.search_shuffles': [[6]],
    'jobs.generating_job.TrainJob.test_path': ['sokoban_test'], 
    'jobs.generating_job.TrainJob.lr': [3e-4],
    'jobs.generating_job.TrainJob.model_type': ["@LNConvNet"],
    'new_array_dataset_len.GenDataset.gamma': [0.9],
    'new_array_dataset_len.GenDataset.double_batch': [1, 2],
    # 'new_array_dataset_len.GenDataset.i_weights': ['none', 'front', 'back'],
    'new_array_dataset_len.GenDataset.path': ['sokoban_train'],
    f'new_array_dataset_len.GenDataset.max_horizon': [200],
    'new_array_dataset_len.GenDataset.sampling_probability': ['uniform'],
    'losses.vanilla_loss.loss_type': ['symmetric'],
    'losses.vanilla_loss.normalize': [False],
    'losses.vanilla_loss.exclude_diagonal': [True],
    # 'losses.vanilla_loss.logsumexp_penalty': [0.01],
    
    'LNConvNet.input_size': [7],
    'LNConvNet.repr_dim': [64],
    'LNConvNet.hidden_size': [64],
    'LNConvNet.depth': [8],
    'VanillaPolicyRubik.possible_actions': [12],

    'BestFSSolverRubik.value_estimator_class': ['@value_function.ValueEstimatorRubik'],


    'run.job_class':  ['@jobs.generating_job.TrainJob'],
    'jobs.generating_job.TrainJob.eval_job_class': ['@JobSolveRubik'],
    'jobs.generating_job.TrainJob.solving_interval': [50000],
    'run.seed': [0, 1, 2],

    'CustomSokobanGenerator.boards_path': ["boards_sokoban_12x12x4_0.joblib"], 
    'JobSolveRubik.generate_problems': ['@search.gen_problems_sokoban.generate_problems_sokoban'],
    'VanillaPolicyRubik.env': ['@sokoban_env.CustomSokobanEnv'],
    'sokoban_env.CustomSokobanEnv.generator': ['@sokoban_env.CustomSokobanGenerator'],
    'sokoban_env.CustomSokobanEnv.grid_size': [12],

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
