
import sklearn.manifold
import gin
import matplotlib.pyplot as plt

import numpy as np

import matplotlib.pyplot as plt


from datasets.new_array_dataset import GenDataLoader, tokenize_pair
from search.value_function import ValueEstimatorRubik

import torch
import math
import functools
import sklearn
import sklearn.decomposition
from scipy.stats import spearmanr

from losses import mrn_loss

from pathlib import Path

def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier

def CosineAnnealingLRWarmup(optimizer, T_max, T_warmup):
    _decay_func = functools.partial(
        _cosine_decay_warmup, 
        warmup_iterations=T_warmup, total_iterations=T_max
    )
    scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler

@gin.configurable
class TrainJob():
    def __init__(
        self,
        loggers,
        train_steps, 
        batch_size, 
        dataset_class,
        lr,
        model_type,
        loss_fn,
        metric,
        search_shuffles,
        use_log_lambda=False,
        chunk_size=None,
        output_dir='result',
        n_test_traj=100, 
        c=10,
        include_actions=False,
        separate_goal_encoder=False,
        do_eval=True,
        solving_interval=None,
        tokenizer=tokenize_pair,
        eval_job_class=None,
        checkpoint_path=None,
        test_path=None
            ):
        self.loggers = loggers
        self.train_steps = train_steps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        self.model = model_type().to(self.device)
        self.solving_interval = solving_interval
        if separate_goal_encoder:
            model1 = model_type().to(self.device)
            model2 = model_type().to(self.device)
            self.models = [model1, model2]
        else:
            model = model_type().to(self.device)
        
        self.separate_goal_encoder = separate_goal_encoder
        
        self.c = c
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.include_actions = include_actions
        self.do_eval = do_eval
        self.eval_job_class = eval_job_class
        self.metric = metric
        self.use_log_lambda = use_log_lambda
        
        assert ~((self.metric == 'mrn') ^ (self.loss_fn == mrn_loss))
        
        self.c = c
        if not self.separate_goal_encoder:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(list(self.models[0].parameters()) + list(self.models[1].parameters()), lr=self.lr)
            
        
        if self.checkpoint_path is not None:
            self.read_checkpoint(self.checkpoint_path)

        assert not (eval_job_class is None) and do_eval, "need to specify eval job class if eval is to be performed"
        
        
        # train_indices, test_indices = perm[:int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]
        self.dataset = dataset_class(loggers=self.loggers, device=self.device)
    
        self.train_dataloader = GenDataLoader(self.dataset, batch_size=self.batch_size)


        if test_path is None:
            self.test_dataset = dataset_class(loggers=self.loggers, device=self.device)
        else:
            self.test_dataset = dataset_class(path=test_path, loggers=self.loggers, device=self.device)
        self.test_dataloader = GenDataLoader(self.test_dataset, batch_size=self.batch_size)

        if self.do_eval:
            self.test_trajectories = [self.test_dataset._get_trajectory() for _ in range(n_test_traj)]
        self.output_dir = output_dir

        self.search_shuffles = search_shuffles

    def save_checkpoint(self, step):
        if self.separate_goal_encoder:
            model_checkpoint_path1= f"{self.output_dir}/{step}/model1.pt"
            model_checkpoint_path2= f"{self.output_dir}/{step}/model2.pt"
            optimizer_checkpoint_path= f"{self.output_dir}/{step}/optimizer"
            path1 = Path(model_checkpoint_path1)
            path2 = Path(model_checkpoint_path2)
            path_opt = Path(optimizer_checkpoint_path)
            path1.parent.mkdir(parents=True, exist_ok=True)

            torch.save(self.model.state_dict(), path1)
            # torch.save(self.models[1].state_dict(), path2)
            torch.save(self.optimizer.state_dict(), path_opt)
        else:
            model_checkpoint_path= f"{self.output_dir}/{step}/model.pt"
            optimizer_checkpoint_path= f"{self.output_dir}/{step}/optimizer"
            path = Path(model_checkpoint_path)
            path_opt = Path(optimizer_checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), path)
            torch.save(self.optimizer.state_dict(), path_opt)
            
            
    def read_checkpoint(self, path):
        if self.separate_goal_encoder:
            raise NotImplementedError()
        else:
            model_checkpoint_path= f"{path}/model.pt"
            optimizer_checkpoint_path= f"{path}/optimizer"
            model_checkpoint = torch.load(model_checkpoint_path, weights_only=True, map_location=torch.device(self.device))
            optimizer_checkpoint = torch.load(optimizer_checkpoint_path, weights_only=True, map_location=torch.device(self.device))
            self.model.load_state_dict(model_checkpoint)
            self.optimizer.load_state_dict(optimizer_checkpoint)


    def gen_plot_distances(self):        
        value_estimator = ValueEstimatorRubik(self.model, self.metric)
        all_distances = []
        for i, s in enumerate(self.test_trajectories):
            import pdb; pdb.set_trace()
            distances = value_estimator.get_solved_distance_batch(s, s[-1])
            all_distances.append(distances.cpu().numpy())

        all_distances = np.array(all_distances).mean(axis=0) 
        plt.plot(np.arange(len(all_distances)), all_distances)
        self.loggers.log_figure(f'avg distances solved', 0, plt.gcf())
        plt.clf()
        
    def gen_conf_matrix_distances(self):
        for i in range(4):        
            trajectory = self.test_trajectories[i]
    
            with torch.no_grad():
                psi, _ = self.model(trajectory)
                
            if self.metric == 'mrn':
                psi = psi[..., psi.shape[-1] // 2:]
            
            pdist = torch.mean((psi[:, None] - psi[None])**2, axis=-1).cpu().numpy()
            plt.figure(figsize=(6, 6))
            plt.imshow(pdist, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            self.loggers.log_figure(f'distances {i}', 0, plt.gcf())
            plt.clf()
            
            
    def gen_plot_1(self):
        for traj in self.test_trajectories:
            with torch.no_grad():
                traj = traj#.to(self.device)
                psi, _ = self.model(traj)
                if self.metric == 'mrn':
                    psi = psi[..., psi.shape[-1] // 2:]
                psi = psi.cpu()
                traj = traj#.to('cpu')
                del traj

            tsne = sklearn.manifold.TSNE(n_components=2, perplexity=5)
            psi = tsne.fit_transform(psi)

            plt.scatter(psi[:, 0], psi[:, 1], marker='.', c=np.arange(len(psi)), cmap='Reds')

        plt.gca().set_aspect('equal')
        self.loggers.log_figure("All reps", 0, plt.gcf())
        plt.clf()

    def gen_plot_2(self):
        for i, s in enumerate(self.test_trajectories):
            if i == 4:
                break

            with torch.no_grad():
                s = s# .to(self.device)
                psi, _ = self.model(s)
                
                if self.metric == 'mrn':
                    psi = psi[..., psi.shape[-1] // 2:]
                    
                psi = psi.cpu()
                s = s# .to('cpu')
                del s
            

            tsne = sklearn.manifold.TSNE(n_components=2, perplexity=5)
            psi = tsne.fit_transform(psi)
            beginning = psi[0]
            end = psi[-1]

            c_vec = plt.rcParams['axes.prop_cycle'].by_key()['color']
            plt.text(psi[0, 0], psi[0, 1], '$x_0$', ha='center', va='bottom', fontsize=16)
            plt.text(psi[-1, 0], psi[-1, 1], '$x_T$', ha='center', va='bottom', fontsize=16)

            plt.plot(psi[:, 0], psi[:, 1], '-', c=c_vec[0], linewidth=1, alpha=0.1)
            plt.scatter(psi[:, 0], psi[:, 1], c=np.arange(len(psi)), cmap='plasma')

            n_wypt = 5

            vec = np.linspace(beginning, end, n_wypt)
            plt.scatter(vec[:, 0], vec[:, 1], c=np.arange(len(vec)), cmap='Greys')

            plt.gca().set_aspect('equal')
            self.loggers.log_figure(f'plot {i}', 0, plt.gcf())
            plt.clf()

    def gen_plot_3(self, c=1):
        n_wypt = 5

        for i, s in enumerate(self.test_trajectories):
            if i == 4:
                break

            with torch.no_grad():
                s = s#.to(self.device)
                psi, _ = self.model(s)
                if self.metric == 'mrn':
                    psi = psi[..., psi.shape[-1] // 2:]
                    
                psi = psi.cpu().numpy()
                s = s#.to('cpu')
                del s

            A = self.model.A.cpu().detach().numpy()

            tsne = sklearn.manifold.TSNE(n_components=2, perplexity=5)

            n = self.model.repr_dim
            I = np.eye(n)
            
            M = np.zeros((n_wypt * n, n_wypt * n))
            for i in range(n_wypt):
                print(((c + 1) / c * I).shape) 
                print((c / (c + 1) * A.T @ A).shape )
                M[n*i:n*(i+1), n*i:n*(i+1)] = (c + 1) / c * I + c / (c + 1) * A.T @ A
                if i + 1 < n_wypt:
                    M[n*i:n*(i+1), n*(i+1):n*(i+2)] = -A.T
                    M[n*(i+1):n*(i+2), n*i:n*(i+1)] = -A

            eta = np.block([A @ psi[0], np.zeros((n_wypt - 2)*n), A.T @ psi[-1]])
            w_vec = np.linalg.solve(M, eta).reshape((-1, n))
            vec = np.concatenate([[psi[0]], w_vec, [psi[-1]]], axis=0)

            print(psi.shape, vec.shape, np.concatenate([vec, psi]).shape)
            tsne_res = tsne.fit_transform(np.concatenate([vec, psi]))
            print(tsne_res.shape)

            vec = tsne_res[:len(vec)]
            psi = tsne_res[len(vec):]
            # psi = tsne.fit_transform(psi)
            beginning = psi[0]
            end = psi[-1]

            c_vec = plt.rcParams['axes.prop_cycle'].by_key()['color']
            plt.text(psi[0, 0], psi[0, 1], '$x_0$', ha='center', va='bottom', fontsize=16)
            plt.text(psi[-1, 0], psi[-1, 1], '$x_T$', ha='center', va='bottom', fontsize=16)

            plt.plot(psi[:, 0], psi[:, 1], '-', c=c_vec[0], linewidth=1, alpha=0.1)
            plt.scatter(psi[:, 0], psi[:, 1], c=np.arange(len(psi)), cmap='plasma')


            plt.scatter(vec[:, 0], vec[:, 1], c=np.arange(len(vec)), cmap='Reds')
            plt.gca().set_aspect('equal')
            self.loggers.log_figure(f'plot waypoints {i}', 0, plt.gcf())
            plt.clf()

    def gen_plot_monotonicity(self):
        value_estimator = ValueEstimatorRubik(self.model, self.metric)
        correlations = []
        for i, s in enumerate(self.test_trajectories):
            distances = value_estimator.get_solved_distance_batch(s, s[-1]) #.to(self.device)).to('cpu')
            s = s# .to('cpu')
            del s
            correlation = spearmanr(distances.cpu(), np.arange(len(distances.cpu()))).statistic
            correlations.append(correlation)
            if i < 4:

                self.loggers.log_scalar(f'correlation {i}', 0, correlation)


                plt.plot(np.arange(distances.cpu().shape[-1]), distances.cpu())

                self.loggers.log_figure(f'monotonicity {i}', 0, plt.gcf())
                plt.clf()
        
        self.loggers.log_scalar(f'correlation', 0, sum(correlations)/len(correlations))


    def execute(self): 
        c = self.c

        if self.chunk_size is not None:
            self.grad_cache = GradCache(
                models=[self.model, self.model],
                chunk_sizes=self.chunk_size,
                loss_fn=self.loss_fn,
            )
        else:
            self.grad_cache = None

        seen = 0
        while seen < self.train_steps:
            for t, data in enumerate(self.train_dataloader):
                self.model.train()
                    
                data = data#.to(self.device)
                self.optimizer.zero_grad()
                x0 = data[:, 0]
                xT = data[:, 1]
                
                if self.grad_cache is not None:
                    loss = self.grad_cache(x0, xT)
                else:
                    psi_0 = self.model(x0)
                    psi_T = self.model(xT)
                    if self.use_log_lambda:
                        loss, self.metrics = self.loss_fn(psi_0, psi_T, distance_fun=self.metric, log_lambda=self.model.log_lambda, c=self.c)
                    else:
                        loss, self.metrics = self.loss_fn(psi_0, psi_T, distance_fun=self.metric)

                    loss.backward()

                self.optimizer.step()


                if (seen // self.batch_size) % 10 == 0:
                    for name, value in self.metrics.items():
                        print(name, t, value)
                        self.loggers.log_scalar(name, t, value)
                        
                    self.loggers.log_scalar('step', t, t)
                    
                    for test_data in self.test_dataloader:
                            with torch.no_grad():
                                test_data = test_data# .to(self.device)
                                x0 = test_data[:, 0]
                                xT = test_data[:, 1]
                                psi_0 = self.model(x0)
                                psi_T = self.model(xT)
                                loss, self.metrics = self.loss_fn(psi_0, psi_T, distance_fun=self.metric)

                            for name, value in self.metrics.items():
                                self.loggers.log_scalar("test_" + name, t, value)

                            break

                if seen % (self.batch_size * 10000) == 0:
                    with torch.no_grad():
                        if self.do_eval:
                            self.gen_plot_monotonicity()
                            self.gen_plot_1()
                            self.gen_plot_2()
                            self.gen_plot_3()
                            self.gen_conf_matrix_distances()
# 
                            if seen % (self.batch_size * self.solving_interval) == 0:
                                for shuffles in self.search_shuffles:
                                    eval_job = self.eval_job_class(loggers=self.loggers, network=self.model.cpu(), metric=self.metric, shuffles=shuffles)
                                    eval_job.execute()
                                    
                                self.save_checkpoint(seen)
                                    
                            self.model.to(self.device)

                
                seen += len(data)
                del data

        if self.do_eval:
            self.gen_plot_1()
            self.gen_plot_2()
            self.gen_plot_3()
        self.save_checkpoint('final')
