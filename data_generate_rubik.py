import gin
import joblib
import os
import numpy as np
import torch
import random
import copy
import time
import cloudpickle
import tqdm

def get_mapping():    
    return  {'o':0, 'y':1, 'r':2, 'b':3, 'g':4, 'w':5}

GOAL = 'yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww'

NUM_GOAL = []
for char in GOAL:
    NUM_GOAL.append(get_mapping()[char])
    
NUM_GOAL = torch.tensor(NUM_GOAL)

reverse_actions = torch.tensor([3, 6, 7, 0, 11, 9, 1, 2, 10, 5, 8, 4], dtype=int)

perms = torch.from_numpy(np.array([[[15, 16, 17, 42, 43, 44, 33, 34, 35, 24, 25, 26, 45, 46, 47, 48, 49, 50, 51, 52, 53], [42, 43, 44, 33, 34, 35, 24, 25, 26, 15, 16, 17, 51, 48, 45, 52, 49, 46, 53, 50, 47]],
    [[9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38, 0, 1, 2, 3, 4, 5, 6, 7, 8] , [18, 19, 20, 27, 28, 29, 36, 37, 38, 9, 10, 11, 6, 3, 0, 7, 4, 1, 8, 5, 2]],
    [[0, 3, 6, 38, 41, 44, 45, 48, 51, 18, 21, 24, 9, 10, 11, 12, 13, 14, 15, 16, 17] , [44, 41, 38, 51, 48, 45, 18, 21, 24, 0, 3, 6, 15, 12, 9, 16, 13, 10, 17, 14, 11]],
    [[15, 16, 17, 24, 25, 26, 33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53] , [24, 25, 26, 33, 34, 35, 42, 43, 44, 15, 16, 17, 47, 50, 53, 46, 49, 52, 45, 48, 51]],
    [[0, 1, 2, 29, 32, 35, 51, 52, 53, 9, 12, 15, 36, 37, 38, 39, 40, 41, 42, 43, 44] , [29, 32, 35, 53, 52, 51, 9, 12, 15, 2, 1, 0, 42, 39, 36, 43, 40, 37, 44, 41, 38]],
    [[2, 5, 8, 20, 23, 26, 47, 50, 53, 36, 39, 42, 27, 28, 29, 30, 31, 32, 33, 34, 35] , [20, 23, 26, 47, 50, 53, 42, 39, 36, 8, 5, 2, 33, 30, 27, 34, 31, 28, 35, 32, 29]],
    [[9, 10, 11, 36, 37, 38, 27, 28, 29, 18, 19, 20] + list(range(9)) , [36, 37, 38, 27, 28, 29, 18, 19, 20, 9, 10, 11, 2, 5, 8, 1, 4, 7, 0, 3, 6]],
    [[0, 3, 6, 18, 21, 24, 45, 48, 51, 38, 41, 44] + list(range(9, 18)), [18, 21, 24, 45, 48, 51, 44, 41, 38, 6, 3, 0, 11, 14, 17, 10, 13, 16, 9, 12, 15]],
    [[6, 7, 8, 27, 30, 33, 45, 46, 47, 11, 14, 17] + list(range(18, 27)) , [27, 30, 33, 47, 46, 45, 11, 14, 17, 8, 7, 6, 20, 23, 26, 19, 22, 25, 18, 21, 24]],
    [[2, 5, 8, 36, 39, 42, 47, 50, 53, 20, 23, 26] + list(range(27, 36)), [42, 39, 36, 53, 50, 47, 20, 23, 26, 2, 5, 8, 29, 32, 35, 28, 31, 34, 27, 30, 33]],
    [[6, 7, 8, 11, 14, 17, 45, 46, 47, 27, 30, 33] + list(range(18, 27)) , [17, 14, 11, 45, 46, 47, 33, 30, 27, 6, 7, 8, 24, 21, 18, 25, 22, 19, 26, 23, 20]],
    [[0, 1, 2, 9, 12, 15, 51, 52, 53, 29, 32, 35] + list(range(36, 45)), [15, 12, 9, 51, 52, 53, 35, 32, 29, 0, 1, 2, 38, 41, 44, 37, 40, 43, 36, 39, 42]],
    [list(range(21)), list(range(21))]])) # last is an identity transform

    
def tokenize_pair(batch_text):
    batch_num = []
    letter_to_number = get_mapping()
    for elt in batch_text:
        batch_num.append(([float(letter_to_number[char]) for char in elt[0]], [float(letter_to_number[char]) for char in elt[1]]))

    return batch_num

def tokenize_traj(traj_text):
    traj_num = []
    letter_to_number = get_mapping()
    for elt in traj_text:
        traj_num.append([letter_to_number[char] for char in elt])
    return traj_num

def get_dataset_stats(directories):
    num_traj = 0
    file_paths = []
    assert all(map(lambda x: os.path.isdir(x), directories)) or not any(map(lambda x: os.path.isdir(x), directories))
    for directory in directories:
        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.pkl'):
                    num_traj += 1
                    file_paths.append((directory, filename))
        
        else:
            arr = joblib.load(directory)
            num_traj += len(arr)

    file_paths = sorted(file_paths)
    return num_traj, file_paths

@gin.configurable
class GenDataset():
    def __init__(self, loggers, gamma, max_horizon, length = 20, noise_prob = 0., min_ind=None, max_ind=None, max_size=None, sampling_probability='uniform', device='cpu'):
        # super(Dataset).__init__()

        self.gamma = gamma
        self.loggers = loggers
        self.max_horizon = max_horizon

        self.min_ind = min_ind
        self.max_ind = max_ind
        
        self.device = device
        self.max_size = max_size
        self.sampling_probability = sampling_probability
        self.length = length
        self.noise_prob = noise_prob
        
        self.NUM_GOAL = copy.deepcopy(NUM_GOAL).to(self.device)
        
        self.perms = copy.deepcopy(perms).to(self.device)
        
        self.reverse_actions = copy.deepcopy(reverse_actions).to(self.device)
        
        
    def _get_trajs(self, n_traj):
        new_data = torch.zeros((n_traj, self.length, 54), device=self.device)
        # all_actions = torch.zeros((n_traj, 22), device=self.device, dtype=int)

        current_data = self.NUM_GOAL.unsqueeze(0).repeat(n_traj, 1)
        prev_actions_ind = torch.full((n_traj,), -1, device=self.device, dtype=int)
        
        for i in range(self.length):
            # import pdb; pdb.set_trace()
            new_data[:, self.length - i - 1] = current_data
            actions = torch.randint(high=len(perms) - 1, size=(n_traj,), device=self.device)
            mask = torch.zeros(n_traj, dtype=bool)
            mask3 = torch.ones(n_traj, dtype=bool)

            if not (prev_actions_ind == -1).all() and self.noise_prob is not None:
                probabilities = torch.full((n_traj,), self.noise_prob, device=self.device)
                mask = torch.bernoulli(probabilities).to(bool) # 1 with probability probabilities
                mask2 = prev_actions_ind != -1
                mask3 = prev_actions_ind < 21 # Don't reverse actions over 20
                
                current_data[torch.logical_not(mask3)] *= 0
                # print(prev_actions_ind)
                assert (prev_actions_ind >= -1).all()
                
                mask = torch.logical_and(mask, mask2)
                mask = torch.logical_and(mask, mask3)
                
                tmp_prev_actions_ind = prev_actions_ind
                tmp_prev_actions_ind[torch.logical_not(mask2)] += 1

                prev_actions = torch.gather(all_actions, 1, tmp_prev_actions_ind.unsqueeze(1)).squeeze(1)
                reverse_prev_actions = self.reverse_actions[prev_actions]
                actions[mask] = reverse_prev_actions[mask]
                actions[torch.logical_not(mask3)] -= 1
                
                prev_actions_ind[mask] -= 1

            to_apply = self.perms[actions]
            permuted_values = torch.gather(current_data, 1, to_apply[:, 1])
            current_data.scatter_(1, to_apply[:, 0], permuted_values)
            
            prev_actions_ind[torch.logical_and(torch.logical_not(mask), mask3)] += 1
            #if not mask.all():
            #    all_actions[torch.logical_and(torch.logical_not(mask), mask3), prev_actions_ind[torch.logical_and(torch.logical_not(mask), mask3)]] = \
            #            actions[torch.logical_and(torch.logical_not(mask), mask3)]


        return new_data
            
            
        
    def _get_batch(self, batch_size):
        trajs = self._get_trajs(batch_size)
        if self.max_ind is not None:
            trajs = trajs[:, :self.max_ind]
        if self.min_ind is not None:
            trajs = trajs[:, self.min_ind:]
        
        if self.sampling_probability == 'uniform':
        # Uniformly sample an index from each trajectory
            i = torch.randint(high=len(trajs[0]) - 1, size=(len(trajs),))
            
        elif self.sampling_probability == 'exp':
            # Exponential sampling - more weight towards earlier timesteps
            weights = torch.exp(-torch.arange(len(trajs[0]) - 1, dtype=torch.float32))
            weights /= weights.sum()  # Normalize weights
            i = torch.multinomial(weights, num_samples=len(trajs), replacement=True)
            
        elif self.sampling_probability == 'rev_exp':
            # Reverse exponential sampling - more weight towards later timesteps
            weights = torch.exp(torch.arange(len(trajs[0]) - 1, dtype=torch.float32))
            weights /= weights.sum()  # Normalize weights
            i = torch.multinomial(weights, num_samples=len(trajs), replacement=True)
            
        elif self.sampling_probability == 'normal':
            # Gaussian (normal) sampling around the middle of the sequence
            mean = (len(trajs[0]) - 1) / 2
            std = (len(trajs[0]) - 1) / 6  # Assuming 99.7% of samples lie within ±3σ
            normal_dist = torch.normal(mean=mean, std=std, size=(len(trajs),))
            i = torch.clamp(normal_dist.round(), min=0, max=len(trajs[0]) - 2).long()
            
        else:
            raise ValueError(f"Unknown sampling probability: {self.sampling_probability}")


        horizon = len(trajs[0]) - i - 1

        # Calculate probabilities and mask
        probs = self.gamma ** torch.arange(self.max_horizon).unsqueeze(0).repeat(len(trajs), 1).float()

        mask = torch.arange(self.max_horizon).repeat(len(trajs), 1) <= horizon.unsqueeze(1)
        probs *= mask.float()
        probs /= probs.sum(dim=1, keepdim=True)

        delta = torch.multinomial(probs, num_samples=1).squeeze()

        # # print(i)
        # # print(delta)
        # a = torch.concat((torch.arange(len(trajs)).unsqueeze(1), i.unsqueeze(1)), axis=1)
        # # print(a.shape)
        # # print(a)

        return torch.concat((trajs[torch.arange(len(trajs)), i].unsqueeze(1), trajs[torch.arange(len(trajs)), i+delta].unsqueeze(1)), axis=1) # , trajs[:, i+delta, :])
    
    def _get_trajectory(self, tokenize=False):
        traj = self._get_trajs(1).squeeze()
        
        if self.max_ind is not None:
            traj = traj[:self.max_ind]
        if self.min_ind is not None:
            traj = traj[self.min_ind:]

        if tokenize:
            return torch.from_numpy(np.array(tokenize_traj(traj), dtype=np.float32))
        else:
            return traj



@gin.configurable  
class GenDataLoader():
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return self


    def __next__(self):
        return self.dataset._get_batch(self.batch_size)



def main():
    length=20
    save_path = "rubik_dataset"
    os.mkdir(save_path)
    n_trajs = 500000000
    bs = 2048
    
    dataset = GenDataset(None, 0.9, 200, noise_prob=None, length=length)
    dataloader = GenDataLoader(dataset, bs)
    
    done = 0
    print("n iter", int(n_trajs/bs))
    for i, a in tqdm.tqdm(enumerate(dataloader)):
        trajs = dataset._get_trajs(bs)
        with open(save_path + f"part_{i}.pkl", 'wb') as f:
            cloudpickle.dump(trajs, f)
        done += len(trajs)
        del trajs
        
        if done > n_trajs:
            break


        

if __name__ == "__main__":
    main()

