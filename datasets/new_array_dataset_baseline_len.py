import re
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

def get_dataset_stats(directory):
    num_traj = 0
    file_paths = []

    pattern = r'^dataset_part_\d+\.pkl$'
    
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            digit = int(filename.split('_')[-1].split('.')[0])
            
            len_filename = f'dataset_part_{digit}_lens.pkl'
            num_traj += 1 #len(arr)                
            # del arr
            file_paths.append((directory, filename, len_filename))
                

    file_paths = sorted(file_paths)
    return num_traj, file_paths

@gin.configurable
class GenDataset():
    def __init__(self, loggers, path, gamma, max_horizon, double_batch = 1, two_sided_sampling = False, length = 20, noise_prob = 0., min_ind=None, max_ind=None, max_size=None, sampling_probability='uniform', device='cpu', weights=None):
        self.gamma = gamma
        self.loggers = loggers
        self.max_horizon = max_horizon
        
        self.num_traj, self.file_paths = get_dataset_stats(path)
        self.path = path

        print(path)
        
        self.buffer = None
        self.buffer_ind = 0
        self.file_ind = 0

        self.min_ind = min_ind
        self.max_ind = max_ind
        
        self.device = device
        self.max_size = max_size
        self.sampling_probability = sampling_probability

        self.weights = weights
        self.two_sided_sampling = two_sided_sampling

        assert self.two_sided_sampling in [True, False]
        assert self.sampling_probability in ['uniform', 'equalized', 'exp', 'rev_exp', 'normal', 'custom']  
        assert self.sampling_probability != 'custom' or self.weights is not None
        assert device in ['cpu', 'cuda']

        
        self.double_batch = double_batch
        assert type(double_batch) == int

        self.NUM_GOAL = copy.deepcopy(NUM_GOAL).to(self.device)
        
        self.perms = copy.deepcopy(perms).to(self.device)
        
        self.reverse_actions = copy.deepcopy(reverse_actions).to(self.device)
        
    def _read_next_file(self):
        self.file_ind = np.random.randint(len(self.file_paths))
        
        current_dir, current_file, curren_len_file = self.file_paths[self.file_ind]
        current_path = current_dir + '/' + current_file
        current_len_path = current_dir + '/' + curren_len_file
        
        with open(current_path, 'rb') as f:
            self.buffer = joblib.load(f)

        with open(current_len_path, 'rb') as f:
            self.buffer_lens = joblib.load(f)
        
        self.buffer = self.buffer[self.buffer_lens != 0]
        self.buffer_lens = self.buffer_lens[self.buffer_lens != 0]
        
        self.buffer_ind = 0
        
        
    def _get_trajs(self, n_traj):
        if self.buffer is None or self.buffer_ind + n_traj > len(self.buffer):
            self._read_next_file()
            
        self.buffer_ind += n_traj        
        current_buffer, current_lens = self.buffer[self.buffer_ind - n_traj:self.buffer_ind], self.buffer_lens[self.buffer_ind - n_traj:self.buffer_ind]

        return current_buffer, current_lens
            
        
    def _get_batch(self, batch_size):
        if self.double_batch:
            assert batch_size % (self.double_batch) == 0
            trajs, lens = self._get_trajs(batch_size // self.double_batch)
            trajs = trajs.repeat(self.double_batch, 1, 1, 1)
            lens = lens.repeat(self.double_batch)
        else:
            trajs, lens = self._get_trajs(batch_size)
        
        if self.max_ind is not None:
            trajs = trajs[:, :self.max_ind]
        if self.min_ind is not None:
            trajs = trajs[:, self.min_ind:]
            
        assert len(trajs) > 0
        weights = torch.zeros((trajs.shape[0], trajs.shape[1]))
        mask = torch.arange(len(trajs[0])).unsqueeze(0).repeat(len(trajs), 1) < lens.unsqueeze(1).cpu()
        weights[mask] = 1

        i = torch.multinomial(weights.float(), num_samples=1).squeeze()
        j = torch.multinomial(weights.float(), num_samples=1).squeeze()
        sorted_i_j = torch.concat([i.unsqueeze(1), j.unsqueeze(1)], axis=1).sort(axis=1)[0]
        i = sorted_i_j[:, 0]
        j = sorted_i_j[:, 1]


        return torch.concat((trajs[torch.arange(len(trajs)), i].flatten(1, 2), trajs[torch.arange(len(trajs)), j].flatten(1, 2)), axis=1).to(torch.float32).to(self.device), (j - i).to(self.device) # , trajs[:, i+delta, :])
    
    def _get_trajectory(self, tokenize=False):
        traj, len = self._get_trajs(1)
        traj = traj.squeeze()
        traj = traj.flatten(1)
        len = len.item()

        traj = traj[:len]
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
    save_path = "noise_45_shuffle_20/"
    n_trajs = 5000000
    bs = 512
    
    dataset = GenDataset(None, save_path, 0.9, 200, noise_prob=0.45, length=150)
    dataloader = GenDataLoader(dataset, None, bs)
    
    for i, a in tqdm.tqdm(enumerate(dataloader)):
        import pdb; pdb.set_trace()
        print(a)


        

if __name__ == "__main__":
    main()