import torch
# import flax
# import jax.numpy as jnp
from datasets.dataset import tokenize
# from jax import random
import copy

from losses import get_mrn_dist

GOAL = 'yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww'
import gin

@gin.configurable
class ValueEstimatorRubik:
    def __init__(self, model, metric, include_actions=False, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.distance = (lambda x, y: (((x-y)**2).sum(axis=1))**(0.5)) if metric == 'l22' else (lambda x, y: ((x-y)**2).sum(axis=1)) if metric == 'l2' else (lambda x, y: (torch.abs((x-y))).sum(axis=1)) if metric == 'l1' else (lambda x, y: -(torch.matmul(x, y.T))) if metric == 'dot' else None # torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.include_actions = include_actions
        
        if metric == 'mrn':
            self.distance = get_mrn_dist
                
        if self.distance is None:
            raise ValueError()
        
    def construct_networks(self):
        if self.checkpoint_path is None:
            return
        
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=torch.device('cpu')))
    
    def get_solved_distance(self, state_str, goal, action_in=None):
        num_state = torch.tensor(state_str).unsqueeze(0)
        num_goal = torch.tensor(goal).unsqueeze(0).to(num_state.device)

        net_input = torch.concatenate([num_state, num_goal], axis=1)
        self.model.eval()

        with torch.no_grad():
            distance, _ = self.model(net_input)

        
        return distance.argmax(axis=1).item()
    
    def get_solved_distance_batch(self, states, goal):
        num_goal = torch.tensor(goal).unsqueeze(0).repeat(len(states), 1).to(states.device)
        net_input = torch.concatenate([states, num_goal], axis=1)
        self.model.eval()
        with torch.no_grad():
            distances, _ = self.model(net_input)


        return distances.argmax(axis=1).squeeze()

    





