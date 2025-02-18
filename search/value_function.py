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
        num_state = torch.tensor(state_str)
        num_goal = torch.tensor(goal)
        self.model.eval()
        
        with torch.no_grad():
            goal_repr, _ = self.model(num_goal.unsqueeze(0))
            _, state_repr = self.model(num_state.unsqueeze(0))

        distance = self.distance(state_repr, goal_repr)
        
        return distance
    
    def get_solved_distance_batch(self, states, goal):
        num_goal = torch.tensor(goal)
        self.model.eval()
        with torch.no_grad():
            goal_repr, _ = self.model(num_goal.unsqueeze(0).to(states.device))
            _, state_repr = self.model(states)

        distance = self.distance(state_repr, goal_repr)

        return distance.squeeze()
    
 
