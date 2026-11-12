import random
import torch
from torch.distributions import Normal

# Computer the reward and the delta
def compute_reward_delta( curr_value, next_value, reward, termination, gamma = 0.99 ):
    target = reward + gamma * next_value * ( 1 - termination )
    delta = target - curr_value
    return delta