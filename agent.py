import torch
import torch.nn as nn
from torch.distributions import Normal

# Actor Network 
class Actor ( nn.Module ):
    def __init__( self, i_size, h_size, o_size ):
        super().__init__()
        self.actor = nn.Sequential( nn.Linear(i_size, h_size), nn.ReLU(), 
                                   nn.Linear(h_size, h_size), nn.ReLU(), nn.Linear(h_size,o_size) )

        self.log_std = nn.Parameter(torch.zeros(o_size))

    def forward( self, x ):
        mu = self.actor(x)
        std = self.log_std.exp()
        return Normal(mu, std)

# Critic Network
class Critic ( nn.Module ):
    def __init__( self, i_size, h_size ):
        super().__init__()
        self.critic = nn.Sequential( nn.Linear(i_size, h_size), nn.ReLU(), 
                                     nn.Linear(h_size, h_size), nn.ReLU(), nn.Linear(h_size, 1) )
        
    def forward(self, x ):
        return self.critic(x)