import torch
import torch.nn as nn
import torch.nn.functional as F
from GroupedLinear import GroupedLinear

class WeavedMLP(nn.Module):
    def __init__(self, dim_in, dim_out,hidden_dims, n_groups=2, dropout=0.):
        super(GroupedMLP, self).__init__()
        self.depth = len(hidden_dims)-1
        self.n_groups = n_groups

        self.layers = nn.ModuleList([])
        for i in range(self.depth):
            self.layers.append(
                GroupedLinear(hidden_dims[i], 
                              hidden_dims[i+1], 
                              n_groups=n_groups)
                )
        self.outlayer = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
   def forward(self, x):
        b, n = x.shape
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

            # Verweben
            x = x.reshape((b,n/self.groups, self.groups))
            x = x.transpose(-1,-2)
            x = x.flatten(start_dim=-2, end_dim=-1)
        return self.activation(self.out(x))
