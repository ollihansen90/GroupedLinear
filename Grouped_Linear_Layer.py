class GroupedLinear(nn.Module):
    def __init__(self, dim_in, dim_out, n_groups=2):
        super(GroupedLinear, self).__init__()
        self.layer = nn.Conv1d(dim_in, dim_out, 1, groups=n_groups)
    
    def forward(self, x):
        out = x.unsqueeze(-1)
        out = self.layer(out)
        out.squeeze()
        return out
