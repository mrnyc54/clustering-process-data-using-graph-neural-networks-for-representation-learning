import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool as gap

class Net(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = GraphConv(num_features, 16)
        self.conv2 = GraphConv(16, 16)
        self.lin1 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = gap(x, data.batch)
        x = self.lin1(x)
        return x.squeeze(1)