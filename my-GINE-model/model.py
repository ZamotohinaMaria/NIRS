import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool


class GINEMalwareClassifier(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.node_encoder = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_add_pool(x, batch)
        return self.head(x)
