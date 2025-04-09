import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool
class QKVGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(QKVGATConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W_query = nn.Linear(in_channels, out_channels)

        # Initialize W_key with trainable and non-trainable parts
        self.W_key = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.W_value = nn.Linear(in_channels, out_channels)

        # Create a mask (1s are trainable, 0s are not)
        self.mask = torch.Tensor(in_channels, out_channels).fill_(1)
        # Example: Making the first column non-trainable
        self.mask[:, 0] = 0  

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_query.weight)
        nn.init.xavier_uniform_(self.W_value.weight)

        # Initialize W_key weights
        nn.init.xavier_uniform_(self.W_key)
        # Apply mask to make certain weights non-trainable
        self.W_key.data *= self.mask

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        q = self.W_query(x)
        # Apply mask on W_key during forward pass
        k = F.linear(x, self.W_key * self.mask)
        v = self.W_value(x)

        # Start propagating messages
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, q=q, k=k, v=v)

    def message(self, edge_index_i, q_i, k_j, v_j, size_i):
        # Compute attention coefficients.
        key_dim_sqrt = torch.sqrt(torch.tensor(self.out_channels, dtype=torch.float))
        attention_coefficients = (q_i * k_j).sum(dim=-1) / key_dim_sqrt

        # Reshape attention coefficients to [num_edges, num_heads]
        # Assuming a single attention head for this example
        attention_coefficients = attention_coefficients.view(-1, 1)

        # Apply softmax to the reshaped attention coefficients
        attention_coefficients = F.softmax(attention_coefficients, dim=0)

        # Multiply the values by the attention coefficients.
        return v_j * attention_coefficients.view(-1, 1)

class GATGraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(GATGraphClassifier, self).__init__()
        self.gat_conv = QKVGATConv(in_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # Apply the GAT layer
        x = self.gat_conv(x, edge_index)
        x = F.relu(x)

        # Aggregate node embeddings for each graph
        x = global_mean_pool(x, batch)  # Alternatively, use global_max_pool or global_add_pool

        # Apply the fully connected layer for classification
        x = self.fc(x)
        return x


# Example usage
num_nodes = 10
num_edges = 15
in_channels = 5
hidden_channels = 10
num_classes = 3

# Create random node features, edge indices, and batch indices for demonstration
x = torch.rand((num_nodes, in_channels))
edge_index = torch.randint(0, num_nodes, (2, num_edges))
batch = torch.randint(0, 2, (num_nodes,))  # Assuming 2 graphs in the batch

# Create a GAT graph classifier model
model = GATGraphClassifier(in_channels, hidden_channels, hidden_channels, num_classes)
out = model(x, edge_index, batch)

print(out)