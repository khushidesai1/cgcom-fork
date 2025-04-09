
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
import torch

class GAT(torch.nn.Module):
    def __init__(self,inputnodefeatures, hidden_channels,num_classes):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(inputnodefeatures, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # 1. Obtain node embeddings 
        []
        x,attentionscore = self.conv1(x, edge_index,return_attention_weights=True)
        attentionscores=  attentionscore[1].tolist()
        x = x.relu()
        x,attentionscore = self.conv2(x, edge_index,return_attention_weights=True)
        for i in range(len(attentionscore[1].tolist())):
            attentionscores[i].append(attentionscore[1].tolist()[i][0])
        x = x.relu()
        x,attentionscore = self.conv3(x, edge_index,return_attention_weights=True)
        for i in range(len(attentionscore[1].tolist())):
            attentionscores[i].append(attentionscore[1].tolist()[i][0])
        averageattention = []
        for values in attentionscores:
            averageattention.append(sum(values)/len(values))

        # 2. Readout layer
        x = global_mean_pool(x,None)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x,averageattention,attentionscore[0]








