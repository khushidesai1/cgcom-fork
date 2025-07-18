import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomGATConv(nn.Module):
    def __init__(self, p1_channels, p2_channels, mask_indexes,device):
        super(CustomGATConv, self).__init__()
        self.p1_channels = p1_channels
        self.p2_channels = p2_channels
        self.device = device
        # Trainable weight matrices
        self.W_query = nn.Linear(p2_channels, p2_channels)
        self.W_value = nn.Linear(p2_channels, p2_channels)
        # Initialize W_key with trainable and non-trainable parts
         # Initialization code
        self.W_key = nn.Parameter(torch.Tensor(p1_channels,p2_channels)).to(device)  # Correct shape
        self.mask =  torch.Tensor(p2_channels,p1_channels).fill_(0).to(device) # Correct shape  # Correct shape
        for mask_index in mask_indexes:
            self.mask[mask_index[0], mask_index[1]] = 1
        
        print(self.W_key.shape)
        print(self.mask.shape)

        self._init_weights()


    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_query.weight)
        nn.init.xavier_uniform_(self.W_value.weight)
        nn.init.xavier_uniform_(self.W_key)
        # nn.init.xavier_uniform_(self.attention_scoring)
        self.W_key.data *= (self.mask.T)


    def forward(self, x, edge_index,batch):
        # Splitting input features        
        P1, P2 = x[:, :self.p1_channels], x[:, self.p1_channels:self.p1_channels+self.p2_channels]
        maskedweight = self.W_key * (self.mask.T)
        # Create a DataFrame from the numpy array
        maskedweight = maskedweight.T
        # Compute Key, Query, Value
        K = F.linear(P1, maskedweight)
        manual_output = P1.unsqueeze(1) * maskedweight.unsqueeze(0)        
        Q = self.W_query(P2)
        # focuesedQ = torch.stack([Q[batch == i][0] for i in range(batch.max() + 1)])
        # Reshape Q to [6445, 24, 1] for broadcasting
        Q_expanded = Q.unsqueeze(-1)  # Adds a new dimension at the end

        # Perform element-wise multiplication
        communication = manual_output * Q_expanded
        V = self.W_value(P2)
        # Querying
        alpha = Q * K 
        alpha = alpha - alpha.max()
        attention_coefficients = F.softmax(alpha, dim=0)
        return attention_coefficients * V,communication,attention_coefficients,V


class GATGraphClassifier(nn.Module):
    def __init__(self, FChidden_channels_2,FChidden_channels_3,FChidden_channels_4, num_classes,device,ligand_channel,receptor_channel,TF_channel,mask_indexes):
        super(GATGraphClassifier, self).__init__()
        # p1_channels, p2_channels, out_channels,mask_indexes,device
        
        self.gat_conv = CustomGATConv(ligand_channel, receptor_channel, mask_indexes,device)
        self.communicationFC1 = nn.Linear(receptor_channel, FChidden_channels_3)        
        self.communicationFC2= nn.Linear(FChidden_channels_3, FChidden_channels_4)
        
        self.fc1 = nn.Linear(ligand_channel+receptor_channel+TF_channel, FChidden_channels_2)        
        self.fc2 = nn.Linear(FChidden_channels_2, FChidden_channels_3)
        self.fc3 = nn.Linear(FChidden_channels_3, FChidden_channels_4)
        self.fc4 = nn.Linear(2*FChidden_channels_4, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x, edge_index, batch):

        x1,communication,attention_coefficients,V = self.gat_conv(x, edge_index,batch)
        x1 = torch.stack([x1[batch == i][0] for i in range(batch.max() + 1)])
        x1 = self.communicationFC1(x1)
        x1 = self.dropout(x1)
        x1 = self.communicationFC2(x1)
        x1 = self.dropout(x1)
        
        x2 = self.fc1(torch.stack([x[batch == i][0] for i in range(batch.max() + 1)]))
        x2 = self.dropout(x2)
        x2 = self.fc2(x2)
        x2 = self.dropout(x2)
        x2 = self.fc3(x2)
        x2 = self.dropout(x2)
        x = self.fc4(torch.cat((x1, x2), dim=1))
        
       
        return x,communication,attention_coefficients,V
