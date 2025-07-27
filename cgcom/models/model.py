import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class CustomGATConv(nn.Module):
    def __init__(self, p1_channels, p2_channels, mask_indexes, device, disable_lr_masking=False):
        super(CustomGATConv, self).__init__()
        self.p1_channels = p1_channels
        self.p2_channels = p2_channels
        self.device = device
        self.disable_lr_masking = disable_lr_masking
        
        # Trainable weight matrices
        self.W_query = nn.Linear(p2_channels, p2_channels)
        self.W_value = nn.Linear(p2_channels, p2_channels)
        # Initialize W_key with proper device placement
        self.W_key = nn.Parameter(torch.Tensor(p1_channels, p2_channels))
        
        # Create mask and register as buffer so it moves with the model
        mask = torch.zeros(p2_channels, p1_channels)
        
        if disable_lr_masking:
            # Allow all connections - set mask to all ones
            mask.fill_(1)
        else:
            # Use LR pairs only
            for mask_index in mask_indexes:
                mask[mask_index[0], mask_index[1]] = 1
        
        # Register mask as a buffer so it automatically moves with the model
        self.register_buffer('mask', mask)
        
        # Register initialization flag as buffer so it persists across device moves
        self.register_buffer('_weights_initialized', torch.tensor(False))
        
        print(f"W_key shape: {self.W_key.shape}")
        print(f"Mask shape: {self.mask.shape}")
        print(f"LR masking disabled: {disable_lr_masking}")

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_query.weight)
        nn.init.xavier_uniform_(self.W_value.weight)
        nn.init.xavier_uniform_(self.W_key)
        
    def forward(self, x, edge_index, batch):
        # Apply initial weight masking on first forward pass when tensors are on same device
        if not self._weights_initialized.item():
            with torch.no_grad():
                self.W_key.data *= (self.mask.T)
                self._weights_initialized.fill_(True)
            
        if self.disable_lr_masking:
            # Both P1 and P2 use the full gene set (same input)
            P1 = x[:, :self.p1_channels]  # Full genes
            P2 = x[:, :self.p2_channels]  # Same full genes
        else:
            # Original LR splitting
            P1, P2 = x[:, :self.p1_channels], x[:, self.p1_channels:self.p1_channels+self.p2_channels]
        
        # Apply mask to weights (ensure both are on same device)
        maskedweight = self.W_key * (self.mask.T)
        maskedweight = maskedweight.T
        
        # Compute Key, Query, Value
        K = F.linear(P1, maskedweight)
        manual_output = P1.unsqueeze(1) * maskedweight.unsqueeze(0)        
        Q = self.W_query(P2)
        Q_expanded = Q.unsqueeze(-1)

        # Perform element-wise multiplication
        communication = manual_output * Q_expanded
        V = self.W_value(P2)
        
        # Querying
        alpha = Q * K 
        alpha = alpha - alpha.max()
        attention_coefficients = F.softmax(alpha, dim=0)
        
        return attention_coefficients * V, communication, attention_coefficients, V


class GATGraphClassifier(nn.Module):
    def __init__(self, FChidden_channels_2, FChidden_channels_3, FChidden_channels_4, num_classes, device, 
                 ligand_channel, receptor_channel, TF_channel, mask_indexes, disable_lr_masking=False):
        super(GATGraphClassifier, self).__init__()
        
        self.gat_conv = CustomGATConv(ligand_channel, receptor_channel, mask_indexes, device, disable_lr_masking)
        self.communicationFC1 = nn.Linear(receptor_channel, FChidden_channels_3)        
        self.communicationFC2 = nn.Linear(FChidden_channels_3, FChidden_channels_4)
        
        # Adjust input size for fc1 based on whether LR masking is disabled
        if disable_lr_masking:
            # When disabled, we use full genes for both ligand and receptor channels
            # so total input is ligand_channel + TF_channel (no separate receptor channel)
            fc1_input_size = ligand_channel + TF_channel
        else:
            # Original: ligand + receptor + TF channels
            fc1_input_size = ligand_channel + receptor_channel + TF_channel
            
        self.fc1 = nn.Linear(fc1_input_size, FChidden_channels_2)        
        self.fc2 = nn.Linear(FChidden_channels_2, FChidden_channels_3)
        self.fc3 = nn.Linear(FChidden_channels_3, FChidden_channels_4)
        self.fc4 = nn.Linear(2 * FChidden_channels_4, num_classes)
        self.dropout = nn.Dropout()
        
        self.disable_lr_masking = disable_lr_masking

    def forward(self, x, edge_index, batch):
        x1, communication, attention_coefficients, V = self.gat_conv(x, edge_index, batch)
        x1 = torch.stack([x1[batch == i][0] for i in range(batch.max() + 1)])
        x1 = self.communicationFC1(x1)
        x1 = self.dropout(x1)
        x1 = self.communicationFC2(x1)
        x1 = self.dropout(x1)
        
        # Extract center node features for direct pathway
        x2_input = torch.stack([x[batch == i][0] for i in range(batch.max() + 1)])
        
        x2 = self.fc1(x2_input)
        x2 = self.dropout(x2)
        x2 = self.fc2(x2)
        x2 = self.dropout(x2)
        x2 = self.fc3(x2)
        x2 = self.dropout(x2)
        x = self.fc4(torch.cat((x1, x2), dim=1))
        
        return x, communication, attention_coefficients, V
