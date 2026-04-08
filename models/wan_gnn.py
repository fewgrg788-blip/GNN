import torch
import torch.nn as nn
from torch_geometric.nn import GATConv # 或者是 GCNConv，視你最終採用的版本而定

class WAN_GNN(nn.Module):
    def __init__(self, node_features=3, hidden_dim=128, seq_len=24, horizon=6):
        super(WAN_GNN, self).__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        
        # 1. 時間特徵提取 (一定要有 GRU)
        self.temporal = nn.GRU(node_features, hidden_dim, num_layers=2, batch_first=True)
        
        # 2. 空間特徵提取 (真正的 GNN 部份)
        self.spatial = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # 3. 輸出層
        self.fc_out = nn.Linear(hidden_dim, horizon)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        # x: [B, N, T, F] -> [B*N, T, F]
        B, N, T, F = x.shape
        x = x.reshape(B * N, T, F)
        
        # 時間處理
        _, h_n = self.temporal(x)
        out = h_n[-1] # 取最後一層的 Hidden State
        
        # 空間處理 (GAT)
        # 注意：在 Render 部署時也需要傳入 edge_index
        out = self.spatial(out, edge_index)
        out = self.relu(out)
        
        # 預測未來
        out = self.fc_out(out) # [B*N, Horizon]
        return out.reshape(B, N, -1)
