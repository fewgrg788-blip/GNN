import torch
import torch.nn as nn
from torch_geometric.nn import GATConv # [UPDATE: 從 GCNConv 改為 GATConv]

class HK_Pro_Model(nn.Module):
    def __init__(self, node_features=3, hidden_dim=128, seq_len=24, horizon=6):
        """
        UPDATED: 升級為 GAT (Graph Attention Network) 結構
        """
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        
        # Temporal Layer (GRU)
        self.temporal = nn.GRU(
            input_size=node_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # --- [UPDATE 1: Spatial Layers 升級為 GAT] ---
        # heads=4 代表 4 個注意力頭，讓模型從 4 種不同角度學習站點間的相互影響
        # concat=False 代表將多頭輸出的結果取平均，確保維度維持在 hidden_dim，不干擾下游層
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        # ---------------------------------------------
        
        # Output Projection
        self.fc_out = nn.Linear(hidden_dim, horizon)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        """
        Input x shape: [B, N, T, F] -> (Batch, Stations, Time, Features)
        Example: [8, 18, 24, 3]
        """
        B, N, T, F = x.shape
        device = x.device
        
        # --- [保留: BATCHED EDGE INDEX LOGIC] ---
        edge_index_all = []
        for i in range(B):
            edge_index_all.append(edge_index + i * N)
        batched_edge_index = torch.cat(edge_index_all, dim=1).to(device)

        # 展平 B 和 N 以供 GRU 處理
        x = x.reshape(B * N, T, F)
        
        # 1. Temporal Phase: GRU 提取時間特徵
        temporal_out, _ = self.temporal(x)
        temporal_out = temporal_out[:, -1, :]
        
        # 2. Spatial Phase: GAT 動態分配圖神經網路權重
        # [UPDATE 2: 使用 gat 取代 gcn]
        spatial_out = self.gat1(temporal_out, batched_edge_index)
        spatial_out = self.relu(spatial_out)
        spatial_out = self.dropout(spatial_out)
        
        spatial_out = self.gat2(spatial_out, batched_edge_index)
        spatial_out = self.relu(spatial_out)
        
        # 3. Output Phase: 預測未來
        out = self.fc_out(spatial_out)
        
        # Reshape 回到 batch 格式: [B, N, horizon]
        out = out.view(B, N, self.horizon)
        return out