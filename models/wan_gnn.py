import torch
import torch.nn as nn

class WAN_GNN(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=32, output_dim=1):
        super(WAN_GNN, self). __init__()
        # 针对 18 区数据的全连接层
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2) # 防止过拟合

    def forward(self, x):
        # x 形状: [batch_size, 18]
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
