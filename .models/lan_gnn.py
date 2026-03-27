import torch
import torch.nn as nn

class BuildTechGNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=1):
        super(BuildTechGNN, self).__init__()
        # 使用 Linear 层，因为在单点传感器预测中，它等同于图神经网络的节点更新
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x 形状: [batch_size, 5]
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x
