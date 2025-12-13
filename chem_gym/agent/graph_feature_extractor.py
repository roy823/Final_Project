import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GraphResidualBlock(nn.Module):
    """
    图残差块：模拟 GCN 层，增加了残差连接和层归一化，训练更稳定。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin_msg = nn.Linear(hidden_dim, hidden_dim, bias=False) # 消息变换
        self.lin_upd = nn.Sequential( # 节点更新 MLP
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        # h: (B, N, C) - 节点特征
        # a_norm: (B, N, N) - 归一化邻接矩阵
        
        # 1. 消息传递 (Message Passing): A * (W * H)
        msg = torch.bmm(a_norm, self.lin_msg(h))
        
        # 2. 节点更新 (Update)
        out = self.lin_upd(msg)
        
        # 3. 残差连接 + 归一化
        return self.norm(h + out)


class CrystalGraphFeatureExtractor(BaseFeaturesExtractor):
    """
    SB3 MultiInputPolicy 专用的晶体图特征提取器。
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__(observation_space, features_dim)

        node_f = observation_space["node_features"].shape[1]

        # 1. 节点嵌入层 (Input Embedding)
        self.node_in = nn.Sequential(
            nn.Linear(node_f, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 2. 图卷积层堆叠 (Backbone)
        self.blocks = nn.ModuleList([GraphResidualBlock(hidden_dim) for _ in range(n_layers)])

        # 3. 输出头 (Output Head)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, features_dim),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        x = observations["node_features"]   # (B, N, F)
        a = observations["adjacency"]       # (B, N, N)
        m = observations.get("node_mask", None)  # (B, N)

        # Embedding
        h = self.node_in(x)

        # 归一化邻接矩阵 (Symmetric Normalization): D^{-1/2} A D^{-1/2}
        # 这能防止深层网络中特征数值爆炸
        deg = a.sum(dim=-1).clamp(min=1e-6)  # 度矩阵 (B, N)
        deg_inv_sqrt = deg.rsqrt()
        a_norm = a * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)

        # GNN Layers
        for blk in self.blocks:
            h = blk(h, a_norm)

        # Global Pooling (Readout)
        if m is None:
            g = h.mean(dim=1)
        else:
            # Masked Mean Pooling (只对真实原子求平均)
            m = m.unsqueeze(-1)  # (B, N, 1)
            h = h * m
            denom = m.sum(dim=1).clamp(min=1e-6)
            g = h.sum(dim=1) / denom

        return self.head(g)