import gymnasium as gym
import sb3_contrib
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

# 2. GCN特征提取器
class GCNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 64, 
                 num_nodes: int = 10, node_feat_dim: int = 5):
        super().__init__(observation_space, features_dim)
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.gcn1 = GCNConv(node_feat_dim, 32)
        self.gcn2 = GCNConv(32, 64)
        self.relu = nn.ReLU()
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # 解析节点特征
        node_feat_flat = observations[:, :self.num_nodes * self.node_feat_dim]
        node_features = node_feat_flat.reshape(-1, self.node_feat_dim).float()
        
        # 获取边索引
        edge_index = self._get_edge_index(batch_size, device=observations.device)
        
        # 生成batch索引
        batch = torch.arange(batch_size, device=observations.device).repeat_interleave(self.num_nodes)
        
        # GCN前向传播
        x = self.gcn1(node_features, edge_index)
        x = self.relu(x)
        x = self.gcn2(x, edge_index)
        graph_feat = global_mean_pool(x, batch)
        
        return graph_feat
    
    def _get_edge_index(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # JSP析取图的边索引生成（作业内任务先后关系）
        edge_index = []
        # 假设每个作业有2个任务，共2个作业，总计4个节点
        for job_id in range(2):  # 2个作业
            task1_idx = job_id * 2  # 第一个任务索引
            task2_idx = job_id * 2 + 1  # 第二个任务索引
            edge_index.append([task1_idx, task2_idx])  # 任务先后关系
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        return edge_index

# 3. 基于GCN的自定义策略
class GCNMaskablePolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GCNFeaturesExtractor,
            features_extractor_kwargs={
                "features_dim": 64,
                "num_nodes": 4,    # 2个作业 × 2个任务 = 4个节点（根据实际情况调整）
                "node_feat_dim": 4 # 节点特征维度（根据实际观测调整）
            },
            *args,** kwargs,
        )
        self.share_features_extractor = False

# 4. 环境初始化与训练（最终版本）
if __name__ == "__main__":
    from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
    from graph_jsp_env.disjunctive_graph_logger import log

    # 定义JSP问题实例
    jsp = np.array([
        [[1, 2, 0],  # job 0
         [0, 2, 1]],  # job 1
        [[17, 12, 19],  # task durations of job 0
         [8, 6, 2]]  # task durations of job 1
    ])

    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=True,
        action_mode='task',  # alternative 'job'
    )
    env = Monitor(env)


    def mask_fn(env: gym.Env) -> np.ndarray:
        return env.unwrapped.valid_action_mask()
    env = ActionMasker(env, mask_fn)

    # 初始化模型
    model = sb3_contrib.MaskablePPO(
        GCNMaskablePolicy,
        env,
        verbose=1,
        policy_kwargs={
            "net_arch": {
                "pi": [64, 64],
                "vf": [64, 64]
            }
        }
    )

    # 训练
    log.info("开始训练GCN-PPO模型")
    model.learn(total_timesteps=10_000)
    