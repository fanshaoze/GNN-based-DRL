import gymnasium as gym
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, GCNConv, SAGEConv, GATConv, GATv2Conv, GINConv, TransformerConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy



# GCN features extractor
class GCNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 64, 
                 num_nodes: int = 10, node_feat_dim: int = 5, layer_type="GATv2Conv"):
        super().__init__(observation_space, features_dim)
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        print(f'create {layer_type} GCNFeaturesExtractor')
        if layer_type == "GCNConv":
            self.gnn_layer_1 = GCNConv(node_feat_dim, 128)
            self.gnn_layer_2 = GCNConv(128, 128)
        elif layer_type == "SAGEConv":
            self.gnn_layer_1 = SAGEConv(node_feat_dim, 128)
            self.gnn_layer_2 = SAGEConv(128, 128)
        elif layer_type == "GATConv":
            self.gnn_layer_1 = GATConv(node_feat_dim, 128)
            self.gnn_layer_2 = GATConv(128, 128)
        elif layer_type == "GATv2Conv":
            self.gnn_layer_1 = GATv2Conv(node_feat_dim, 128)
            self.gnn_layer_2 = GATv2Conv(128, 128)
        elif layer_type == "GINConv":
            nn1 = nn.Sequential(nn.Linear(node_feat_dim, 128), nn.ReLU(), nn.Linear(128, 128))
            nn2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
            self.gnn_layer_1 = GINConv(nn1)
            self.gnn_layer_2 = GINConv(nn2)
        elif layer_type == "TransformerConv":
            self.gnn_layer_1 = TransformerConv(node_feat_dim, 128, heads=4, dropout=0.1)
            self.gnn_layer_2 = TransformerConv(128*4, 128, heads=1, dropout=0.1)
        else:
            raise ValueError(f"Unsupported layer_type: {layer_type}")

        self.relu = nn.ReLU()
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # decode the node feature
        node_feat_flat = observations[:, :self.num_nodes * self.node_feat_dim]
        node_features = node_feat_flat.reshape(-1, self.node_feat_dim).float()
        
        # get edge index, the edge list
        edge_index = self._get_edge_index(batch_size, device=observations.device)
        
        # generate the batch vector
        batch = torch.arange(batch_size, device=observations.device).repeat_interleave(self.num_nodes)
        
        # gnn forward
        x = self.gnn_layer_1(node_features, edge_index)
        x = self.relu(x)
        x = self.gnn_layer_2(x, edge_index)
        # global pooling
        graph_feat = global_mean_pool(x, batch)
        
        return graph_feat
    
    def _get_edge_index(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # generate the edge index for the JSP graph. 
        # This need to be consistent with the environment's graph structure if you change the environment
        # This defines the connections in the graph
        # Here we assume a fixed JSP structure with 3 jobs and 6 tasks each as an example
        edge_index = []
        num_jobs = 3  # number of jobs 
        tasks_per_job = 6  # number of tasks per job
        
        for job_id in range(num_jobs):
            # each job's tasks are connected in sequence
            for task_idx in range(tasks_per_job - 1):
                # compute the global node indices
                from_node = job_id * tasks_per_job + task_idx
                to_node = job_id * tasks_per_job + (task_idx + 1)
                edge_index.append([from_node, to_node])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        return edge_index

# 3. the GCN-based Maskable Policy
class GCNMaskablePolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        layer_type,
        *args,
        **kwargs,
        
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GCNFeaturesExtractor,
            features_extractor_kwargs={
                "features_dim": 128,
                "num_nodes": 18,    # number of nodes, (in jsp is jobs * tasks_per_job)
                "node_feat_dim": 4, # node feature dimension
                'layer_type': layer_type
            },
            *args,** kwargs,
        )
        self.share_features_extractor = False






