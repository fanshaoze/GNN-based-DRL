import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, GINConv, TransformerConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Linear, BatchNorm1d, Parameter, Sequential, ReLU
from torch_scatter import scatter_add  # Import scatter_add from torch_scatter
import matplotlib.pyplot as plt
import time
import numpy as np

# ----------------------------
# Enhanced Custom Convolution (Fixed)
# ----------------------------
class EnhancedCustomConv(MessagePassing):
    """
    Enhanced custom convolution layer with fixed scatter_add usage
    """
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__(aggr='add')  # Base aggregation
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        
        self.lin = Linear(in_channels, out_channels)
        self.att_lin = Linear(2 * out_channels, 1)
        self.agg_lin = Linear(out_channels, out_channels)
        self.update_lin = Linear(out_channels, out_channels)
        self.bn = BatchNorm1d(out_channels)
        self.agg_param = Parameter(torch.Tensor(1, out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att_lin.reset_parameters()
        self.agg_lin.reset_parameters()
        self.update_lin.reset_parameters()
        self.bn.reset_parameters()
        torch.nn.init.xavier_uniform_(self.agg_param)
        
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, x=x)
        
    def message(self, x_i, x_j):
        att_input = torch.cat([x_i, x_j], dim=-1)
        att = self.att_lin(att_input).sigmoid()
        att = F.dropout(att, p=self.dropout, training=self.training)
        return x_j * att
        
    def aggregate(self, inputs, index, dim_size):
        """Fixed aggregation using torch_scatter.scatter_add"""
        # Use torch_scatter's scatter_add instead of self.scatter_add
        out = scatter_add(inputs, index, dim=0, dim_size=dim_size)
        
        out = out * self.agg_param
        out = self.agg_lin(out)
        return F.elu(out)
        
    def update(self, inputs, x):
        out = inputs + x
        out = self.update_lin(out)
        out = F.relu(out)
        return self.bn(out)

# ----------------------------
# GNN Model Wrapper
# ----------------------------
class GNN(torch.nn.Module):
    def __init__(self, conv_type, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = self._get_conv(conv_type, in_channels, hidden_channels)
        self.conv2 = self._get_conv(conv_type, hidden_channels, out_channels)
        
    def _get_conv(self, conv_type, in_channels, out_channels):
        if conv_type == 'GCN':
            return GCNConv(in_channels, out_channels)
        elif conv_type == 'SAGE':
            return SAGEConv(in_channels, out_channels)
        elif conv_type == 'GAT':
            return GATConv(in_channels, out_channels)
        elif conv_type == 'GATv2':
            return GATv2Conv(in_channels, out_channels)
        elif conv_type == 'Transformer':
            return TransformerConv(in_channels, out_channels, heads=1)
        elif conv_type == 'GIN':
            mlp = Sequential(
                Linear(in_channels, out_channels),
                ReLU(),
                Linear(out_channels, out_channels)
            )
            return GINConv(mlp)
        elif conv_type == 'EnhancedCustom':
            return EnhancedCustomConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown convolution type: {conv_type}")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ----------------------------
# Training and Evaluation
# ----------------------------
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            if mask is None:
                continue
            correct = int((pred[mask] == data.y[mask]).sum())
            acc = correct / int(mask.sum())
            accs.append(acc)
    return accs

# ----------------------------
# Experiment Execution
# ----------------------------
def run_experiment(conv_type, dataset, data, epochs=2000, device='cpu'):
    if data.train_mask is None:
        num_nodes = data.num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[:int(0.3*num_nodes)] = True
        val_mask[int(0.3*num_nodes):int(0.5*num_nodes)] = True
        test_mask[int(0.5*num_nodes):] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    model = GNN(conv_type, dataset.num_features, 16, dataset.num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_test_acc = 0
    train_times = []
    train_losses = []
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss = train(model, data, optimizer, criterion)
        train_time = time.time() - start_time
        train_times.append(train_time)
        train_losses.append(loss)
        
        accs = test(model, data)
        train_acc, val_acc, test_acc = accs if len(accs) == 3 else (accs[0], accs[1], accs[1])
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        if epoch % 10 == 0:
            print(f'{conv_type} - Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    avg_train_time = np.mean(train_times)
    return best_test_acc, avg_train_time, train_losses

# ----------------------------
# Results Visualization
# ----------------------------
def plot_test_accuracy(results):
    """Plot test accuracy comparison as a separate figure"""
    conv_types = list(results.keys())
    test_accs = [results[conv]['test_acc'] for conv in conv_types]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(conv_types, test_accs, color='skyblue')
    plt.xlabel('Convolution Type', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Test Accuracy Comparison (PubMed Dataset)', fontsize=14)
    plt.ylim(0.6, 0.9)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('pubmed_test_accuracy.png', dpi=300, bbox_inches='tight')

def plot_training_efficiency(results):
    """Plot training efficiency comparison as a separate figure"""
    conv_types = list(results.keys())
    avg_times = [results[conv]['avg_time'] for conv in conv_types]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(conv_types, avg_times, color='lightgreen')
    plt.xlabel('Convolution Type', fontsize=12)
    plt.ylabel('Average Training Time (s/epoch)', fontsize=12)
    plt.title('Training Efficiency Comparison', fontsize=14)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.6f}', ha='center', va='bottom', rotation=45, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('pubmed_training_efficiency.png', dpi=300, bbox_inches='tight')

def plot_training_loss(losses):
    """Plot training loss curves as a separate figure"""
    conv_types = list(losses.keys())
    epochs = len(next(iter(losses.values())))
    
    plt.figure(figsize=(12, 6))
    for conv_type, loss in losses.items():
        plt.plot(range(1, epochs+1), loss, label=conv_type, linewidth=0.3)
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.1)
    
    plt.tight_layout()
    plt.savefig('pubmed_training_loss.png', dpi=300, bbox_inches='tight')

# ----------------------------
# Main Execution
# ----------------------------
def main():
    dataset = Planetoid(root='data/Planetoid', name='PubMed')
    data = dataset[0]
    
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Graph: {data}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    conv_types = ['EnhancedCustom', 'GCN', 'SAGE', 'GAT', 'GATv2', 'Transformer', 'GIN']
    results = {}
    all_losses = {}
    
    for conv_type in conv_types:
        print(f"\n===== Starting {conv_type} experiment =====")
        test_acc, avg_time, losses = run_experiment(conv_type, dataset, data, device=device)
        results[conv_type] = {'test_acc': test_acc, 'avg_time': avg_time}
        all_losses[conv_type] = losses
        print(f"{conv_type} - Best test accuracy: {test_acc:.4f}, "
              f"Average training time per epoch: {avg_time:.6f} seconds")
    
        # Generate separate plots
    plot_test_accuracy(results)
    plot_training_efficiency(results)
    plot_training_loss(all_losses)

if __name__ == "__main__":
    main()
