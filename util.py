import json
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def calculate_rolling_average(rewards, window_size):
    """
    compute the rolling average of rewards with the specified window size
    """
    avg_rewards = []
    if len(rewards) < window_size:
        return avg_rewards
    for i in range(len(rewards) - window_size + 1):
        window = rewards[i:i+window_size]
        avg_rewards.append(sum(window) / window_size)
    
    return avg_rewards


def plot(result_file="training_metrics_gnn.json"):
    plt.figure(figsize=(14, 6))
    avg_rewards = json.load(open(result_file, "r"))
    plot_steps = avg_rewards['steps']
    legend_labels = []
    for layer_type, rolling_avg_rewards in avg_rewards.items():
        if layer_type == 'steps':
            continue
        print(layer_type)
        legend_labels.append(layer_type)
        plt.plot(plot_steps, rolling_avg_rewards)

    plt.title('PPO with GNN-based Policy')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.grid(True, alpha=0.3)
    plt.legend(legend_labels)
    plt.tight_layout()
    plt.savefig(result_file.replace(".json", ".png"), dpi=300) 

def feed_random_seeds(seed):
    """
    feed seed to all the random functions to fix generation
    @param seed:
    @return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TrainingMetricsCallback(BaseCallback):
    # self defined callback to log training metrics
    def __init__(self, monitor_log_dir, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.monitor_log_dir = monitor_log_dir
        self.losses = []
        self.episode_rewards = []
        self.episode_counts = []
        
    def _on_step(self) -> bool:
        # collect loss if available
        if 'losses' in self.locals:
            total_loss = sum(self.locals['losses'].values())
            self.losses.append(total_loss.item())
        return True
        
    def _on_rollout_end(self) -> None:
        # load monitor logs to get episode rewards
        try:
            x, y = ts2xy(load_results(self.monitor_log_dir), 'timesteps')
            if len(y) > len(self.episode_rewards):
                # only append new rewards
                new_rewards = y[len(self.episode_rewards):]
                self.episode_rewards.extend(new_rewards)
                self.episode_counts.extend(range(
                    len(self.episode_rewards) - len(new_rewards),
                    len(self.episode_rewards)
                ))
        except Exception as e:
            print(f"meet error when loading monitor logs: {e}")

def get_tensorboard_logs(base_dir):
    # read the tensorboard logs from the specified directory
    log_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not log_dirs:
        print("can not find any log dirs")
        return
    log_path = os.path.join(base_dir, log_dirs[0])
    event_acc = EventAccumulator(log_path).Reload()

    train_results = {}
    for tag in event_acc.Tags()['scalars']:
        train_results['steps'] = [event.step for event in event_acc.Scalars(tag)]
        train_results[tag.split('/')[1]] = [event.value for event in event_acc.Scalars(tag)]
    return train_results