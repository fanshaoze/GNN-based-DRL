import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import json
from util import feed_random_seeds
from net_arch_design import CustomSharedPolicy

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
feed_random_seeds(42)

class TrainingCallback(BaseCallback):
    """
    Custom callback to track training metrics during training
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = [0.0 for _ in range(self.locals.get("n_envs", 1))]  # Track per-env rewards
        self.current_lengths = [0.0 for _ in range(self.locals.get("n_envs", 1))]  # Track per-env lengths
        self.losses = []  # Track loss values
        
    def _on_training_start(self) -> None:
        # Initialize per-env tracking when training starts
        self.current_rewards = [0.0 for _ in range(self.locals.get("n_envs", 1))]
        self.current_lengths = [0.0 for _ in range(self.locals.get("n_envs", 1))]
        
    def _on_step(self) -> bool:
        # Get the rewards and dones from the current step
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        n_envs = len(rewards)
        
        # Update per-env rewards and lengths
        for i in range(n_envs):
            self.current_rewards[i] += rewards[i]
            self.current_lengths[i] += 1
            
            # If episode is done, record the episode reward and length
            if dones[i]:
                self.episode_rewards.append(self.current_rewards[i])
                self.episode_lengths.append(self.current_lengths[i])
                # Reset per-env tracking for this environment
                self.current_rewards[i] = 0.0
                self.current_lengths[i] = 0.0
        
        # Track loss if available in locals
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            if 'train/loss' in self.model.logger.name_to_value:
                self.losses.append(self.model.logger.name_to_value['train/loss'])
            elif 'train/policy_loss' in self.model.logger.name_to_value:
                policy_loss = self.model.logger.name_to_value['train/policy_loss']
                value_loss = self.model.logger.name_to_value.get('train/value_loss', 0)
                entropy_loss = self.model.logger.name_to_value.get('train/entropy_loss', 0)
                total_loss = policy_loss + value_loss + entropy_loss
                self.losses.append(total_loss)
        
        return True

def train_ppo_variant(env_id, variant_name, params, total_timesteps=100000):
    """
    Train a PPO variant with specified parameters
    """
    print(f"Training {variant_name} on {env_id}...")
    
    # Create environment
    env = DummyVecEnv([lambda: gym.make(env_id)])
    eval_env = gym.make(env_id)
    
    # Extract policy from params if present, otherwise use MlpPolicy
    policy = params.pop("policy", "MlpPolicy")
    
    # Create model with specified parameters
    model = PPO(
        policy,
        env,
        verbose=1,
        **params
    )
    
    # Create callback to track training progress
    callback = TrainingCallback()
    
    # Train the model
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    training_time = time.time() - start_time
    
    print(f"Completed {variant_name} training in {training_time:.2f}s")
    
    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"{variant_name} - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    eval_env.close()
    env.close()
    
    return {
        'model': model,
        'training_time': training_time,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'episode_rewards': callback.episode_rewards,
        'losses': callback.losses
    }

def run_tricks_study(env_id="LunarLander-v3", total_timesteps=100000):
    """
    Run the tricks study comparing different PPO variants
    """
    # Using LunarLander-v3 for better differentiation
    
    # Define the different PPO variants with exact specifications
    variants = {
        # # PPO without entropy (ent_coef=0)
        "dual-clip": {
            "clip_range": 0.2,              # Standard policy clip range
            "clip_range_vf": None,           # With value function clipping
            "ent_coef": 0.01,              # With entropy regularization
            "target_kl": None,             # With KL divergence early stopping
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "device": "cpu",

            "dual_clip": 0.5,
        },

        # PPO_adaptive_clip_kl
        "adaptive-clip with kl": {
            "clip_range": 0.2,              # Standard policy clip range
            "clip_range_vf": None,           # With value function clipping
            "ent_coef": 0.01,              # With entropy regularization
            "target_kl": None,             # With KL divergence early stopping
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "device": "cpu",

            "adaptive_clip_kl": 0.01,
        },

        # PPO_entropy_decay
        "entropy decay": {
            "clip_range": 0.2,              # Standard policy clip range
            "clip_range_vf": None,           # With value function clipping
            "ent_coef": 0.2,              # With entropy regularization
            "target_kl": None,             # With KL divergence early stopping
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "device": "cpu",

            "entropy_decay": 0.99,
        },
        # Plain PPO
        # "PPO_split_net": {
        #     "clip_range": 0.2,
        #     "ent_coef": 0.01,
        #     "learning_rate": 3e-4,
        #     "n_steps": 2048,
        #     "batch_size": 64,
        #     "n_epochs": 10,
        #     "gae_lambda": 0.95,
        #     "max_grad_norm": 0.5,
        #     "device": "cpu",
        #     "policy_kwargs": {
        #         "share_features_extractor": False,
        #         "net_arch": {
        #             "pi": [64, 64, 64, 64, 64, 64],
        #             "vf": [64, 64, 64, 64, 64, 64]
        #         }
        #     }
        # },
        "winsorize advantage": {
            "clip_range": 0.2,              # Standard policy clip range
            "clip_range_vf": None,           # With value function clipping
            "ent_coef": 0.2,              # With entropy regularization
            "target_kl": None,             # With KL divergence early stopping
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "device": "cpu",

            "winsorize_advantage": 5,
        },

        "popart": {
            "clip_range": 0.2,              # Standard policy clip range
            "clip_range_vf": None,           # With value function clipping
            "ent_coef": 0.2,              # With entropy regularization
            "target_kl": None,             # With KL divergence early stopping
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "device": "cpu",

            "popart": True,
        },

        "PPO split optimizer": {
            "clip_range": 0.2,              # Standard policy clip range
            "clip_range_vf": None,           # With value function clipping
            "ent_coef": 0.2,              # With entropy regularization
            "target_kl": None,             # With KL divergence early stopping
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "device": "cpu",

            "split_optimizer": True,
            "value_optimizer_update_steps": 3,
        },

         "PPO policy regularization": {
            "clip_range": 0.2,              # Standard policy clip range
            "clip_range_vf": None,           # With value function clipping
            "ent_coef": 0.2,              # With entropy regularization
            "target_kl": None,             # With KL divergence early stopping
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "device": "cpu",

            "policy_reg": True,
        },

        "PPO": {
            "clip_range": 0.2,              # Standard policy clip range
            "clip_range_vf": None,           # With value function clipping
            "ent_coef": 0.2,              # With entropy regularization
            "target_kl": None,             # With KL divergence early stopping
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "device": "cpu",
        },

        "6 layers: split P and V net": {
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "device": "cpu",
            "policy": CustomSharedPolicy,
            "policy_kwargs": {
                "net_arch_config": {
                    "policy_net": [64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh"],
                    "value_net": [64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh"],
                    "shared_layers": [] 
                }
            }
        },
        "6 layers: [1,2,3,4,5] share, [6] split": {
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "device": "cpu",
            "policy": CustomSharedPolicy,
            "policy_kwargs": {
                "net_arch_config": {
                    "policy_net": [64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh"],
                    "value_net": [64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh"],
                    "shared_layers": [0, 2, 4, 6, 8]  # 总共6层，前5层共享，最后1层分离
                }
            }
        },
        "6 layers: [1,2,3] share, [4,5,6] split": {
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "device": "cpu",
            "policy": CustomSharedPolicy,
            "policy_kwargs": {
                "net_arch_config": {
                    "policy_net": [64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh"],
                    "value_net": [64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh"],
                    "shared_layers": [0, 2, 4]  # 总共6层，前3层共享，后3层分离
                }
            }
        },
        "6 layers: [1,3,5] share, [2,4,6] split": {
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "device": "cpu",
            "policy": CustomSharedPolicy,
            "policy_kwargs": {
                "net_arch_config": {
                    "policy_net": [64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh"],
                    "value_net": [64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh", 64, "tanh"],
                    "shared_layers": [0, 4, 8]  # 交替共享：第1,3,5层共享，第2,4,6层分离
                }
            }
        },


        # # (4) Plain PPO
        # "PPO": {
        #     "clip_range": 0.2,              # Standard policy clip range
        #     "clip_range_vf": None,           # With value function clipping
        #     "ent_coef": 0.01,              # With entropy regularization
        #     "target_kl": None,             # With KL divergence early stopping
        #     "learning_rate": 3e-4,
        #     "n_steps": 2048,
        #     "batch_size": 64,
        #     "n_epochs": 10,
        #     "gae_lambda": 0.95,
        #     "max_grad_norm": 0.5,
        #     "device": "cpu"
        # },
        
        
        
        
    }
    
    # Train each variant
    results = {}
    if os.path.exists(f"ppo_tricks_{env_id}.json"):
        results = json.load(open(f"ppo_tricks_{env_id}.json"))
    else:
        results = {}
    total_variants = len(variants)
    
    for i, (variant_name, params) in enumerate(variants.items()):
        if variant_name in results:
            continue
        
        print(f"\n[{i+1}/{total_variants}] Training {variant_name}")
        result = train_ppo_variant(env_id, variant_name, params, total_timesteps=total_timesteps)
        results[variant_name] = result
        print(f"Completed {variant_name}")
    
    return results, list(variants.keys())

def plot_results(results, variant_names=None, env_id="LunarLander-v3", sub_folder_name=""):
    """
    Plot the results of the tricks study
    """
    # Create directory for plots
    plot_dir = f"ppo_tricks_{env_id}/{sub_folder_name}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot cumulative rewards over episodes
    plt.figure(figsize=(15, 10))
    
    variant_names = list(results.keys()) if variant_names is None else variant_names
    if "PPO" in variant_names:
        variant_names.remove("PPO")
        variant_names.append("PPO")
    print(variant_names)
    for variant_name in variant_names:
        result = results[variant_name]
        episode_rewards = result['episode_rewards']
        cumulative_rewards = np.cumsum(episode_rewards)
        plt.plot(cumulative_rewards, label=variant_name, alpha=0.7, linewidth=1.5)
    
    plt.title('Cumulative Reward Comparison Across PPO Variants')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'cumulative_rewards.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot smoothed episode rewards
    plt.figure(figsize=(15, 10))
    
    variant_names = list(results.keys()) if variant_names is None else variant_names
    if "PPO" in variant_names:
        variant_names.remove("PPO")
        variant_names.append("PPO")
    print(variant_names)
    for variant_name in variant_names:
        result = results[variant_name]
        episode_rewards = result['episode_rewards']
        window_size = min(1000, len(episode_rewards) // 10)
        if window_size > 0:
            smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_rewards, label=variant_name, alpha=0.8, linewidth=2)
        else:
            plt.plot(episode_rewards, label=variant_name, alpha=0.7, linewidth=1.5)
    
    plt.title(f'PPO compare on {env_id}: Smoothed Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'smoothed_rewards.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot loss curves if available
    plt.figure(figsize=(15, 10))
    has_losses = False
    variant_names = list(results.keys()) if variant_names is None else variant_names
    if "PPO" in variant_names:
        variant_names.remove("PPO")
        variant_names.append("PPO")
    print(variant_names)
    for variant_name in variant_names:
        result = results[variant_name]
        if 'losses' in result and len(result['losses']) > 0:
            losses = result['losses']
            plt.plot(losses, label=variant_name, alpha=0.7, linewidth=1.5)
            has_losses = True
    
    if has_losses:
        plt.title(f'PPO compare on {env_id}: Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot training performance summary
    variant_names = list(results.keys()) if variant_names is None else variant_names
    mean_rewards = [results[name]['mean_reward'] for name in variant_names]
    training_times = [results[name]['training_time'] for name in variant_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot mean rewards
    bars1 = ax1.bar(range(len(variant_names)), mean_rewards, color='skyblue', alpha=0.7)
    ax1.set_xlabel('PPO Variant')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title(f'PPO compare on {env_id}: Mean Reward')
    ax1.set_xticks(range(len(variant_names)))
    ax1.set_xticklabels([name.replace('PPO_', '\n') for name in variant_names], rotation=45, ha='right')
    
    for bar, value in zip(bars1, mean_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_rewards)*0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot training times
    bars2 = ax2.bar(range(len(variant_names)), training_times, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('PPO Variant')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title(f'PPO compare on {env_id}: Training Time')
    ax2.set_xticks(range(len(variant_names)))
    ax2.set_xticklabels([name.replace('PPO_', '\n') for name in variant_names], rotation=45, ha='right')
    
    for bar, value in zip(bars2, training_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.01, 
                f'{value:.2f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {plot_dir}/ directory")

def print_results_summary(results):
    """
    Print a summary of the results
    """
    print("\n" + "="*100)
    print("tricks STUDY RESULTS SUMMARY")
    print("="*100)
    
    print(f"{'Variant':<50} {'Mean Reward':<12} {'Std Reward':<12} {'Training Time':<12} {'Episodes':<10}")
    print("-" * 100)
    
    for variant_name, result in results.items():
        print(f"{variant_name:<50} {result['mean_reward']:<12.2f} {result['std_reward']:<12.2f} {result['training_time']:<12.2f} {len(result['episode_rewards']):<10}")
    
    print("="*100)
    
    print("\nDetailed Analysis:")
    print("-" * 50)
    for variant_name, result in results.items():
        print(f"{variant_name}:")
        print(f"  - Final Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  - Training Time: {result['training_time']:.2f}s")
        print(f"  - Episodes Completed: {len(result['episode_rewards'])}")
        print(f"  - Max Episode Reward: {max(result['episode_rewards']) if result['episode_rewards'] else 0:.2f}")
        print()

def save_results(results, filename="ppo_tricks.json"):
    """
    Save results to a JSON file
    """
    serializable_results = {}
    for variant_name, result in results.items():
        serializable_results[variant_name] = {
            'mean_reward': float(result['mean_reward']),
            'std_reward': float(result['std_reward']),
            'training_time': float(result['training_time']),
            'episode_rewards': [float(r) for r in result['episode_rewards']],
            'losses': [float(l) for l in result.get('losses', [])],
            'num_episodes': len(result['episode_rewards'])
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filename}")

def main():
    
    """
    Main function to run the tricks study
    """
    print("Starting Final PPO tricks Study with MaskablePPO...")
    print("Comparing different PPO variants:")
    print()
    
    env_id = "LunarLander-v3" # 1_000_000
    # env_id = "CartPole-v1" # 200000
    # env_id = "MountainCarContinuous-v0" # 150000
    results, plot_keys = run_tricks_study(env_id, total_timesteps=1_000_000)
    print_results_summary(results)
    save_results(results, filename=f"ppo_tricks_{env_id}.json")
    with open(f"ppo_tricks_{env_id}.json", 'r') as f:
        loaded_results = json.load(f)
    plot_results(loaded_results, variant_names=plot_keys, env_id=env_id)
    
    print("\ntricks study completed successfully!")
    print("Check the 'ppo_tricks/' directory for plots.")

if __name__ == "__main__":
    # main()
    env_id = "LunarLander-v3"
    with open(f"ppo_tricks_{env_id}.json", 'r') as f:
        loaded_results = json.load(f)
    settings = [
         [['dual-clip', 'PPO'], "dual-clip"],
         [['adaptive-clip with kl', 'PPO'], 'adaptive-clip with kl'],
         [['entropy decay', 'PPO'], 'entropy decay'],
         [['winsorize advantage', 'PPO'], 'winsorize advantage'],
         [['popart', 'PPO'], 'popart'],
         [['PPO split optimizer', 'PPO'], 'PPO split optimizer'],
         [['PPO policy regularization', 'PPO'], 'PPO policy regularization'],
         [['6 layers: [1,2,3,4,5] share, [6] split', '6 layers: [1,2,3] share, [4,5,6] split', '6 layers: [1,3,5] share, [2,4,6] split', '6 layers: split P and V net', 'PPO'], 'different net structure'],
         ]
    for variant_names, sub_folder_name in settings:
        plot_results(loaded_results, variant_names=variant_names, env_id=env_id, sub_folder_name=sub_folder_name)

    
# ['dual-clip', 'adaptive-clip with kl', 'entropy decay', 'winsorize advantage', 'popart', 'PPO split optimizer', 'PPO policy regularization', 
# '6 layers: split P and V net', 
# '6 layers: [1,2,3,4,5] share, [6] split', '6 layers: [1,2,3] share, [4,5,6] split', '6 layers: [1,3,5] share, [2,4,6] split', 'PPO']