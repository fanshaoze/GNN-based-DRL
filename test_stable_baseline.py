import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO, DQN, DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
import time
import os

# Create directory for plots
plot_dir = "stable_baseline3_compare"
os.makedirs(plot_dir, exist_ok=True)

# Environments (simple to complex)
environments = [
    "CartPole-v1",
    # "MountainCar-v0",
    "LunarLander-v2",
    "BipedalWalker-v3",
    "Ant-v4"
]

# Algorithms
algorithms = {
    "A2C": A2C,
    "PPO": PPO,
    "DQN": DQN,
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC
}

# Reduced training steps
# steps: the total number of training steps for each environment
# policy: the type of policy network to use. "MlpPolicy" is a multi-layer perceptron (fully connected neural network) suitable for most environments.
    # check the "xxx Policies" for more info about the policies of the RL algorithm "xxx" 
    # For exmaple, for PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#ppo-policies
# log_interval: how often (in steps) to log (record) the training progress and evaluate performance.
    # Smaller intervals give more frequent feedback but may slow down training and increase the amount of logged data.
# train_params = {
#     "CartPole-v1": {"steps": 1000, "policy": "MlpPolicy", "log_interval": 100},
#     # "MountainCar-v0": {"steps": 20000, "policy": "MlpPolicy", "log_interval": 200},
#     "LunarLander-v2": {"steps": 3000, "policy": "MlpPolicy", "log_interval": 300},
#     "BipedalWalker-v3": {"steps": 1000, "policy": "MlpPolicy", "log_interval": 1000},
#     "Ant-v4": {"steps": 10000, "policy": "MlpPolicy", "log_interval": 1000}
# }

train_params = {
    "CartPole-v1": {"steps": 10000, "policy": "MlpPolicy", "log_interval": 100},
    # "MountainCar-v0": {"steps": 20000, "policy": "MlpPolicy", "log_interval": 200},
    "LunarLander-v2": {"steps": 30000, "policy": "MlpPolicy", "log_interval": 300},
    "BipedalWalker-v3": {"steps": 150000, "policy": "MlpPolicy", "log_interval": 1000},
    "Ant-v4": {"steps": 100000, "policy": "MlpPolicy", "log_interval": 1000}
}


def train_agent(env_name, algo_name, policy, total_steps, log_interval):
    """
    Train a given algorithm on a specified environment and return rewards over time.
    param env_name: str - Environment name
    param algo_name: str - Algorithm name
    param policy: str - Policy type
    param total_steps: int - Total training steps
    param log_interval: int - Steps between logging
    return: dict - {"timesteps": [...], "rewards": [...]}, or None if training failed
    """
    try:
        env = gym.make(env_name)
        model = None
        rewards = []
        timesteps = []
        current_step = 0
        
        # Handle action space compatibility
        if isinstance(env.action_space, gym.spaces.Box):  # Continuous environments
            if algo_name == "DQN":
                return None
                
            n_actions = env.action_space.shape[-1]
            # set up action noise for DDPG and TD3
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions)) if algo_name in ["DDPG", "TD3"] else None
            
            # Initialize model.
            # policy can be "MlpPolicy", "CnnPolicy", "MultiInputPolicy", etc.
            # env: the environment instance
            # verbose=0: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
            if algo_name == "A2C":
                model = A2C(policy, env)
            elif algo_name == "PPO":
                model = PPO(policy, env)
            elif algo_name == "DDPG":
                model = DDPG(policy, env, action_noise=action_noise)
            elif algo_name == "TD3":
                model = TD3(policy, env, action_noise=action_noise)
            elif algo_name == "SAC":
                model = SAC(policy, env)
                
        elif isinstance(env.action_space, gym.spaces.Discrete):  # Discrete environments
            # Do not run continuous algorithms on discrete envs.
            if algo_name in ["DDPG", "TD3", "SAC"]:
                return None
                
            # Only A2C(the actor-critic algorithm), PPO(the policy optimization algorithm) and DQN(the Q-learning algorithm) are suitable for discrete action spaces.
            if algo_name == "A2C":
                model = A2C(policy, env)
            elif algo_name == "PPO":
                model = PPO(policy, env)
            elif algo_name == "DQN":
                model = DQN(policy, env)
        
        if model:
            print(f"Training {algo_name} on {env_name}...")
            start_time = time.time()
            
            # Train in intervals and track rewards. The model train for "log_interval" steps each loop and then evaluate the performance of the model. 
            # This keep track of performance over time.
            while current_step < total_steps:
                # Train for log_interval steps
                model.learn(total_timesteps=log_interval)
                current_step += log_interval

                # Evaluate the model over several episodes to get a more stable estimate of performance
                eval_episodes = 10
                total_reward = 0
                # Run several episodes and accumulate rewards 
                for _ in range(eval_episodes):
                    obs, _ = env.reset()
                    episode_reward = 0
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        episode_reward += reward
                        done = terminated or truncated
                        
                    total_reward += episode_reward

                # recard the average reward over the evaluation episodes
                rewards.append(total_reward / eval_episodes)
                timesteps.append(current_step)
                
                # # Evaluate current performance.
                # obs, _ = env.reset()
                # episode_reward = 0
                # done = False
                # # Run one episode until done.
                # while not done:
                #     action, _ = model.predict(obs, deterministic=True)
                #     obs, reward, terminated, truncated, _ = env.step(action)
                #     episode_reward += reward
                #     done = terminated or truncated
                
                # # Record results to the rewards and timesteps lists
                # rewards.append(episode_reward)
                # timesteps.append(current_step)
                env.reset()
            
            print(f"Completed in {time.time()-start_time:.1f}s")
            env.close()
            return {"timesteps": timesteps, "rewards": rewards}
            
    except Exception as e:
        # Catch any error, print a short message of the expectation and return None
        print(f"Error with {algo_name} on {env_name}: {str(e)[:50]}...")
        return None

def plot_and_save(env_name, results):
    """Create simple line plot and save to file."""
    plt.figure(figsize=(10, 6))
    
    for algo, data in results.items():
        if data:
            plt.plot(data["timesteps"], data["rewards"], label=algo, alpha=0.8)
    
    plt.title(f"Algorithm Comparison: {env_name}")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plot_dir, f"compare_on_{env_name}.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Saved plot: {plot_path}")
    
    # Show plot
    # plt.show()
    plt.close()

def main():
    for env_name in environments:
        print(f"\n===== {env_name} =====")
        params = train_params[env_name]
        results = {}
        
        for algo_name in algorithms:
            result_data = train_agent(
                env_name, 
                algo_name, 
                params["policy"], 
                params["steps"],
                params["log_interval"]
            )
            if result_data:
                results[algo_name] = result_data
        
        plot_and_save(env_name, results)

if __name__ == "__main__":
    main()
    