import gymnasium as gym
import sb3_contrib
import numpy as np
import os
import shutil
import json
from stable_baselines3.common.monitor import Monitor

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from graph_jsp_env.disjunctive_graph_logger import log
from sb3_contrib.common.wrappers import ActionMasker

from util import *
from gcn_mask_ppo import GCNMaskablePolicy

feed_random_seeds(42)
def main():
    MONITOR_LOG_DIR = "./monitor_logs"
    os.makedirs(MONITOR_LOG_DIR, exist_ok=True)
    # define the JSP instance
    jsp = np.array([
        [
            [1, 2, 4, 3, 5, 0],  # job 0
            [0, 3, 4, 2, 5, 1],  # job 1
            [0, 3, 4, 2, 5, 1],  # job 1
        ],  

        [
            [17, 10, 12, 10, 19, 5],  # task durations of job 0
            [8, 6, 8, 2, 20, 5],     # task durations of job 1
            [8, 6, 8, 2, 20, 5],     # task durations of job 1
        ]  
    ])

    avg_rewards = {}
    # types of GNN layers to experiment with
    for layer_type in [
                        "GCNConv", 
                        "SAGEConv", 
                        "GATConv", 
                        "GATv2Conv", 
                        "GINConv", 
                        "TransformerConv"
                    ]:
        
        # create the JSP environment
        env = DisjunctiveGraphJspEnv(
            jps_instance=jsp,
            perform_left_shift_if_possible=True,
            normalize_observation_space=True,
            flat_observation_space=True,
            action_mode='task',
        )
        # save the monitor logs to the specified directory
        env = Monitor(env, filename=MONITOR_LOG_DIR)

        # create the mask function
        def mask_fn(env: gym.Env) -> np.ndarray:
            return env.unwrapped.valid_action_mask()

        # create the environment with action masking
        env = ActionMasker(env, mask_fn)
        
        # set the tensorboard log dir
        TENSORBOARD_BASE_DIR = "./tensorboard_logs/"
        FIXED_LOG_NAME = "PPO_log_gat"
        TENSORBOARD_LOG_DIR = os.path.join(TENSORBOARD_BASE_DIR, FIXED_LOG_NAME)
        os.makedirs(TENSORBOARD_BASE_DIR, exist_ok=True)
        if os.path.exists(TENSORBOARD_LOG_DIR):
            log.info(f"remove previous: {TENSORBOARD_LOG_DIR}")
            shutil.rmtree(TENSORBOARD_LOG_DIR)
        
        # init model
        model = sb3_contrib.MaskablePPO(
            GCNMaskablePolicy,
            env,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            policy_kwargs={
                "net_arch": {
                    "pi": [64, 64],
                    "vf": [64, 64]
                },
                "layer_type": layer_type
            }
        )
        # create callback instance for logging
        callback = TrainingMetricsCallback(monitor_log_dir=MONITOR_LOG_DIR, verbose=1)

        # Train the model
        log.info("开始训练模型")
        model.learn(total_timesteps=20_000, callback=callback)

        train_results = get_tensorboard_logs(TENSORBOARD_LOG_DIR)

        window_size = 500
        rolling_avg_rewards = calculate_rolling_average(callback.episode_rewards, window_size)
        avg_rewards[layer_type] = rolling_avg_rewards


    plot_steps = callback.episode_counts[window_size-1 : window_size-1 + len(rolling_avg_rewards)]
    rolling_avg_rewards_mlp = json.load(open("training_metrics_mlp.json", "r"))
    avg_rewards["MLP"] = rolling_avg_rewards_mlp
    avg_rewards['steps'] = plot_steps
    json.dump(avg_rewards, open("training_metrics_gnn.json", "w"))

if __name__ == "__main__":
    main()
    plot(result_file="training_metrics_gnn.json")

