import gymnasium as gym
import sb3_contrib
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import json
import random
import torch
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from graph_jsp_env.disjunctive_graph_logger import log
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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

feed_random_seeds(42)

# 确保日志目录存在
MONITOR_LOG_DIR = "./monitor_logs"
os.makedirs(MONITOR_LOG_DIR, exist_ok=True)

# 自定义回调类，修复路径问题
class TrainingMetricsCallback(BaseCallback):
    def __init__(self, monitor_log_dir, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.monitor_log_dir = monitor_log_dir  # 显式传入监控日志目录
        self.losses = []
        self.episode_rewards = []
        self.episode_counts = []
        
    def _on_step(self) -> bool:
        # 收集PPO的损失数据（不同版本可能存储在不同位置）
        if 'losses' in self.locals:
            # 对于PPO，损失通常是一个字典包含多种损失
            total_loss = sum(self.locals['losses'].values())
            self.losses.append(total_loss.item())
        return True
        
    def _on_rollout_end(self) -> None:
        # 从正确的监控日志目录加载数据
        try:
            x, y = ts2xy(load_results(self.monitor_log_dir), 'timesteps')
            if len(y) > len(self.episode_rewards):
                # 只添加新的奖励数据
                new_rewards = y[len(self.episode_rewards):]
                self.episode_rewards.extend(new_rewards)
                self.episode_counts.extend(range(
                    len(self.episode_rewards) - len(new_rewards),
                    len(self.episode_rewards)
                ))
        except Exception as e:
            print(f"获取奖励数据时出错: {e}")

def get_tensorboard_logs(base_dir):
    log_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not log_dirs:
        print("未找到日志文件")
        return

    log_path = os.path.join(base_dir, log_dirs[0])
    event_acc = EventAccumulator(log_path).Reload()

    train_results = {}
    for tag in event_acc.Tags()['scalars']:
        train_results['steps'] = [event.step for event in event_acc.Scalars(tag)]
        train_results[tag.split('/')[1]] = [event.value for event in event_acc.Scalars(tag)]
    return train_results

def calculate_rolling_average(rewards, window_size):
    """
    简单的滚动平均计算，只使用基础循环和切片
    rewards: 原始奖励列表
    window_size: 窗口大小
    """
    avg_rewards = []
    # 只有当数据量足够一个窗口时才计算
    if len(rewards) < window_size:
        return avg_rewards
    
    # 从第window_size个数据开始，计算每个窗口的平均值
    for i in range(len(rewards) - window_size + 1):
        # 取当前窗口的奖励（i到i+window_size-1）
        window = rewards[i:i+window_size]
        # 计算平均值并添加到结果
        avg_rewards.append(sum(window) / window_size)
    
    return avg_rewards


# # 定义JSP问题实例
# jsp = np.array([
#     [[1, 2, 0],  # job 0
#      [0, 2, 1]],  # job 1
#     [[17, 12, 19],  # task durations of job 0
#      [8, 6, 2]]  # task durations of job 1
# ])

# # # 定义JSP问题实例
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

# 创建环境，显式指定监控日志目录
env = DisjunctiveGraphJspEnv(
    jps_instance=jsp,
    perform_left_shift_if_possible=True,
    normalize_observation_space=True,
    flat_observation_space=True,
    action_mode='task',
)
# 将监控日志存储在专门的目录
env = Monitor(env, filename=MONITOR_LOG_DIR)

# 动作掩码函数
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()

env = ActionMasker(env, mask_fn)

TENSORBOARD_BASE_DIR = "./tensorboard_logs/"
FIXED_LOG_NAME = "PPO_log_mlp"
TENSORBOARD_LOG_DIR = os.path.join(TENSORBOARD_BASE_DIR, FIXED_LOG_NAME)
os.makedirs(TENSORBOARD_BASE_DIR, exist_ok=True)
if os.path.exists(TENSORBOARD_LOG_DIR):
    log.info(f"remove previous: {TENSORBOARD_LOG_DIR}")
    shutil.rmtree(TENSORBOARD_LOG_DIR)


# 初始化模型
model = sb3_contrib.MaskablePPO(
    MaskableActorCriticPolicy, 
    env, 
    verbose=1,
    tensorboard_log=TENSORBOARD_LOG_DIR,
)

# 创建回调时传入监控日志目录
callback = TrainingMetricsCallback(monitor_log_dir=MONITOR_LOG_DIR, verbose=1)

# 训练代理
log.info("开始训练模型")
model.learn(total_timesteps=20_000, callback=callback)

train_results = get_tensorboard_logs(TENSORBOARD_LOG_DIR)

# 绘制结果
plt.figure(figsize=(14, 6))

# 绘制Loss曲线
plt.subplot(1, 2, 1)
plt.plot(train_results['steps'], train_results['policy_gradient_loss'])
plt.title('Loss')
plt.xlabel('steps')
plt.ylabel('Loss value')
plt.grid(True)

plt.subplot(1, 2, 2)
window_size = 500
rolling_avg_rewards = calculate_rolling_average(callback.episode_rewards, window_size)
plot_steps = callback.episode_counts[window_size-1 : window_size-1 + len(rolling_avg_rewards)]
plt.plot(plot_steps, rolling_avg_rewards)
plt.title('500-Step Rolling Average Reward')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.grid(True, alpha=0.3)
plt.legend()  

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300)  # 提高分辨率保存

json.dump(rolling_avg_rewards, open("training_metrics_mlp.json", "w"))


    