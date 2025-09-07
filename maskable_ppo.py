import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker  # 导入动作掩码包装器
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# define a random mask
def mask_fn(env):
    """generate mask function"""
    # use 0/1 to represent the mask (True/False also works)
    mask = np.ones(2, dtype=bool)
    # randomly mask one action with 30% probability
    if np.random.random() < 0.3:
        action_to_mask = np.random.choice(2)
        mask[action_to_mask] = False
    return mask

# generate the environment with the mask function
def make_masked_env(env_id):
    def _init():
        env = gym.make(env_id)
        # use a action masker wrapper to apply the mask function
        env = ActionMasker(env, mask_fn)
        return env
    return _init

env_id = "CartPole-v1"

env = DummyVecEnv([make_masked_env(env_id)])
eval_env = make_masked_env(env_id)()

# evaluation callback
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/", eval_freq=5000,
    deterministic=True, verbose=1
)

# initialize the MaskablePPO model
model = MaskablePPO(policy="MlpPolicy", env=env, learning_rate=3e-4, n_steps=1024,
    batch_size=64, n_epochs=4, gamma=0.99, verbose=1, device="auto")

# train the model
total_timesteps = 50000
print(f"开始训练，总时间步: {total_timesteps}")
model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

# 保存模型
model.save("maskable_ppo_cartpole")

# 加载最佳模型
best_model = MaskablePPO.load("./logs/best_model")

# 评估模型
mean_reward, std_reward = evaluate_policy(
    best_model,
    eval_env,
    n_eval_episodes=10,
    deterministic=True
)
print(f"评估结果: 平均奖励 = {mean_reward:.2f} ± {std_reward:.2f}")

# 可视化结果
print("开始可视化...")
vis_env = gym.make(env_id, render_mode="human")
vis_env = ActionMasker(vis_env, mask_fn)  # 为可视化环境添加掩码
obs, _ = vis_env.reset()

for _ in range(1000):
    # 获取动作掩码
    action_mask = vis_env.action_masks()
    # 根据掩码选择动作
    action, _ = best_model.predict(obs, action_masks=action_mask, deterministic=True)
    obs, reward, terminated, truncated, info = vis_env.step(action)
    
    if terminated or truncated:
        obs, _ = vis_env.reset()
        print(f"回合结束，奖励: {reward}")

vis_env.close()
    