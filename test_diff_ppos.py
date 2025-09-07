import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
import json

os.environ["QT_QPA_PLATFORM"] = "wayland"


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


# 基础策略网络（适用于离散动作空间）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# 连续动作空间的策略网络（用于SoftAC）
class ContinuousActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, action_scale=1.0, action_bias=0.0):
        super(ContinuousActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # 动作缩放和偏移
        self.action_scale = torch.tensor(action_scale)
        self.action_bias = torch.tensor(action_bias)

        # 限制标准差的范围
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # 重参数化技巧
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # 计算对数概率，考虑tanh的雅可比行列式
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # 缩放动作到环境的动作空间
        action = action * self.action_scale + self.action_bias

        return action, log_prob, mean


# 基础价值网络
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# Q网络（用于SoftAC）
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        # Q1架构
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)

        # Q2架构
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)

        return q1, q2


# 基础Agent类
class BaseAgent:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
        self.gamma = gamma
        self.rewards_history = []

    def select_action(self, state):
        raise NotImplementedError

    def update(self):
        # 基础类不实现具体方法，由子类实现
        raise NotImplementedError

    def train(self, episodes, max_steps=200, verbose=True):
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                done = terminated or truncated

                self.store_transition(state, action, reward, next_state, done)

                if done:
                    break

                state = next_state

            self.update()  # 调用子类实现的update方法
            self.rewards_history.append(total_reward)

            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                print(f"回合: {episode + 1}, 总奖励: {total_reward:.2f}, 最近100回合平均奖励: {avg_reward:.2f}")

        return self.rewards_history

    def store_transition(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards_history)
        plt.title('reward change')
        plt.xlabel('episodes')
        plt.ylabel('total reward')
        plt.grid(True)
        plt.savefig('reward_change.png')


        # 平滑曲线
        window_size = 100
        if len(self.rewards_history) >= window_size:
            smoothed_rewards = np.convolve(self.rewards_history, np.ones(window_size) / window_size, mode='valid')
            plt.figure(figsize=(10, 6))
            plt.plot(smoothed_rewards)
            plt.title(f'reward change, window={window_size}')
            plt.xlabel('')
            plt.ylabel('average reward')
            plt.grid(True)
            plt.savefig('result.png')


# REINFORCE算法
class REINFORCEAgent(BaseAgent):
    def __init__(self, env, gamma=0.99, lr=3e-4, hidden_dim=64):
        super(REINFORCEAgent, self).__init__(env, gamma)

        self.policy = Actor(self.state_dim, self.action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def update(self):
        # 计算折扣奖励
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # 标准化奖励
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # 计算策略损失
        loss = 0
        for state, action, G in zip(self.states, self.actions, returns):
            state = torch.FloatTensor(state).unsqueeze(0)
            logits = self.policy(state)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor(action))
            loss -= log_prob * G

        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空缓存
        self.states = []
        self.actions = []
        self.rewards = []


# Actor-Critic算法
class ACAgent(BaseAgent):
    def __init__(self, env, gamma=0.99, actor_lr=3e-4, critic_lr=1e-3, hidden_dim=64):
        super(ACAgent, self).__init__(env, gamma)

        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim)
        self.critic = Critic(self.state_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        # AC算法在线更新，不需要存储大量轨迹
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        # 计算TD目标
        value = self.critic(state)
        next_value = self.critic(next_state) if not done else torch.tensor([[0.0]], dtype=torch.float32)
        td_target = reward + self.gamma * next_value
        td_error = td_target - value

        # 更新Actor
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(torch.tensor(action))
        actor_loss = -log_prob * td_error.detach()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新Critic
        critic_loss = F.mse_loss(value, td_target.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update(self):
        # AC算法在store_transition中已经在线更新，这里不需要额外操作
        pass

# LSTM Actor网络
class LSTMActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, lstm_hidden=32):
        super(LSTMActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)
        self.fc2 = nn.Linear(lstm_hidden, action_dim)
        self.lstm_hidden = lstm_hidden

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, state_dim) if using sequence, else (batch, state_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加序列维度
            
        x = F.relu(self.fc1(x))
        
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
            
        logits = self.fc2(lstm_out.squeeze(1))  # 移除序列维度
        return logits, hidden

# LSTM Critic网络
class LSTMCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, lstm_hidden=32):
        super(LSTMCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)
        self.fc2 = nn.Linear(lstm_hidden, 1)
        self.lstm_hidden = lstm_hidden

    def forward(self, x, hidden=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加序列维度
            
        x = F.relu(self.fc1(x))
        
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
            
        value = self.fc2(lstm_out.squeeze(1))  # 移除序列维度
        return value, hidden

# 内在好奇心模块(ICM)
class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ICM, self).__init__()
        # 特征提取器
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 反向模型 - 从状态和下一状态预测动作
        self.inverse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 正向模型 - 从状态和动作预测下一状态特征
        self.forward = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, state, next_state, action):
        # 提取特征
        state_feature = self.feature(state)
        next_state_feature = self.feature(next_state)
        
        # 反向模型预测动作
        inverse_input = torch.cat([state_feature, next_state_feature], dim=1)
        pred_action = self.inverse(inverse_input)
        
        # 准备动作的one-hot编码
        action_onehot = F.one_hot(action, num_classes=action.shape[-1]).float()
        
        # 正向模型预测下一状态特征
        forward_input = torch.cat([state_feature, action_onehot], dim=1)
        pred_next_state_feature = self.forward(forward_input)
        
        return pred_action, pred_next_state_feature, next_state_feature, state_feature

class PPOAgent(BaseAgent):
    def __init__(self, env, version="clip", gamma=0.99, lr=3e-4, hidden_dim=64,
                 clip_epsilon=0.2, kl_coef=0.01, K_epochs=10, batch_size=32,
                 gae_lambda=0.95, vf_clip=0.4, lstm_hidden=32, icm_coef=0.1, 
                 sequence_length=10):
        super(PPOAgent, self).__init__(env, gamma)

        # 验证版本是否有效，新增"clip+kl+vf"版本
        valid_versions = ["clip", "kl", "clip+kl", "gae", "vf-clipping", 
                         "2nets", "lstm", "icm", "clip+kl+vf"]
        if version not in valid_versions:
            raise ValueError(f"未知的PPO版本: {version}，可选版本为{valid_versions}")

        self.version = version
        
        # 根据版本初始化不同的网络
        if version == "lstm":
            self.actor = LSTMActor(self.state_dim, self.action_dim, hidden_dim, lstm_hidden)
            self.critic = LSTMCritic(self.state_dim, hidden_dim, lstm_hidden)
            self.sequence_length = sequence_length
            self.hidden_state = None  # LSTM隐藏状态
        elif version == "2nets":
            self.actor = Actor(self.state_dim, self.action_dim, hidden_dim)
            self.critic1 = Critic(self.state_dim, hidden_dim)
            self.critic2 = Critic(self.state_dim, hidden_dim)
        else:
            self.actor = Actor(self.state_dim, self.action_dim, hidden_dim)
            self.critic = Critic(self.state_dim, hidden_dim)
        
        # ICM模块
        if version == "icm":
            self.icm = ICM(self.state_dim, self.action_dim, hidden_dim)
            self.icm_coef = icm_coef
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()) + 
                list(self.icm.parameters()),
                lr=lr
            )
        elif version == "2nets":
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic1.parameters()) + 
                list(self.critic2.parameters()),
                lr=lr
            )
        else:
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=lr
            )

        # 超参数
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda  # GAE参数
        self.vf_clip = vf_clip  # 价值函数裁剪参数

        # 存储轨迹
        self.reset_buffers()

    def reset_buffers(self):
        """重置所有存储的轨迹数据"""
        self.old_logits = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = []
        self.dones = []
        self.next_states = []  # 用于ICM
        self.values = []  # 用于GAE和价值裁剪
        
        # LSTM特殊存储
        if self.version == "lstm":
            self.hidden_states = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        
        if self.version == "lstm":
            logits, self.hidden_state = self.actor(state, self.hidden_state)
            self.hidden_states.append((self.hidden_state[0].detach(), self.hidden_state[1].detach()))
        else:
            logits = self.actor(state)
            
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # 对于需要价值估计的版本，存储价值
        if self.version in ["gae", "vf-clipping", "icm", "clip+kl+vf"]:
            if self.version == "2nets":
                value1 = self.critic1(state)
                value2 = self.critic2(state)
                self.values.append(torch.min(value1, value2).item())
            elif self.version == "lstm":
                value, _ = self.critic(state)
                self.values.append(value.item())
            else:
                value = self.critic(state)
                self.values.append(value.item())

        # 存储旧策略的信息
        self.old_logits.append(logits.detach().squeeze().numpy())
        self.old_log_probs.append(log_prob.item())
        return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def compute_gae(self, last_value):
        """计算广义优势估计(GAE)"""
        advantages = []
        last_advantage = 0
        
        values = self.values + [last_value]
        rewards = self.rewards
        dones = self.dones
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            advantages.insert(0, last_advantage)
            
        returns = [a + v for a, v in zip(advantages, values[:-1])]
        
        return advantages, returns

    def update(self):
        if not self.states:
            return
        
        # 转换为张量
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.old_log_probs)
        old_logits = torch.FloatTensor(self.old_logits)
        next_states = torch.FloatTensor(self.next_states)
        
        # 计算回报和优势
        if self.version == "gae":
            last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0)
            if self.version == "lstm":
                last_value, _ = self.critic(last_state)
                last_value = last_value.item()
            else:
                last_value = self.critic(last_state).item()
            advantages, returns = self.compute_gae(last_value)
            advantages = torch.FloatTensor(advantages)
            returns = torch.FloatTensor(returns)
        else:
            # 标准回报计算
            returns = []
            G = 0
            for r, done in reversed(list(zip(self.rewards, self.dones))):
                if done:
                    G = 0
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns)
            
            # 计算优势
            if self.version in ["2nets"]:
                values1 = self.critic1(states).squeeze()
                values2 = self.critic2(states).squeeze()
                values = torch.min(values1, values2)
            elif self.version == "lstm":
                val, _ = self.critic(states.view(-1, self.state_dim))
                values = val.view(states.shape[0]).squeeze()
            else:
                values = self.critic(states).squeeze()
            advantages = returns - values.detach()
        
        # 优势归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        
        # 处理LSTM的序列数据
        if self.version == "lstm":
            num_sequences = len(states) // self.sequence_length
            states = states[:num_sequences * self.sequence_length].view(
                num_sequences, self.sequence_length, -1)
            actions = actions[:num_sequences * self.sequence_length].view(
                num_sequences, self.sequence_length)
            returns = returns[:num_sequences * self.sequence_length].view(
                num_sequences, self.sequence_length)
            advantages = advantages[:num_sequences * self.sequence_length].view(
                num_sequences, self.sequence_length)
            old_log_probs = old_log_probs[:num_sequences * self.sequence_length].view(
                num_sequences, self.sequence_length)
            old_logits = old_logits[:num_sequences * self.sequence_length].view(
                num_sequences, self.sequence_length, -1)
        
        # 多轮更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0
        
        for epoch in range(self.K_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # 获取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_logits = old_logits[batch_indices]
                
                # 处理LSTM批次
                if self.version == "lstm":
                    batch_size, seq_len, _ = batch_states.shape
                    flat_states = batch_states.view(batch_size * seq_len, -1)
                    logits, _ = self.actor(flat_states)
                    logits = logits.view(batch_size, seq_len, -1)
                    
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    
                    values, _ = self.critic(flat_states)
                    values = values.view(batch_size, seq_len)
                else:
                    # 计算新的策略分布和价值
                    logits = self.actor(batch_states)
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    
                    if self.version == "2nets":
                        values1 = self.critic1(batch_states).squeeze()
                        values2 = self.critic2(batch_states).squeeze()
                    else:
                        values = self.critic(batch_states).squeeze()

                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # 根据版本选择不同的PPO损失计算方式
                if self.version in ["clip", "gae", "vf-clipping", "icm", "lstm"]:
                    # Clip PPO损失
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                elif self.version == "kl":
                    # KL惩罚版本的PPO损失
                    kl_divergence = torch.distributions.kl_divergence(
                        Categorical(logits=batch_old_logits),
                        Categorical(logits=logits)
                    )
                    actor_loss = (kl_divergence / (self.K_epochs - epoch) * self.kl_coef - 
                                 batch_advantages * ratio).mean()

                elif self.version == "clip+kl":
                    # clip + KL混合版本的损失
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    clip_loss = -torch.min(surr1, surr2).mean()

                    kl_divergence = torch.distributions.kl_divergence(
                        Categorical(logits=batch_old_logits),
                        Categorical(logits=logits)
                    ).mean()

                    actor_loss = clip_loss + kl_divergence / (self.K_epochs - epoch) * self.kl_coef
                    
                elif self.version == "2nets":
                    # 双Q网络版本
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                elif self.version == "clip+kl+vf":
                    # 同时包含clip、KL惩罚和价值裁剪的混合版本
                    # 1. 计算clip部分损失
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    clip_loss = -torch.min(surr1, surr2).mean()

                    # 2. 计算KL散度惩罚
                    kl_divergence = torch.distributions.kl_divergence(
                        Categorical(logits=batch_old_logits),
                        Categorical(logits=logits)
                    ).mean()
                    
                    # 3. 组合actor损失
                    actor_loss = clip_loss + kl_divergence / (self.K_epochs - epoch) * self.kl_coef
                    
                else:
                    raise ValueError(f"未知的PPO版本: {self.version}")

                # 计算价值损失
                if self.version == "2nets":
                    critic_loss1 = F.mse_loss(values1, batch_returns)
                    critic_loss2 = F.mse_loss(values2, batch_returns)
                    critic_loss = (critic_loss1 + critic_loss2) / 2
                elif self.version in ["vf-clipping", "clip+kl+vf"]:
                    # 价值函数裁剪
                    old_values = torch.FloatTensor(self.values)
                    batch_old_values = old_values[batch_indices]
                    
                    # 裁剪价值函数更新
                    values_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values, 
                        -self.vf_clip, 
                        self.vf_clip
                    )
                    
                    # 计算裁剪和未裁剪的损失，取最大值
                    loss_unclipped = F.mse_loss(values, batch_returns)
                    loss_clipped = F.mse_loss(values_clipped, batch_returns)
                    critic_loss = torch.max(loss_unclipped, loss_clipped).mean()
                else:
                    # 标准MSE损失
                    critic_loss = F.mse_loss(values, batch_returns)

                # 计算ICM损失
                icm_loss = 0
                if self.version == "icm":
                    pred_action, pred_next_feat, next_feat, curr_feat = self.icm(
                        batch_states, next_states[batch_indices], batch_actions.unsqueeze(1))
                    
                    inverse_loss = F.cross_entropy(pred_action, batch_actions)
                    forward_loss = 0.5 * F.mse_loss(pred_next_feat, next_feat.detach())
                    icm_loss = inverse_loss + forward_loss

                # 总损失
                total_loss_batch = actor_loss + 0.5 * critic_loss
                if self.version == "icm":
                    total_loss_batch += self.icm_coef * icm_loss

                # 梯度下降
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
                
                # 累计损失
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_loss += total_loss_batch.item()

        # 清空缓存
        self.reset_buffers()
        
        # 重置LSTM隐藏状态
        if self.version == "lstm":
            self.hidden_state = None
            
        # 返回平均损失
        return {
            "actor_loss": total_actor_loss / (self.K_epochs * (len(states) // self.batch_size + 1)),
            "critic_loss": total_critic_loss / (self.K_epochs * (len(states) // self.batch_size + 1)),
            "total_loss": total_loss / (self.K_epochs * (len(states) // self.batch_size + 1))
        }
     

# Soft Actor-Critic算法
class SoftACAgent(BaseAgent):
    def __init__(self, env, gamma=0.99, tau=0.005,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 hidden_dim=64, buffer_capacity=1000000, batch_size=256,
                 target_entropy=None):
        super(SoftACAgent, self).__init__(env, gamma)

        # 处理离散动作空间（CartPole）
        self.is_discrete = hasattr(env.action_space, 'n')
        if self.is_discrete:
            self.action_dim = env.action_space.n
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_dim = env.action_space.shape[0]
            self.action_scale = (env.action_space.high - env.action_space.low) / 2.0
            self.action_bias = (env.action_space.high + env.action_space.low) / 2.0

        # 构建网络
        self.actor = ContinuousActor(self.state_dim, 1 if self.is_discrete else self.action_dim,
                                     hidden_dim, self.action_scale, self.action_bias)
        self.critic = QNetwork(self.state_dim, 1 if self.is_discrete else self.action_dim, hidden_dim)
        self.critic_target = QNetwork(self.state_dim, 1 if self.is_discrete else self.action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 自动调整熵的温度参数
        if target_entropy is None:
            if self.is_discrete:
                self.target_entropy = -np.log(1.0 / self.action_dim) * 0.98
            else:
                self.target_entropy = -self.action_dim
        else:
            self.target_entropy = target_entropy

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().item()

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.tau = tau  # 目标网络软更新参数

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _, mean = self.actor.sample(state)
            if deterministic:
                action = torch.tanh(mean) * self.action_scale + self.action_bias

        action = action.item()

        # 对于离散动作空间，将连续输出转换为离散动作
        if self.is_discrete:
            action = 0 if action < 0.5 else 1

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从缓冲区采样
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(
            self.batch_size)

        # 转换为张量
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1)

        # 计算目标Q值
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state_batch)

            # 处理离散动作
            if self.is_discrete:
                next_action_discrete = (next_action > 0.5).float()
                q1_target, q2_target = self.critic_target(next_state_batch, next_action_discrete)
            else:
                q1_target, q2_target = self.critic_target(next_state_batch, next_action)

            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            target = reward_batch + (1 - done_batch) * self.gamma * q_target

        # 计算当前Q值并更新Critic
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 冻结Critic参数，更新Actor
        for param in self.critic.parameters():
            param.requires_grad = False

        # 计算Actor损失
        action, log_prob, _ = self.actor.sample(state_batch)

        if self.is_discrete:
            action_discrete = (action > 0.5).float()
            q1, q2 = self.critic(state_batch, action_discrete)
        else:
            q1, q2 = self.critic(state_batch, action)

        q_min = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 解冻Critic参数
        for param in self.critic.parameters():
            param.requires_grad = True

        # 更新温度参数alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # 软更新目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


# 比较不同算法的性能
def compare_algorithms(env_name="CartPole-v1", episodes=400, max_steps=100):
    env = gym.make(env_name)

    valid_versions = ["clip", "kl", "clip+kl", "gae", "vf-clipping", 'clip+kl+vf'
                       #  "2nets", "icm}, "lstm", 
                       ]
    

    agents = {"AC": ACAgent(env)}
    for version in valid_versions:
        agents[f"PPO {version}"] = PPOAgent(env, version=version)

    # 训练每个智能体
    results = {}
    for name, agent in agents.items():
        print(f"开始训练 {name} ...")
        start_time = time.time()
        rewards = agent.train(episodes, max_steps, verbose=True)
        end_time = time.time()
        results[name] = rewards
        print(f"{name} 训练完成，耗时 {end_time - start_time:.2f} 秒")
        print(f"{name} 最终平均奖励: {np.mean(rewards[-100:]):.2f}")
    
    # save results to a json file
    with open('results.json', 'w') as f:
        json.dump(results, f)
    return results


def plot_results(results, window_size=100):
    plt.figure(figsize=(12, 8))

    plt.rcParams.update({
    'font.size': 14})
    window_size = 100
    lengends =  ["actor-clip", "kl", "actor-clip + kl", "gae", "critic-clip", 'actor-clip + kl + critic-clip']
    
    for name, rewards in results.items():
        if name  == 'PPO clip':
            name = 'PPO: actor-clip'
        elif name  == 'PPO kl':
            name = 'PPO: KL'
        elif name  == 'PPO clip+kl':
            name = 'PPO: actor-clip + KL'
        elif name  == 'PPO gae':
            name = 'PPO: GAE'
        elif name  == 'PPO vf-clipping':
            name = 'PPO: critic-clip'
        elif name  == 'PPO clip+kl+vf':
            name = 'PPO: actor-clip + KL + critic-clip'
        elif name  == 'AC':
            name = 'Actor-Critic'
        if len(rewards) >= window_size:
            smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
            plt.plot(smoothed, label=name, linewidth=2.)
        else:
            plt.plot(rewards, label=name, linewidth=2.)

    # plt.title('compare')
    plt.xlabel('episodes')
    plt.ylabel(f'total reward, window={window_size}')
    # 将图例放在底部中央，设置2列（可根据实际图例数量调整）
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.21), fontsize=12)
    plt.fontsize = 24
    plt.grid(True)
    # 调整布局避免图例被截断
    plt.tight_layout()
    plt.savefig('ppo_compare.png', bbox_inches='tight')

    # 打印最终性能
    print("\n最终性能（最后100回合平均奖励）：")
    for name, rewards in results.items():
        if len(rewards) >= 100:
            final_avg = np.mean(rewards[-100:])
            print(f"{name}: {final_avg:.2f}")

    return results



# 主函数
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # results = compare_algorithms(episodes=2000, max_steps=500)
    results = json.load(open('ppo_results.json'))
    plot_results(results)
