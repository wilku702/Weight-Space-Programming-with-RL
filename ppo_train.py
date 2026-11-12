import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
import torch.nn.functional as F
from agent import *
import os

def update_epoch ( actor, critic, old_critic, actor_optim, critic_optim, cur_states, actions, advantages, old_logp, epsilon = 0.2 ):
    cur_values = critic(cur_states).squeeze(-1)
    # TD target
    with torch.no_grad():
        old_values = old_critic(cur_states).squeeze(-1)
        target = advantages + old_values

    # Critic loss
    critic_loss = ( target - cur_values ).pow(2).mean()
    
    # Actor loss
    new_dist = actor(cur_states) # Normal, (64, 6)
    new_logp = new_dist.log_prob(actions).sum(-1) # (64,)
    ratio = torch.exp( new_logp - old_logp ) 
    
    # Clip the gradient?
    norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    surrogate_loss = ratio * norm_advantages
    clipped_loss = torch.clip( ratio, 1 - epsilon, 1 + epsilon ) * norm_advantages
    actor_loss = -torch.min(clipped_loss, surrogate_loss).mean()
    
    actor_optim.zero_grad()
    critic_optim.zero_grad()
    critic_loss.backward()
    actor_loss.backward()
    actor_optim.step()
    critic_optim.step()
    

def compute_advantages( value_cur, value_next, buf_reward, buf_term, buf_trun, gamma = 0.99, lambda_ = 0.95 ):
    n = len(buf_reward)
    advantages = [0] * n
    # Calculate the advantages backwards
    A_next = 0
    for i in range( n-1, -1, -1 ):
        v_cur, v_next = value_cur[i], value_next[i]
        delta = (buf_reward[i] + gamma * v_next * (1 - buf_term[i])) - v_cur
        A_t = delta + gamma * lambda_ * A_next * (1 - (buf_term[i] | buf_trun[i]))
        advantages[i] = A_t
        A_next = A_t
    return advantages


if __name__ == '__main__':
    env = gym.make('HumanoidStandup-v5')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=0.99)
    
    # env = gym.make('InvertedPendulum-v5')
    observation, info = env.reset()
    n_inputs, n_actions = env.observation_space.shape[0], env.action_space.shape[0]
    h_size = 256
    
    # Init Network
    actor = Actor( n_inputs, h_size, n_actions )
    critic = Critic( n_inputs, h_size )
    
    path = '../runs/standup/'
    file_name = 'latest.pt'
    
    old_critic = Critic( n_inputs, h_size )
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    
    if os.path.exists(path + file_name):
        checkpoint = torch.load(path + file_name, weights_only=False)
        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
        actor_optimizer.load_state_dict(checkpoint['actor_optim'])
        critic_optimizer.load_state_dict(checkpoint['critic_optim'])
        best_avg = checkpoint['best_avg']
    else:
        best_avg = -float('inf')
    
    # Configs
    total_steps = int(3e6)
    gamma = 0.99
    
    # Calculate returns
    episode_return = 0.0
    returns = []
    N = 2048
    epochs = 10
    samples = 64
    
    buf_state, buf_action, buf_next_state, buf_reward, buf_term, buf_trun, buf_logp = [], [], [], [], [], [], []
    
    for step in range(total_steps):
        state_tensor = torch.as_tensor(observation, dtype=torch.float32)
        
        with torch.no_grad():
            dist = actor(state_tensor)
            action = dist.sample() #(6,)
            old_logp = dist.log_prob(action).sum(-1)
        env_action = np.clip(action.numpy(),env.action_space.low, env.action_space.high) # Clip action
        next_state, reward, terminated, truncated, info = env.step(env_action)
        episode_return += reward 
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)
        
        buf_state.append(state_tensor)
        buf_action.append(action)
        buf_logp.append(old_logp)
        buf_next_state.append(next_state_tensor)
        buf_reward.append(reward)
        buf_term.append(terminated)
        buf_trun.append(truncated)
        
        if len(buf_state) >= N:
            # Calcualte the advantages and tranform into tensors
            cur_states_t = torch.stack(buf_state)
            actions_t = torch.stack(buf_action)
            next_states_t = torch.stack(buf_next_state)
            rewards_t = torch.tensor(buf_reward, dtype=torch.float32)
            terminated_t = torch.tensor(buf_term, dtype=torch.float32)
            
            with torch.no_grad():
                value_cur  = critic(cur_states_t).squeeze(-1)    # (N, 1) -> (N,)
                value_next = critic(next_states_t).squeeze(-1)   # (N,)
            buf_advantages = compute_advantages( value_cur, value_next, buf_reward, buf_term, buf_trun )
            advantages_t = torch.stack(buf_advantages)
            old_logp_t = torch.stack(buf_logp)
            
            old_critic.load_state_dict(critic.state_dict()) # old critic network
            
            for epoch in range(epochs):
                idx = torch.randperm(N)
                for start in range(0, N, samples):
                    mb = idx[start:start + samples] # mini_batch
                    update_epoch( actor, critic, old_critic, actor_optimizer, critic_optimizer, cur_states_t[mb], actions_t[mb], advantages_t[mb], old_logp_t[mb] )
            
            # Reset
            buf_state, buf_action, buf_next_state, buf_reward, buf_term, buf_trun, buf_logp = [], [], [], [], [], [], []
        
        if terminated or truncated:
            returns.append(episode_return)
            episode_return = 0.0
            if len(returns) % 10 == 0:
                avg = np.mean(returns[-10:])
                print(f"step {step}  episode {len(returns)}  avg return (last 10): {avg:.1f}")
                ckpt = {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "actor_optim": actor_optimizer.state_dict(),
                    "critic_optim": critic_optimizer.state_dict(),
                    "step": step,
                    "best_avg": best_avg,
                }
                torch.save(ckpt, path + file_name) 
                if avg > best_avg:
                    best_avg = avg
                    ckpt["best_avg"] = best_avg
                    torch.save(ckpt, path + 'best.pt') 
            observation, info = env.reset()
        else:
            observation = next_state