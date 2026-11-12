import numpy as np
import torch
import gymnasium as gym
from agent import Actor

# env = gym.make("InvertedPendulum-v5", render_mode="human")
env = gym.make("HumanoidStandup-v5", render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.NormalizeObservation(env)
env = gym.wrappers.NormalizeReward(env, gamma=0.99)

n_inputs, n_actions = env.observation_space.shape[0], env.action_space.shape[0]

actor = Actor(n_inputs, 256, n_actions)          # h_size must match training
file_name = "./runs/standup/latest.pt"

actor.load_state_dict(torch.load(file_name, weights_only=False)["actor"])
actor.eval()

for ep in range(5):
    obs, info = env.reset()
    done, ep_return = False, 0.0
    while not done:
        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            a = actor(s).mean.squeeze(0).numpy()   # the mean: no exploration noise
        a = np.clip(a, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(a)
        ep_return += reward
        done = terminated or truncated
    print(f"episode {ep}: return {ep_return:.1f}")
env.close()