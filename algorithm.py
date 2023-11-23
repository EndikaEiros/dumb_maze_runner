from MazeEnv import MazeEnv
import pygame
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO


def evaluate_policy(model, env, n_eval_episodes: int = 100):
    total_steps = 0
    total_reward = 0
    wins = 0

    for n in range(n_eval_episodes):

        episode_steps = 0
        episode_reward = 0
        done = False

        obs, _ = env.reset()

        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

        if not truncated: wins += 1

        print(f" - Episode: {n + 1}\t - Reward: {episode_reward}\t - Steps: {episode_steps}")
        total_reward += episode_reward
        total_steps += episode_steps

    avg_reward = total_reward / n_eval_episodes
    avg_steps = total_steps / n_eval_episodes
    print(f"\n######### EVALUATION FINISHED #########")
    print(f"- Average reward per episode: {avg_reward}")
    print(f"- Average steps per episode: {avg_steps}")

    return avg_reward, avg_steps



env = MazeEnv(initial_state=(2, 2), final_state=(-1, -1),
              obstacles=[(1,1), (1,2), (2,1), (4,1), (5,1), (4,3), (5,3), (6,3), (1,6), (2,6), (3,6), (4,6), (5,6), (7,6), (2,4)],
              render=False)

render_env = MazeEnv(initial_state=(2, 2), final_state=(-1, -1),
              obstacles=[(1,1), (1,2), (2,1), (4,1), (5,1), (4,3), (5,3), (6,3), (1,6), (2,6), (3,6), (4,6), (5,6), (7,6), (2,4)],
              render=True)

# check_env(env)


# Crear y entrenar el modelo PPO
model = PPO("MlpPolicy", env, verbose=0)

for i in range(10):
    model.learn(total_timesteps=2)
    model.save(f"ppo_snake_{i}.model")
    rew, steps = evaluate_policy(model=model, env=render_env)



