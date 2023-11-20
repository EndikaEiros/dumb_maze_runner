from MazeEnv import MazeEnv
import pygame
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO


env = MazeEnv(initial_state=(2, 2), final_state=(7, 7),
              obstacles=[(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (6, 7), (6, 6), (6, 4), (6, 3)])

# check_env(env)


# Crear y entrenar el modelo PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
#
# Guardar el modelo entrenado
# model.save("ppo_snake.model")

