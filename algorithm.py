from MazeEnv import MazeEnv
import pygame

env = MazeEnv(initial_state=(0, 0), final_state=(7, 7),
              obstacles=[(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (6, 7), (6, 6), (6, 5), (6, 4), (6, 3)])

state, reward, done = env.reset()

dic = {'a': 3, 'w': 0, 's': 1, 'd': 2}

env.render_human()

while not done:

    action = input()
    if action in dic:
        next_state, reward, done = env.step(dic[action])
        env.render_human()
        state = next_state
#env.action_log = [2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2]
print(f"Action log: {env.action_log}")
env.render()

env.close()
