from MazeEnv import MazeEnv

env = MazeEnv(initial_state=(0, 0), final_state=(7, 7),
              obstacles=[(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (6, 7), (6, 6), (6, 5), (6, 4), (6, 3)])

state, reward, done = env.reset()

dic = {'a': 3, 'w': 0, 's': 1, 'd': 2}

env.render()

while not done:

    action = input("ACTION: \n")
    if action in dic:
        next_state, reward, done = env.step(dic[action])
        env.render()
        state = next_state

print(f"Action log: {env.action_log}")

env.close()
