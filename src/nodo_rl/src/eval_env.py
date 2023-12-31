import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from MazeEnv import MazeEnv

def evaluate_policy(model, env, n_eval_episodes: int = 100):

    """ Evalúa la política del modelo entrenado """

    total_steps = 0
    total_reward = 0

    for n in range(n_eval_episodes):

        episode_steps = 0
        episode_reward = 0
        done = False
        info = {'Solution': []}

        obs, _ = env.reset()

        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

        print(f"{n + 1}: - Steps: {episode_steps}  - Solution: {info['action_log']} ")
        cv2.destroyAllWindows()
        total_reward += episode_reward
        total_steps += episode_steps

    avg_reward = total_reward / n_eval_episodes
    avg_steps = total_steps / n_eval_episodes
    print(f"\n######### EVALUATION FINISHED #########")
    print(f"- Average reward per episode: {avg_reward}")
    print(f"- Average steps per episode: {avg_steps}")


######## Generate Environment ########

maze_env = MazeEnv(render=False)

######### Check Environment ##########

# while True:
#     check_env(maze_env)

############ Train Model #############

# ppo_model = PPO("MlpPolicy", maze_env, verbose=0)
# ppo_model = PPO.load("ppo_maze_prueba7.model", env=maze_env)
#
# STEPS_PER_BATCH = 200_000
# BATCH_SIZE = 5
# EVAL_EPISODES = 5
#
# for i in range(BATCH_SIZE):
#     print(f"\n training...\n")
#     ppo_model.learn(total_timesteps=STEPS_PER_BATCH)
#     print(f"\n EVALUATION: {i+1}\t TRAINING STEPS: {STEPS_PER_BATCH * (i+1)}\n")
#     evaluate_policy(model=ppo_model, env=MazeEnv(render=True), n_eval_episodes=EVAL_EPISODES)

############ Save Model ##############

# ppo_model.save(f"ppo_maze_prueba8.model")

########## Evaluate Model ############

ppo_model = PPO.load("ppo_maze_prueba7.model", env=maze_env)
evaluate_policy(ppo_model, MazeEnv(render=True), n_eval_episodes=10)

