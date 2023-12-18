from collections import deque
from random import randint
from time import sleep

import cv2
import gymnasium
import numpy as np
from gym import spaces
from gymnasium import spaces

MAX_STEPS_LIMIT = 100
# NUM_OBSTACLES = 8
GOAL_REWARD = 10
DIED_REWARD = -10
FPS = 15


class MazeEnv(gymnasium.Env):

    def __init__(self, render=False, obstacles=set()):

        """ Inicialización del entorno """

        super(MazeEnv, self).__init__()

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-80, high=80, shape=(6,), dtype=np.float64)

        # Initial state
        # self.initial_state = (np.random.choice([0, 7]), np.random.choice([7, 0]))
        self.initial_state = (0, 7)

        # Final state
        # self.final_state = self._get_random_final_state(self.initial_state)
        self.final_state = (7, 0)

        # Manhattan Distance between init and goal
        self.initial_dist_to_goal = np.sum(np.abs(np.array(self.initial_state) - np.array(self.final_state)))

        # Obstacles
        # self.obstacles = self._get_random_obstacles(NUM_OBSTACLES, self.initial_state, self.final_state)
        self.obstacles = obstacles

        # Actions history
        self.action_log = []

        # Visualize game
        self.render = render
        self.letter_color = (255, 144, 30)

        # Game Over
        self.truncated = False
        self.done = False

        # Current state
        self.current_state = list(self.initial_state)

        # Reward and number of steps
        self.reward, self.num_steps = 0, 0

        # Additional information
        self.info = {"current_state": self.current_state}

        if self.render:
            # Generate board
            self.img = np.zeros((640, 640, 3), dtype=np.uint8)
            self._generate_board()

    """ Realizar un paso sobre el entorno """

    def step(self, action):

        ############## Move ###############

        if action == 0:  # UP
            self.current_state[1] -= 1
        if action == 1:  # DOWN
            self.current_state[1] += 1
        if action == 2:  # RIGHT
            self.current_state[0] += 1
        if action == 3:  # LEFT
            self.current_state[0] -= 1

        self.action_log.append(action)
        self.info["current_state"] = self.current_state
        self.num_steps += 1

        ############# If died #############

        if self._is_dead(self.current_state) or self.num_steps >= MAX_STEPS_LIMIT:

            self.info["num_steps"] = self.num_steps
            self.info["action_log"] = []
            self.letter_color = (0, 0, 255)
            self.reward = DIED_REWARD
            self.truncated = True
            self.done = True

        ############# if win ##############

        elif self._has_won(self.current_state):

            self.info["num_steps"] = self.num_steps
            self.info["action_log"] = self.action_log
            msg = f"Solved in {self.num_steps} steps"
            self.letter_color = (0, 255, 0)
            self.reward = GOAL_REWARD / self.num_steps
            self.done = True

        ############# Rewards #############

        else:
            # manhattan_dist_to_goal = np.sum(np.abs(np.array(self.current_state) - np.array(self.final_state)))
            # if goal_reward == 0: self.reward = 1 / manhattan_dist_to_goal
            self.reward = 0

        ########## Visualization ##########

        if self.render:

            board = self.img.copy()
            cv2.circle(board, (self.current_state[0] * 80 + 40, self.current_state[1] * 80 + 40),
                       25, (0, 150, 255), -1)
            cv2.putText(board, f"Steps: {self.num_steps}/{self.initial_dist_to_goal}", (20, 620),
                        cv2.FONT_HERSHEY_PLAIN, 2, self.letter_color, 1, cv2.LINE_AA)
            if self.done:
                cv2.putText(board, 'GAME OVER', (140, 340), cv2.FONT_HERSHEY_SIMPLEX,
                            2, self.letter_color, 2, cv2.LINE_AA)
                sleep_time = 1
            else:
                sleep_time = 1 / FPS
            cv2.imshow('DUMB MAZE RUNNER', board)
            cv2.waitKey(1)
            sleep(sleep_time)

        ############# Return ##############

        observation = self._get_obs()

        return observation, self.reward, self.done, self.truncated, self.info


    def reset(self, seed=None):

        """ Restaurar el entorno para empezar un nuevo episodio """

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-80, high=80, shape=(6,), dtype=np.float64)

        # Initial state
        # self.initial_state = (np.random.choice([0, 7]), np.random.choice([7, 0]))
        self.initial_state = (0, 7)

        # Final state
        # self.final_state = self._get_random_final_state(self.initial_state)
        self.final_state = (7, 0)

        # Manhattan Distance between init and goal
        self.initial_dist_to_goal = np.sum(np.abs(np.array(self.initial_state) - np.array(self.final_state)))

        # Obstacles
        # self.obstacles = self._get_random_obstacles(NUM_OBSTACLES, self.initial_state, self.final_state)

        # Actions history
        self.action_log = []

        # Visualize game
        self.letter_color = (255, 144, 30)

        # Game Over
        self.truncated = False
        self.done = False

        # Current state
        self.current_state = list(self.initial_state)

        # Reward and number of steps
        self.reward, self.num_steps = 0, 0

        # Aditional information
        self.info = {"current_state": self.current_state}

        if self.render:
            # Generate board
            self.img = np.zeros((640, 640, 3), dtype=np.uint8)
            self._generate_board()

        ############# Return ##############

        observation = self._get_obs()

        return observation, self.info


    def _is_dead(self, state):

        """ Dado un estado devuelve True si este es un estado terminal (muerte) """

        # returns true if current state is a terminal state
        return tuple(state) in self.obstacles or -1 in state or 8 in state


    def _has_won(self, state):

        """ Dado un estado devuelve True si es el destino (win) """

        # returns true if current state is the goal
        return self.final_state == tuple(state)


    def _get_obs(self):

        """ Genera la observación del estado actual """

        pos_x = self.current_state[0]
        pos_y = self.current_state[1]

        # Direction to the goal
        final_x = self.final_state[0] - pos_x > 0
        final_y = self.final_state[1] - pos_y > 0

        # If there are elements around
        up = self._is_dead([pos_x, pos_y - 1])
        down = self._is_dead([pos_x, pos_y + 1])
        right = self._is_dead([pos_x + 1, pos_y])
        left = self._is_dead([pos_x - 1, pos_y])

        # returns the observation of current state
        return np.array([final_x, final_y, up, down, right, left])


    def _generate_board(self):

        """ Genera el tablero para la visualización colocando obstáculos y destino """

        # Draw chess board (black and white)
        for x in range(0, 9):
            if x % 2 == 0:
                p = range(0, 8, 2)
            else:
                p = range(1, 9, 2)
            for y in p:
                cv2.rectangle(self.img, (x * 80, y * 80), (x * 80 + 80, y * 80 + 80), (40, 40, 40), -1)

        # Draw obstacles (red)
        for obstacle in self.obstacles:
            cv2.rectangle(self.img, (obstacle[0] * 80 + 2, obstacle[1] * 80 + 2),
                          (obstacle[0] * 80 + 80 - 2, obstacle[1] * 80 + 80 - 2), (0, 0, 100), -1)

        # Draw goal (green)
        cv2.rectangle(self.img, (self.final_state[0] * 80 + 2, self.final_state[1] * 80 + 2),
                      (self.final_state[0] * 80 + 80 - 2, self.final_state[1] * 80 + 80 - 2), (0, 100, 0), -1)

    def _get_random_final_state(self, initial_state):

        """ Gerenación del destino aleatoria """

        final_state = (randint(0, 7), randint(0, 7))
        while final_state == initial_state:
            final_state = (randint(0, 7), randint(0, 7))
        return final_state

    def _get_random_obstacles(self, num_obstacles, initial_state, final_state):

        """ Generación de {NUM_OSTABLES} obstáculos aleatorios """

        obstacles = set()
        while len(obstacles) < num_obstacles:
            new_obstacle = (randint(0, 7), randint(0, 7))
            if new_obstacle != initial_state and new_obstacle != final_state:
                obstacles.add(new_obstacle)
        return list(obstacles)

    def close(self):
        pass
