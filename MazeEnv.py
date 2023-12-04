import random
from collections import deque
from random import randint
from time import sleep

import cv2
import gymnasium
from gym import spaces
from gymnasium import spaces
from pylab import *

MAX_STEPS_LIMIT = 64
NUM_OBSTACLES = 5
GOAL_REWARD = 10
DIED_REWARD = -10


def random_final_state(initial_state):
    # Generar un destino random
    final_state = (randint(5, 7), randint(5, 7))
    while final_state == initial_state:
        final_state = (randint(5, 7), randint(5, 7))
    return final_state


def random_obstacles(num_obstacles, initial_state, final_state):
    obstacles = set()
    while len(obstacles) < num_obstacles:
        new_obstacle = (randint(0, 7), randint(0, 7))
        if new_obstacle != initial_state and new_obstacle != final_state:
            obstacles.add(new_obstacle)
    return list(obstacles)


class MazeEnv(gymnasium.Env):

    def __init__(self, render=False):

        super(MazeEnv, self).__init__()

        # Estado inicial
        self.initial_state = (np.random.choice([0, 7]), np.random.choice([7, 0]))

        self.final_state = random_final_state(self.initial_state)

        # Obstáculos
        self.obstacles = random_obstacles(NUM_OBSTACLES, self.initial_state, self.final_state)

        self.action_log = []
        self.state_log = deque([], maxlen=8)

        # Definimos el espacio de acción
        #   0: UP
        #   1: DOWN
        #   2: RIGHT
        #   3: LEFT
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=80, shape=(6,), dtype=np.float64)

        # Visualización del juego
        self.render = render

        # Fin de la partida
        self.truncated = False
        self.done = False

        # Actual state (fila, columna)
        self.current_state = self.initial_state
        self.next_state = None

        # Número de pasos
        self.num_steps = 0

        # Recompensas
        self.reward, self.prev_reward = 0, 0

        if self.render:

            # Draw board (black and white)
            self.img = np.zeros((640, 640, 3), dtype=np.uint8)
            for x in range(0, 9):
                if x % 2 == 0:
                    p = range(0, 8, 2)
                else:
                    p = range(1, 9, 2)
                for y in p:
                    cv2.rectangle(self.img, (x * 80, y * 80), (x * 80 + 80, y * 80 + 80), (50, 50, 50), -1)

            # Draw obstacles (red)
            for obstacle in self.obstacles:
                cv2.rectangle(self.img, (obstacle[0] * 80, obstacle[1] * 80),
                              (obstacle[0] * 80 + 80, obstacle[1] * 80 + 80), (0, 0, 255), -1)

            # Draw goal (yellow)
            cv2.rectangle(self.img, (self.final_state[0] * 80, self.final_state[1] * 80),
                          (self.final_state[0] * 80 + 80, self.final_state[1] * 80 + 80), (0, 255, 255), -1)

    def step(self, action):

        info = {"Solution": []}

        ############## Mover ##############

        if action == 0: self.next_state = (self.current_state[0] - 1, self.current_state[1])
        if action == 1: self.next_state = (self.current_state[0] + 1, self.current_state[1])
        if action == 2: self.next_state = (self.current_state[0], self.current_state[1] + 1)
        if action == 3: self.next_state = (self.current_state[0], self.current_state[1] - 1)

        if self._is_posible(self.next_state):

            self.action_log.append(action)
            self.current_state = self.next_state
            self.num_steps += 1
            goal_reward = 0

            ########## Visualización ##########

            if self.render:
                board = self.img.copy()
                cv2.circle(board, (self.current_state[0] * 80 + 40, self.current_state[1] * 80 + 40),
                           30, (0, 255, 0), -1)
                cv2.imshow('DUMB MAZE RUNNER', board)
                cv2.waitKey(1)
                sleep(0.33)

            if self._is_terminal(self.next_state):

                if self.render:
                    board = np.zeros((640, 640, 3), dtype=np.uint8)
                    cv2.putText(board, f'Win in {self.num_steps} steps', (200, 320), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('DUMB MAZE RUNNER', board)
                    cv2.waitKey(1)
                    sleep(0.33)

                info["Solution"] = self.action_log
                goal_reward = GOAL_REWARD
                self.done = True

            ############# Rewards #############

            self.reward = goal_reward

            # manhattan_dist_to_goal = np.sum(np.abs(np.array(self.current_state) - np.array(self.final_state)))
            # if goal_reward == 0: self.reward = 1 / manhattan_dist_to_goal

            if self.num_steps >= MAX_STEPS_LIMIT:
                self.done = True
                self.truncated = True
                self.reward = 0

        else:

            self.reward = DIED_REWARD
            self.truncated = True
            self.done = True

        ######################### Return #########################

        pos_x = self.current_state[0]
        pos_y = self.current_state[1]

        # Indicar la dirección del destino
        final_x = self.final_state[0] - pos_x > 0
        final_y = self.final_state[1] - pos_y > 0

        # Indicar si muere tras alguna acción
        up = self._is_posible([pos_x, pos_y - 1])
        down = self._is_posible([pos_x, pos_y + 1])
        right = self._is_posible([pos_x + 1, pos_y])
        left = self._is_posible([pos_x - 1, pos_y])

        observation = [final_x, final_y, up, down, right, left]
        observation = np.array(observation)

        return observation, self.reward, self.done, self.truncated, info

    def reset(self, seed=None):

        self.action_log = []
        self.state_log = deque([], maxlen=8)

        self.initial_state = (np.random.choice([0, 7]), np.random.choice([7, 0]))

        self.final_state = random_final_state(self.initial_state)

        self.obstacles = random_obstacles(NUM_OBSTACLES, self.initial_state, self.final_state)

        # Fin de la partida
        self.truncated = False
        self.done = False

        # Actual state (fila, columna)
        self.current_state = self.initial_state
        self.next_state = None

        # Número de pasos
        self.num_steps = 0

        # Recompensas
        self.reward, self.prev_reward = 0, 0

        if self.render:

            # Draw board (black and white)
            self.img = np.zeros((640, 640, 3), dtype=np.uint8)
            for x in range(0, 9):
                if x % 2 == 0:
                    p = range(0, 8, 2)
                else:
                    p = range(1, 9, 2)
                for y in p:
                    cv2.rectangle(self.img, (x * 80, y * 80), (x * 80 + 80, y * 80 + 80), (50, 50, 50), -1)

            # Draw obstacles (red)
            for obstacle in self.obstacles:
                cv2.rectangle(self.img, (obstacle[0] * 80, obstacle[1] * 80),
                              (obstacle[0] * 80 + 80, obstacle[1] * 80 + 80), (0, 0, 255), -1)

            # Draw goal (yellow)
            cv2.rectangle(self.img, (self.final_state[0] * 80, self.final_state[1] * 80),
                          (self.final_state[0] * 80 + 80, self.final_state[1] * 80 + 80), (0, 255, 255), -1)

        ######################### Return #########################

        pos_x = self.current_state[0]
        pos_y = self.current_state[1]

        # Indicar la dirección del destino
        final_x = self.final_state[0] - pos_x > 0
        final_y = self.final_state[1] - pos_y > 0

        # Indicar si muere tras alguna acción
        up = [pos_x, pos_y - 1] in self.obstacles or pos_y == 0
        down = [pos_x, pos_y + 1] in self.obstacles or pos_y == 7
        right = [pos_x + 1, pos_y] in self.obstacles or pos_x == 7
        left = [pos_x - 1, pos_y] in self.obstacles or pos_x == 0

        observation = [final_x, final_y, up, down, right, left]
        observation = np.array(observation)

        return observation, {}

    def _is_posible(self, state):
        return state not in self.obstacles and -1 < state[0] < 8 and -1 < state[1] < 8

    def _is_terminal(self, state):
        return self.final_state[0] == state[0] and self.final_state[1] == state[1]

    def close(self):
        pass
