import json
import random
from random import randint
from random import random
from time import sleep
from collections import deque
import cv2
import gymnasium
from gym import spaces
from gymnasium import spaces
from pylab import *

MAX_STEPS_LIMIT = 64
with open('environment.json', 'r', encoding='utf8') as file:
    DATA = json.load(file)


def random_final_state(obstacles, initial_state):
    # Generar un destino random
    final_state = (randint(0, 7), randint(0, 7))
    while final_state in obstacles or final_state == initial_state:
        final_state = (randint(0, 7), randint(0, 7))
    return final_state


class MazeEnv(gymnasium.Env):

    def __init__(self, render):

        super(MazeEnv, self).__init__()

        # Estado inicial
        self.initial_state = (0, 0)

        # Obstáculos
        self.obstacles = [(elem[0], elem[1]) for elem in DATA['obstacles']]

        # self.final_state = DATA['final_state']
        self.final_state = (7, 7)

        self.action_log = []
        self.state_log = deque([], maxlen=8)

        # Definimos el espacio de acción
        #   0: UP
        #   1: DOWN
        #   2: RIGHT
        #   3: LEFT
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=80,
                                            shape=(4,), dtype=np.float64)

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

        # Recompensa
        self.reward = 0

        if self.render:

            # Draw board (black and white)
            self.img = np.zeros((640, 640, 3), dtype=np.uint8)
            for x in range(0, 9):
                if x % 2 == 0:
                    p = range(0, 8, 2)
                else:
                    p = range(1, 9, 2)
                for y in p:
                    cv2.rectangle(self.img, (x * 80, y * 80), (x * 80 + 80, y * 80 + 80), (255, 255, 255), -1)

            # Draw obstacles (red)
            for obstacle in self.obstacles:
                cv2.rectangle(self.img, (obstacle[0] * 80, obstacle[1] * 80),
                              (obstacle[0] * 80 + 80, obstacle[1] * 80 + 80), (0, 0, 255), -1)

            # Draw goal (yellow)
            cv2.rectangle(self.img, (self.final_state[0] * 80, self.final_state[1] * 80),
                          (self.final_state[0] * 80 + 80, self.final_state[1] * 80 + 80), (0, 255, 255), -1)

    def step(self, action):

        info = {"Solution": None}

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
                cv2.circle(board, (self.current_state[0] * 80 + 40, self.current_state[1] * 80 + 40), 30, (0, 255, 0),
                           -1)
                cv2.imshow('DUMB MAZE RUNNER', board)
                cv2.waitKey(1)
                sleep(0.2)

            if self._is_terminal(self.next_state):

                if self.render:
                    board = np.zeros((640, 640, 3), dtype=np.uint8)
                    cv2.putText(board, f'Win in {self.num_steps} steps)', (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('DUMB MAZE RUNNER', board)
                    cv2.waitKey(1)
                    sleep(1)

                info["Solution"] = self.action_log
                goal_reward = 100
                self.done = True

            ############# Rewards #############

            manhattan_dist_to_goal = 0.001 + np.sum(np.abs(np.array(self.current_state) - np.array(self.final_state)))
            self.reward = (1 / manhattan_dist_to_goal) - (self.num_steps/100) + goal_reward

        else:
            self.reward = -100
            self.truncated = True
            self.done = True

        if self.num_steps >= MAX_STEPS_LIMIT:
            self.done = True
            self.truncated = True
            self.reward = -100

        ######## Nueva observación ########

        obs = np.array([self.current_state[0], self.current_state[1],
                        abs(self.current_state[0] - self.final_state[0]),
                        abs(self.current_state[1] - self.final_state[1])])

        return obs, self.reward, self.done, self.truncated, info

    def reset(self, seed=None):

        # Estado inicial
        self.initial_state = (0, 0)

        # Obstáculos
        obstacles = DATA['obstacles']

        # self.final_state = DATA['final_state']
        self.final_state = (7, 7)

        self.action_log = []
        self.state_log = deque([], maxlen=8)

        # Definimos el espacio de acción
        #   0: UP
        #   1: DOWN
        #   2: RIGHT
        #   3: LEFT
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=80,
                                            shape=(4,), dtype=np.float64)

        # Lista de coordenadas de los obstaculos
        self.obstacles = [(elem[0], elem[1]) for elem in DATA['obstacles']]

        # Fin de la partida
        self.truncated = False
        self.done = False

        # Actual state (fila, columna)
        self.current_state = self.initial_state
        self.next_state = None

        # Número de pasos
        self.num_steps = 0

        # Recompensa
        self.reward = 0

        if self.render:

            # Draw board (black and white)
            self.img = np.zeros((640, 640, 3), dtype=np.uint8)
            for x in range(0, 9):
                if x % 2 == 0:
                    p = range(0, 8, 2)
                else:
                    p = range(1, 9, 2)
                for y in p:
                    cv2.rectangle(self.img, (x * 80, y * 80), (x * 80 + 80, y * 80 + 80), (255, 255, 255), -1)

            # Draw obstacles (red)
            for obstacle in self.obstacles:
                cv2.rectangle(self.img, (obstacle[0] * 80, obstacle[1] * 80),
                              (obstacle[0] * 80 + 80, obstacle[1] * 80 + 80), (0, 0, 255), -1)

            # Draw goal (yellow)
            cv2.rectangle(self.img, (self.final_state[0] * 80, self.final_state[1] * 80),
                          (self.final_state[0] * 80 + 80, self.final_state[1] * 80 + 80), (0, 255, 255), -1)

        obs = np.array([self.current_state[0], self.current_state[1],
                        abs(self.current_state[0] - self.final_state[0]),
                        abs(self.current_state[1] - self.final_state[1])])

        return obs, info

    def _is_posible(self, state):
        return state not in self.obstacles and -1 < state[0] < 8 and -1 < state[1] < 8

    def _is_terminal(self, state):
        return self.final_state[0] == state[0] and self.final_state[1] == state[1]

    def close(self):
        pass
