import gymnasium
import matplotlib.pyplot as plt
from gym import spaces
from gymnasium import spaces
from matplotlib import colors
from pylab import *

class MazeEnv(gymnasium.Env):

    def __init__(self, initial_state, final_state, obstacles):

        super(MazeEnv, self).__init__()
        self.action_log = []

        # Definimos el espacio de acción
        #   0: UP
        #   1: DOWN
        #   2: RIGHT
        #   3: LEFT
        self.action_space = spaces.Discrete(4)

        self.initial_state = initial_state
        self.final_state = final_state

        # Duración de la ducha en segundos
        self.max_steps = 64

        self.obstacles = obstacles

        # Contador de tiempo
        self.current_step = 0

        # Actual state (fila, columna)
        self.current_state = initial_state
        self.next_state = (0, 0)

        self.reward = 0

    def is_posible(self):

        return self.next_state not in self.obstacles and -1 < self.next_state[0] < 8 and -1 < self.next_state[1] < 8

    def is_terminal(self):

        return self.final_state == self.current_state

    def step(self, action):

        if action == 0: self.next_state = (self.current_state[0] - 1, self.current_state[1])
        if action == 1: self.next_state = (self.current_state[0] + 1, self.current_state[1])
        if action == 2: self.next_state = (self.current_state[0], self.current_state[1] + 1)
        if action == 3: self.next_state = (self.current_state[0], self.current_state[1] - 1)

        if self.is_posible():

            self.action_log.append(action)
            self.current_state = self.next_state
            self.current_step += 1

        else:
            print('ILEGAL ACTION\t')

        if self.is_terminal():
            self.reward = 100

        return self.current_state, self.reward, self.is_terminal()

    def reset(self):

        self.current_step = 0

        self.current_state = (0, 0)

        self.reward = 0

        self.action_log = []

        return self.current_state, self.reward, self.is_terminal()

    def render(self, mode='human'):

        tablero = [
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
        ]
        for obstacle in self.obstacles:
            tablero[obstacle[0]][obstacle[1]] = 2
        tablero[self.current_state[0]][self.current_state[1]] = 3
        tablero[self.final_state[0]][self.final_state[1]] = 4

        # Close all windows
        plt.close('all')

        # Set colours
        cmap = colors.ListedColormap(['white', 'black', 'red', 'green', 'blue'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        # Create figure
        plt.figure(figsize=(8, 8))
        plt.imshow(tablero, cmap=cmap, norm=norm)
        plt.xticks(range(8), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        plt.yticks(range(8), range(1, 9))

        # Show
        plt.ion()
        plt.show()
        plt.pause(.001)

    def close(self):
        pass
