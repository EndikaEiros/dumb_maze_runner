import gymnasium
import matplotlib.pyplot as plt
from gym import spaces
from gymnasium import spaces
from matplotlib import colors
from pylab import *
from time import sleep
import pygame 

# Dimensiones de la ventana y del tablero
WIDTH, HEIGHT = 400, 400
DIMENSION = 8
SQUARE_SIZE = HEIGHT // DIMENSION

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

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

    def render(self):

        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Tablero de Ajedrez')

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            
            self.draw_board(screen, -1)

            for action in self.action_log:
                self.draw_board(screen, action)
                pygame.display.flip()
                pygame.display.flip()
                pygame.display.flip()

            running = False

        pygame.quit()

    def draw_board(self, screen, action):

        if action == -1:
            self.current_state = self.initial_state
        else:
            self.step(action=action)

        for row in range(DIMENSION):
            for col in range(DIMENSION):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        
        for obstacle in self.obstacles:
            pygame.draw.rect(screen, RED, ((obstacle[0])*SQUARE_SIZE, (obstacle[1])*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

        pygame.draw.circle(screen, GREEN, ((self.current_state[0]+0.5)*SQUARE_SIZE, (self.current_state[1]+0.5)*SQUARE_SIZE), (SQUARE_SIZE-8)/2)
        sleep(0.25)


    def render_human(self):

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
