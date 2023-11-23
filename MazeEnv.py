import gymnasium
import matplotlib.pyplot as plt
from gym import spaces
from gymnasium import spaces
from matplotlib import colors
from pylab import *
from time import sleep
import pygame
import cv2
from random import randint

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

    def __init__(self, initial_state, final_state, obstacles, render):

        super(MazeEnv, self).__init__()
        self.action_log = []
        self.position_log = set(initial_state)

        # Definimos el espacio de acción
        #   0: UP
        #   1: DOWN
        #   2: RIGHT
        #   3: LEFT
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=80,
                                            shape=(4,), dtype=np.float64)

        self.initial_state = initial_state
        self.final_state = self.random_final_state(obstacles, initial_state)

        # Duración de la ducha en segundos
        self.max_steps = 64

        self.obstacles = obstacles

        self.renderize = render

        # Contador de tiempo
        self.current_step = 0

        self.truncated = False

        # Actual state (fila, columna)
        self.current_state = initial_state
        self.next_state = (0, 0)

        self.reward = 0

        if self.renderize:

            ## BOARD ##
            self.img =np.zeros((640, 640, 3), dtype=np.uint8)
            for x in range(0, 9):
                if x % 2 == 0: p = range(0, 8, 2)
                else: p = range(1, 9, 2)
                for y in p:
                    cv2.rectangle(self.img, (x*80, y*80), (x*80+80,y*80+80), (255,255,255), -1)       

            # OBSTACLES
            for obstacle in obstacles:
                cv2.rectangle(self.img, (obstacle[0]*80, obstacle[1]*80),
                            (obstacle[0]*80+80,obstacle[1]*80+80), (0,0,255), -1)
                
            # GOAL
            cv2.rectangle(self.img, (self.final_state[0]*80, self.final_state[1]*80),
                            (self.final_state[0]*80+80,self.final_state[1]*80+80), (0,255,255), -1)
        

    def random_final_state(self, obstacles, initial_state):

        final_state = (randint(0,7), randint(0,7))
        while final_state in obstacles or final_state == initial_state:
            final_state = (randint(0,7), randint(0,7))
        return final_state

    def is_posible(self):

        return self.next_state not in self.obstacles and -1 < self.next_state[0] < 8 and -1 < self.next_state[1] < 8

    def is_terminal(self):

        return self.final_state[0] == self.current_state[0] and self.final_state[1] == self.current_state[1]

    def step(self, action):

        if action == 0: self.next_state = (self.current_state[0] - 1, self.current_state[1])
        if action == 1: self.next_state = (self.current_state[0] + 1, self.current_state[1])
        if action == 2: self.next_state = (self.current_state[0], self.current_state[1] + 1)
        if action == 3: self.next_state = (self.current_state[0], self.current_state[1] - 1)

        # print(f'Distancia goal: {np.sum(np.abs(np.array(self.current_state) - np.array(self.final_state)))}')
        if self.is_posible():

            self.action_log.append(action)
            self.current_state = self.next_state
            self.current_step += 1

            self.reward = 15 - np.sum(np.abs(np.array(self.current_state) - np.array(self.final_state)))

            if self.current_state in self.position_log:
                self.reward -= 1


        else:
            #print('ILEGAL ACTION\t')
            self.reward = -100
            # self.truncated = True


        ########## Visualización ##########

        

        if self.is_terminal():

            board = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.putText(board, 'You Win!', (280, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            self.reward = 100
            sleep_time = 1
            # TODO return action log
        
        else: 

            board = self.img.copy()
            cv2.circle(board, (self.current_state[0]*80+40, self.current_state[1]*80+40), 30, (0,255,0), -1)
            sleep_time = 0.05

        if self.renderize:
            cv2.imshow('DUMB MAZE RUNNER', board)
            cv2.waitKey(1)
            sleep(sleep_time)

        ###################################   

        obs = np.array([self.current_state[0], self.current_state[1],
                        abs(self.current_state[0]-self.final_state[0]),
                        abs(self.current_state[1]-self.final_state[1])])
        
        return obs, self.reward, self.is_terminal(), self.truncated, {}

    def reset(self):

        self.current_step = 0
        self.current_state = (0, 0)
        self.reward = 0
        self.action_log = []
        self.final_state = self.random_final_state(self.obstacles, self.initial_state)

        self.img =np.zeros((640, 640, 3), dtype=np.uint8)

        if self.renderize:

            ## BOARD ##
            
            for x in range(0, 9):
                if x % 2 == 0: p = range(0, 8, 2)
                else: p = range(1, 9, 2)
                for y in p:
                    cv2.rectangle(self.img, (x*80, y*80), (x*80+80,y*80+80), (255,255,255), -1)       
            ###############

            for obstacle in self.obstacles:
                cv2.rectangle(self.img, (obstacle[0]*80, obstacle[1]*80),
                            (obstacle[0]*80+80,obstacle[1]*80+80), (0,0,255), -1)
            cv2.rectangle(self.img, (self.final_state[0]*80, self.final_state[1]*80),
                            (self.final_state[0]*80+80,self.final_state[1]*80+80), (0,255,255), -1)

        info = {}

        obs = np.array([self.current_state[0], self.current_state[1],
                        abs(self.current_state[0]-self.final_state[0]),
                        abs(self.current_state[1]-self.final_state[1])])

        return obs, info


    def close(self):
        pass
