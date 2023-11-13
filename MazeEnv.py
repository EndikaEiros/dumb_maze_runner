import numpy as np
from gym import Env
from gym import spaces
from gym.spaces import Box, Discrete
import random
import pygame
import gymnasium
from gymnasium import spaces

class MazeEnv(gymnasium.Env):

    def __init__(self, initial_state, final_state, obstacles):

        super(MazeEnv, self).__init__()
        self.action_log=[]

        # Definimos el espacio de acción
        #   0: UP
        #   1: DOWN
        #   2: RIGHT
        #   3: LEFT
        self.action_space = spaces.Discrete(4)

        self.initial_state = initial_state
        self.final_state = final_state

        self.observation_space = [
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0]
        ]

        # Duración de la ducha en segundos
        self.max_steps = 64

        self.obstacles = obstacles

        # Contador de tiempo
        self.current_step = 0

        # Actual state (fila, columna)
        self.current_state = initial_state
        self.next_state = (0,0)

        self.reward = 0

    def is_posible(self, action):

        posible = True
        if action == 0 and self.current_state[0] == 0: posible = False
        if action == 1 and self.current_state[0] == 7: posible = False
        if action == 2 and self.current_state[1] == 7: posible = False
        if action == 3 and self.current_state[1] == 0: posible = False

        return posible
    
    def is_terminal(self):
        
        return self.final_state == self.current_state


    def step(self, action):

        if self.is_posible(action=action):

            self.action_log.append(action)
            
            if action == 0: self.next_state = (self.current_state[0] - 1, self.current_state[1])
            if action == 1: self.next_state = (self.current_state[0] + 1, self.current_state[1])
            if action == 2: self.next_state = (self.current_state[0], self.current_state[1] + 1)
            if action == 3: self.next_state = (self.current_state[0], self.current_state[1] - 1)

            if self.next_state not in self.obstacles:
                self.current_state = self.next_state
            else:
                print('ILEGAL ACTION\t')
                
            self.current_step += 1

        else: print('ILEGAL ACTION\t')
        
        if self.is_terminal():
            self.reward = 100


        return self.current_state, self.reward, self.is_terminal()

            

    def reset(self):
        # Contador de tiempo
        self.current_step = 0

        # Actual state (fila, columna)
        self.current_state = (0, 0)

        self.reward = 0

        self.action_log = []

        return self.current_state, self.reward, self.is_terminal()

    def render(self, mode='human'):
        tablero = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
        tablero[self.current_state[0]][self.current_state[1]] = 3
        tablero[self.final_state[0]][self.final_state[1]] = 0
        for obstacle in self.obstacles:
            tablero[obstacle[0]][obstacle[1]] = 8


        for line in tablero:
            print(f"{line}\n")
        print('\n')

    def close(self):
        pass
