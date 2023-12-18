import rospy
import numpy as np
import time
from MazeEnv import MazeEnv
from stable_baselines3 import PPO
from nodo_vision.msg import Coord
from nodo_rl.msg import Action


def optimizar_ruta(lista):

    """
        Dada una lista de estados elimina los estados intermedios innecesarios
        Ejemplo: [(0, 7), (1, 7), (2, 7), (1, 7), (1, 6)] --> [(0, 7), (1, 7), (1, 6)]
    """
    lista_simplificada = lista

    for idx, element in enumerate(lista):

        if lista.count(element) > 1:
            pos = lista[idx + 1:].index(element) + len(lista[:idx + 1])
            lista_simplificada = lista[:idx] + lista[pos:]
            break

    return lista_simplificada


def obtener_acciones(lista):

    """ Dada la lista de estados por los que ha pasado devuelve la lista de acciones realizadas """

    acciones = list()
    prev_state = (-10, -10)

    for state in lista:

        if state[0] - prev_state[0] == 1:
            acciones.append(2)
        elif state[0] - prev_state[0] == -1:
            acciones.append(3)
        elif state[1] - prev_state[1] == 1:
            acciones.append(1)
        elif state[1] - prev_state[1] == -1:
            acciones.append(0)
        prev_state = state

    return acciones


class Nodo_RL:

    def __init__(self) -> None:

        """ Inicialización del nodo de RL """

        rospy.init_node("nodo_publisher", anonymous=True)
        self.mi_primer_publicador = rospy.Publisher("topic2", Action, queue_size=5)
        self.tiempo_ciclo = rospy.Rate(10)

    def enviar(self, acciones) -> None:

        """ Publisher: Envia la lista de acciones """

        self.mi_primer_publicador.publish(acciones)
        self.tiempo_ciclo.sleep()

    def start(self) -> None:

        """ Poner en marcha el nodo de RL """

        while not rospy.is_shutdown():
            self.rutina()

    def recibir_mensajes(self, data: Coord) -> None:

        """ Subscriber: Recibe la lista de obstáculos """

        ########## Get position of obstacles ###########

        obstacles = set()
        for numero in data.coordenadas:
            coods = [int(n) - 1 for n in str(np.uint8(numero))]
            obstacles.add((coods[0], coods[1]))

        print(f"\n Obstáculos recibidos: {obstacles}")
        self.calculate_actions(obstacles)

    def calculate_actions(self, obstacles: set()):

        """ Genera el entorno, carga al agente entrenado y obtiene la solución """

        maze_env = MazeEnv(render=True, obstacles=obstacles)
        model = PPO.load('src/nodo_rl/src/ppo_maze_prueba7.model')

        done = False
        actions = []
        obs, info = maze_env.reset()
        states = [tuple(info["current_state"])]
        attempts = 100

        ############## Calculate actions ###############

        while not done and attempts > 0:

            action, _ = model.predict(obs)
            actions.append(int(action))
            obs, reward, done, truncated, info = maze_env.step(action)
            states.append(tuple(info["current_state"]))

            if truncated:
                attempts -= 1
                done = False
                actions = []
                obs, info = maze_env.reset()
                states = [tuple(info["current_state"])]

        if attempts == 0:
            actions = [1]

        print(f"\n Solución obtenida: {actions}")

        # Visualizar solución del agente
        maze_env = MazeEnv(render=True, obstacles=obstacles)
        for action in actions:
            maze_env.step(action)

        ############## Optimize solution ###############

        # Si vuelve al estado inicial eliminar pasos intermedios
        for i, elem in enumerate(reversed(states[1:])):
            if elem == states[0]:
                states = states[len(states) - i - 1:]

        # Eliminar estados innecesarios mientras haya estados repetidos
        while len(states) != len(set(states)):
            states = optimizar_ruta(states)

        # Obtener lista de acciones optimizada
        actions = obtener_acciones(states)
        print(f"\n Solución optimizada: {actions}")

        # Visualizar solución optimizada
        maze_env = MazeEnv(render=True, obstacles=obstacles)
        for action in actions:
            maze_env.step(action)

        ################ Send solution #################

        solution = Action()
        solution.acciones = actions
        self.enviar(solution)

        print('\n Solución enviada\n')


if __name__ == '__main__':
    nodo = Nodo_RL()

    rospy.Subscriber("topic1", Coord, nodo.recibir_mensajes)
    rospy.spin()

    nodo.start()
