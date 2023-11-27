import rospy
import numpy as np
from nodo_vision.msg import Coord
from nodo_RL.msg import Action

class Nodo_RL:

    def __init__(self) -> None:

        rospy.init_node("nodo_RL",anonymous=True)
        self.publisher = rospy.Publisher("topic1", Action ,queue_size=5)
        self.tiempo_ciclo = rospy.Rate(10)

    def rutina(self) -> None:
        # TODO insert action log
        self.mi_primer_publicador.publish(Action())
        self.tiempo_ciclo.sleep()
    
    def start(self) -> None:
        while not rospy.is_shutdown():
            self.rutina() 

    def recibir_mensajes(data: Coord) -> None:
        for numero in data.coordenadas:
            print(np.uint8(numero))

    rospy.Subscriber("topic1", Coord, recibir_mensajes)

    rospy.spin()

if __name__ == '__main__':
    nodo = Nodo_RL()
    nodo.start()
