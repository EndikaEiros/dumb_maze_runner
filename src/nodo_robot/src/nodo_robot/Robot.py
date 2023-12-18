from moveit_commander import MoveGroupCommander
from moveit_commander import PlanningSceneInterface
from moveit_commander import RobotCommander
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from control_msgs.msg import GripperCommandActionGoal
import rospy
import keyboard
import copy
import yaml
import numpy as np
from std_msgs.msg import String
from nodo_rl.msg import Action

class ControlRobot:
    def __init__(self) -> None:
        rospy.init_node("nodo_robot",anonymous=True)
        self.move_group = MoveGroupCommander("robot")
        self.planning_scene = PlanningSceneInterface()
        self.robot_commander = RobotCommander()
        self.publicador = rospy.Publisher("topic3",String, queue_size=5)
        self.dato_recibido = None
        self.lista_mov = {}

        pose_suelo = PoseStamped()
        pose_suelo.header.frame_id = self.robot_commander.get_planning_frame()
        pose_suelo.pose.position.z = -0.011
        self.planning_scene.add_box("suelo",pose_suelo,(3,3,0.02))

        self.move_group.set_planning_time(10)
        self.move_group.set_num_planning_attempts(5)

        self.publicador_pinza = rospy.Publisher("/rg2_action_server/goal",
                                                 GripperCommandActionGoal,
                                                 queue_size=10)
        rospy.Subscriber("topic2", Action, self.recibir_mensajes)
        rospy.sleep(2)
        
    def recibir_mensajes(self, data: Action) -> None:
        lista_acciones = list()
        for n in data.acciones:
            lista_acciones.append(int(np.uint8(n)))
        self.dato_recibido = lista_acciones
        print(self.dato_recibido)
    
    def mover_articulaciones(self, valores_articulaciones: list) -> bool:
        return self.move_group.go(valores_articulaciones)

    def manejar_pinza(self, anchura: float, fuerza: float) -> None:
        #Publicar mensaje al topic
        msg_pinza = GripperCommandActionGoal() #msg = mensaje = message 
        msg_pinza.goal.command.position = anchura
        msg_pinza.goal.command.max_effort = fuerza
        self.publicador_pinza.publish(msg_pinza) #Aqui hay que decirle la anchura y fuerza de esta manera
        
    def enviar(self):
        self.publicador.publish('Terminado')
        rospy.Rate(10)

    def bajar_z(self) -> bool:
        waypoints = []
        #Se inicializa con la posicion inicial
        wpose = self.move_group.get_current_pose().pose
        wpose.position.z = wpose.position.z - 0.015
        waypoints.append(copy.deepcopy(wpose))
        
        (plan3, fraction) = self.move_group.compute_cartesian_path(
                             waypoints,   # waypoints to follow
                             0.01,        # eef_step
                             0.0)         # jump_threshold
        # rospy.sleep(5)

        self.move_group.set_pose_target(wpose)
        print(fraction)
        success, trajectory, _, _ =self.move_group.plan()
        
        if not success:
            return False
        
        if fraction < 1:
            print(fraction)
            return False
        else:
            return self.move_group.execute(plan3)
        
    def subir_z(self) -> bool:
        waypoints = []
        #Se inicializa con la posicion inicial
        wpose = self.move_group.get_current_pose().pose
        wpose.position.z = wpose.position.z + 0.015
        waypoints.append(copy.deepcopy(wpose))
        
        (plan3, fraction) = self.move_group.compute_cartesian_path(
                             waypoints,   # waypoints to follow
                             0.01,        # eef_step
                             0.0)         # jump_threshold
        # rospy.sleep(5)

        self.move_group.set_pose_target(wpose)
        print(fraction)
        success, trajectory, _, _ =self.move_group.plan()
        
        if not success:
            return False
        
        if fraction < 1:
            print(fraction)
            return False
        else:
            return self.move_group.execute(plan3)
       
    
    def mover_a_pose(self, lista_mov: list) -> bool:

        waypoints = []
        wpose = self.move_group.get_current_pose().pose
        
        #MOVER CUADRADO: 0.025 = 2.5cm
        for i in lista_mov:
            if i == 0:
                wpose.position.x = wpose.position.x + 0.025
                waypoints.append(copy.deepcopy(wpose))
            elif i == 1:
                wpose.position.x = wpose.position.x - 0.025
                waypoints.append(copy.deepcopy(wpose))
            elif i == 2:
                wpose.position.y = wpose.position.y - 0.025
                waypoints.append(copy.deepcopy(wpose))
            elif i == 3:
                wpose.position.y = wpose.position.y + 0.025
                waypoints.append(copy.deepcopy(wpose))
        
        (plan3, fraction) = self.move_group.compute_cartesian_path(
                             waypoints,   # waypoints to follow
                             0.01,        # eef_step
                             0.0)         # jump_threshold
        # rospy.sleep(5)

        self.move_group.set_pose_target(wpose)
        print(fraction)
        success, trajectory, _, _ =self.move_group.plan()
        
        if not success:
            return False

        if fraction < 1:
            print(fraction)
            return False
        else:
            return self.move_group.execute(plan3)

    def manejar_pinza(self, anchura: float, fuerza: float) -> None:
        msg_pinza = GripperCommandActionGoal()
        msg_pinza.goal.command.position = anchura
        msg_pinza.goal.command.max_effort = fuerza

        self.publicador_pinza.publish(msg_pinza)

    #Guardar los valores de las articulaciones actuales, se pasa el nombre con el que se guardará en el registro
    def registrar_joints_actuales(self, nombre: str) -> None:
        try:
            with open("registro_joints.yaml","+r") as f:
                current_data = yaml.load(f, yaml.Loader)
        except:
            with open("registro_joints.yaml","+w") as f:
                pass
            current_data = None
        if current_data is None:
            current_data = {}
        current_data.update({nombre:self.move_group.get_current_joint_values()})
        with open("registro_joints.yaml","+w") as f:
            yaml.dump(current_data, f)
            
    def cargar_joints_registradas(self) -> dict:
        try:
            with open("registro_joints.yaml","+r") as f:
                current_data = yaml.load(f, yaml.Loader)
        except:
            current_data = {}
            
        return current_data

if __name__ == '__main__':
    
    while True:
        control_robot = ControlRobot()
        copy_mov_list = None
        
        while copy_mov_list is None:
            copy_mov_list = copy.deepcopy(control_robot.dato_recibido)
        print(copy_mov_list)
        
        control_robot.manejar_pinza(50.0, 25.0) 
        current_data = control_robot.cargar_joints_registradas()
        control_robot.mover_articulaciones(current_data['esquina'])
        control_robot.bajar_z()
        control_robot.manejar_pinza(24.5, 25.0)
        rospy.sleep(1)
        control_robot.subir_z()
                
        control_robot.mover_a_pose(lista_mov=copy_mov_list)
        control_robot.bajar_z()
        control_robot.manejar_pinza(50.0, 25.0)
        control_robot.subir_z()
        
        control_robot.mover_articulaciones(current_data['inicio'])
        
        control_robot.dato_recibido = None
        
        control_robot.enviar()
        
        pass
    