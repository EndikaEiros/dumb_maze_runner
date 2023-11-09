from moveit_commander import MoveGroupCommander
from moveit_commander import PlanningSceneInterface
from geometry_msgs.msg import PoseStamped
from control_msgs.msg import GripperCommandActionGoal
from moveit_commander import RobotCommander
from tf.transformations import quaternion_from_euler
from time import sleep
import rospy


class PruebasRobot:

    def __init__(self) -> None:

        ## Inicializaciones ##
        rospy.init_node("paquete_prueba",anonymous=True)
        rospy.sleep(2)
        self.move_group = MoveGroupCommander("robot")
        planning_scene = PlanningSceneInterface()
        self.robot_commander = RobotCommander()       

        ## Creación de objetos ##

        # Borrar objetos anteriores
        planning_scene.remove_world_object()

        # Objeto 1: suelo
        pose_suelo = PoseStamped()
        pose_suelo.header.frame_id = self.robot_commander.get_planning_frame()
        pose_suelo.pose.position.x = 0.0
        pose_suelo.pose.position.y = 0.0
        pose_suelo.pose.position.z = -0.011
        planning_scene.add_box("suelo",pose_suelo,(10, 10, 0.02))

        # Objeto 2: tablero TODO
        pose_tablero = PoseStamped()
        pose_tablero.header.frame_id = self.robot_commander.get_planning_frame()
        pose_tablero.pose.position.x = 0.4
        pose_tablero.pose.position.y = -0.175
        pose_tablero.pose.position.z = 0.09
        planning_scene.add_box("tablero",pose_tablero,(0.3, 0.3, 0.01))

        self.move_group.set_planning_time(10)
        self.move_group.set_num_planning_attempts(5)        

        self.publicador_pinza = rospy.Publisher("/rg2_action_server/goal",
                                                GripperCommandActionGoal,
                                                queue_size=10)
        
    # Método para mover las 6 articulaciones del robot
    def mover_articulaciones(self, valores_articulaciones: list) -> bool:
        return self.move_group.go(valores_articulaciones)

    # Mover el robor a una pose específica 
    def mover_a_pose(self, lista_pose: list) -> bool:
        orientacion_quaternion = quaternion_from_euler(lista_pose[3],
                                                       lista_pose[4],
                                                       lista_pose[5])        
        
        pose_meta = PoseStamped()
        pose_meta.header.frame_id = self.robot_commander.get_planning_frame()
        pose_meta.pose.position.x = lista_pose[0]
        pose_meta.pose.position.y = lista_pose[1]
        pose_meta.pose.position.z = lista_pose[2]
        pose_meta.pose.orientation.w = orientacion_quaternion[3]
        pose_meta.pose.orientation.x = orientacion_quaternion[0]
        pose_meta.pose.orientation.y = orientacion_quaternion[1]
        pose_meta.pose.orientation.z = orientacion_quaternion[2]

        self.move_group.set_pose_target(pose_meta)
        success, trajectory, _, _ =self.move_group.plan()
        
        if not success:
            return False

        return self.move_group.execute(trajectory)
    
    # Método para controlar el estado de la pinza
    def manejar_pinza(self, anchura: float, fuerza: float) -> None:
        msg_pinza = GripperCommandActionGoal()
        msg_pinza.goal.command.position = anchura
        msg_pinza.goal.command.max_effort = fuerza

        self.publicador_pinza.publish(msg_pinza)



if __name__ == '__main__':
    
    test = PruebasRobot()

    # Mover a la pose 'pose'
    # Cuidado con este método ya que utiliza un algoritmo interno para decidir el trazado de la ruta = puede dar errores
    pose = [0.2,0.2,0.2,0,0,0]
    # test.mover_a_pose(pose)

    # Mover pinza (0-100:Cerrar-Abrir, 0-40:Newtons)
    #test.manejar_pinza(0.0, 40.0)

    # Obtener la posición de cada articulación en radianes (array de 6 floats)
    joints = test.move_group.get_current_joint_values()

    # Obtener el nombre de las 6 articulaciones (array de 6 strings)
    joints = test.move_group.get_joints()

    # Mover las articulaciones del robot a la posición 'joints'
    #test.mover_articulaciones(joints)


    # rospy.spin()
    pass
