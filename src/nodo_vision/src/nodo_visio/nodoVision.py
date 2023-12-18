import rospy
from std_msgs.msg import String
import mediapipe as mp
from joblib import load
import numpy as np
import cv2
import time
from nodo_vision.msg import Coord

class NodoVision:
    def __init__(self) -> None:
        rospy.init_node("nodo_publisher",anonymous=True)
        self.mi_primer_publicador = rospy.Publisher("topic1", Coord,queue_size=5)
        self.tiempo_ciclo = rospy.Rate(10)
        self.bool_enviar = True

    #def rutina(self) -> None:
    #    self.mi_primer_publicador.publish(String(""))
    #    self.tiempo_ciclo.sleep()

    def enviar(self, coordenadas) -> None:
        if self.bool_enviar:
            self.mi_primer_publicador.publish(coordenadas)
        self.bool_enviar = False
        self.tiempo_ciclo.sleep()
    
    def recibir_mensaje(self, data: String) -> None:
        print(data)
        self.bool_enviar = True
    
    #def start(self) -> None:
    #    while not rospy.is_shutdown():
    #        self.rutina()
def enviar_coordenadas(nodo, coordenadas):
    global tiempo_inicial
    global buffer
    global array_coord
    global mensaje
    buffer.append(coordenadas)
    if time.time() - 2 > tiempo_inicial:
        if buffer.count(coordenadas) == len(buffer) and len(buffer) > 10:
            if coordenadas == 0 and len(array_coord) > 0:
                array_coord.pop()
            elif coordenadas == 1:
                #resultado_coord = " ".join(array_coord)
                mensaje.coordenadas = array_coord
                print("enviado:", mensaje)
                nodo.enviar(mensaje)
                array_coord = []
            #elif '0' not in str(coordenadas) and '9' not in str(coordenadas):
            elif coordenadas > 10 and '9' not in str(coordenadas) and '0' not in str(coordenadas):
                if len(array_coord) < 49 and coordenadas not in array_coord:
                    array_coord.append(coordenadas)
        buffer = []
        tiempo_inicial = time.time()

if __name__ == '__main__':
    nodo = NodoVision()
    rospy.Subscriber("topic3", String, nodo.recibir_mensaje)
    model_right = load('modelos/random-forest-classifier-right.joblib')
    #model_right = load('/home/ubuntu/workspaces/catkin_ws/src/nodo_vision/src/nodo_vision/modelos/random-forest-classifier-right.joblib')
    #model_left = load('/home/ubuntu/workspaces/catkin_ws/src/nodo_vision/src/nodo_vision/modelos/random-forest-classifier-left.joblib')
    model_left = load('modelos/random-forest-classifier-left.joblib')
    array_coord = []
    buffer = []
    mensaje = Coord()
    # Inicialización de la cámara (cambia el número de cámara según sea necesario)
    cap = cv2.VideoCapture(0)

    # Inicialización de detección de manos con MediaPipe
    clase_manos = mp.solutions.hands
    manos = clase_manos.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    dibujo = mp.solutions.drawing_utils
    estilos = mp.solutions.drawing_styles
    manos1 = clase_manos.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    offset = 30
    cropSize = 300
    tiempo_inicial = -1
    while True:
        success, img = cap.read()
        if not success:
            break
        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultados = manos.process(img_rgb)
        copia = img.copy()
        predictions = {}
        coordenadas = ""
        if resultados.multi_hand_landmarks:
            for j, landmarks_manos in enumerate(resultados.multi_hand_landmarks):
                X = []
                Y = []
                img_white = np.ones((cropSize, cropSize, 3), np.uint8)*255
                dibujo.draw_landmarks(copia, landmarks_manos, clase_manos.HAND_CONNECTIONS, estilos.get_default_hand_landmarks_style(),estilos.get_default_hand_connections_style())
                tipo_mano = resultados.multi_handedness[j].classification[0].label
                if tipo_mano == 'Right':
                    tipo_mano = 'Left'
                elif tipo_mano == 'Left':
                    tipo_mano = 'Right'
                for i in range(len(landmarks_manos.landmark)):
                    X.append(landmarks_manos.landmark[i].x)
                    Y.append(landmarks_manos.landmark[i].y)
                x_min = max(0, int(min(X)*W) - offset)
                x_max = min(W-1, int(max(X)*W) + offset)
                y_min = max(0, int(min(Y)*H) - offset)
                y_max = min(H-1, int(max(Y)*H) + offset)
                img_rgb_crop = img_rgb[y_min:y_max, x_min:x_max]
                cv2.rectangle(copia, (x_min, y_min), (x_max, y_max), (0,0,255), 4)

                w_hand = x_max - x_min
                h_hand = y_max - y_min
                aspectRatio = h_hand/w_hand

                if aspectRatio > 1:
                    w_crop = int(round(cropSize*(w_hand/h_hand)))
                    cropResize = cv2.resize(img_rgb_crop, (w_crop, cropSize))
                    center_gap = int(round((cropSize-w_crop)/2))
                    img_white[:, center_gap:w_crop+center_gap] = cropResize
                else:
                    h_crop = int(round(cropSize*aspectRatio))
                    cropResize = cv2.resize(img_rgb_crop, (cropSize, h_crop))
                    center_gap = int(round((cropSize-h_crop)/2))
                    img_white[center_gap:h_crop+center_gap,:] = cropResize

                #cv2.imshow("Recorte", img_white)

                resultado_recortado = manos1.process(img_white)

                if resultado_recortado.multi_hand_landmarks:
                    for landmark_mano_recortada in resultado_recortado.multi_hand_landmarks:
                        data_aux = []
                        for i in range(len(landmark_mano_recortada.landmark)):
                            data_aux.append(landmark_mano_recortada.landmark[i].x)
                            data_aux.append(landmark_mano_recortada.landmark[i].y)
                        if tipo_mano == 'Right':
                            prediction = model_right.predict([np.asarray(data_aux)])
                            predictions["mano_derecha"] = prediction
                        elif tipo_mano == 'Left':
                            prediction = model_left.predict([np.asarray(data_aux)])
                            predictions["mano_izquierda"] = prediction
                        cv2.putText(copia, str(prediction[0]) + str(tipo_mano), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
                    if len(predictions.values())==2:
                        coordenadas = str(predictions["mano_derecha"][0]) + str(predictions["mano_izquierda"][0])
                        coordenadas = np.uint8(coordenadas)
                        if tiempo_inicial == -1:
                            tiempo_inicial = time.time()  
                        enviar_coordenadas(nodo, coordenadas)
        
        for i in [10, 20, 30, 40, 50]:
            cv2.putText(copia, ' '.join([str(v) for v in array_coord[(i-10):i]]), (1, 30+(i-10)*5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow("Manos", copia)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    
    cap.release()