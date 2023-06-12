import cv2
import mediapipe as mp
import numpy as np
from yolo import YoloDetector 

# Inicializar o MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Configurar os vídeos de referência e comparação
video_ref_path = 'videos/Polichinelo.mp4'
video_comp_path = 'videos/PolichineloMato.mp4'

# Definir o limite de similaridade
threshold = 0.15

# Inicializar os modelos de detecção de pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Inicializar o detector YOLO
    yolo_detector = YoloDetector()  # Substitua por inicialização do YOLO conforme sua implementação

    # Abrir os vídeos de referência e comparação
    cap_ref = cv2.VideoCapture(video_ref_path)
    cap_comp = cv2.VideoCapture(video_comp_path)

    # Variáveis para armazenar os pontos corporais dos vídeos de referência e comparação
    landmarks_ref = []
    landmarks_comp = []

    # Loop para capturar os pontos corporais dos vídeos de referência e comparação
    while cap_ref.isOpened() and cap_comp.isOpened():
        # Ler os próximos quadros dos vídeos de referência e comparação
        ret_ref, frame_ref = cap_ref.read()
        ret_comp, frame_comp = cap_comp.read()

        # Parar o loop se algum dos vídeos chegou ao fim
        if not ret_ref or not ret_comp:
            break

        # Verificar se há uma pessoa reconhecível nos quadros usando o detector YOLO
        if yolo_detector.detect_person(frame_ref) and yolo_detector.detect_person(frame_comp):
            # Converter os quadros para RGB
            frame_ref_rgb = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)
            frame_comp_rgb = cv2.cvtColor(frame_comp, cv2.COLOR_BGR2RGB)

            # Detectar poses nos quadros
            results_ref = pose.process(frame_ref_rgb)
            results_comp = pose.process(frame_comp_rgb)

            # Verificar se os dois quadros possuem pontos corporais
            if results_ref.pose_landmarks is not None and results_comp.pose_landmarks is not None:
                # Adicionar os pontos corporais dos quadros aos respectivos arrays
                landmarks_ref.append(results_ref.pose_landmarks.landmark)
                landmarks_comp.append(results_comp.pose_landmarks.landmark)

            # Desenhar os pontos corporais nos quadros
            mp_drawing.draw_landmarks(
                frame_ref, results_ref.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                frame_comp, results_comp.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Exibir os quadros com os pontos corporais
        cv2.imshow('Video de Referência', frame_ref)
        cv2.imshow('Video de Comparação', frame_comp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Converter os arrays de pontos corporais em matrizes numpy
    landmarks_ref = np.array([[lm.x, lm.y, lm.z] for frame_landmarks in landmarks_ref for lm in frame_landmarks], dtype=np.float32)
    landmarks_comp = np.array([[lm.x, lm.y, lm.z] for frame_landmarks in landmarks_comp for lm in frame_landmarks], dtype=np.float32)

    # Calcular a similaridade usando a distância Euclidiana entre os pontos corporais
    similarity_scores = np.mean(np.linalg.norm(landmarks_comp - landmarks_ref, axis=1))

    # Exibir o resultado da comparação de movimentos
    if similarity_scores < threshold:
        print("O movimento está sendo executado corretamente.")
    else:
        print("O movimento não está sendo executado corretamente.")

    # Liberar os recursos
    cap_ref.release()
    cap_comp.release()
    cv2.destroyAllWindows()