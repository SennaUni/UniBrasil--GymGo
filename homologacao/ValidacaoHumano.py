import cv2
import mediapipe as mp
import numpy as np

# Inicializar o MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Configurar os vídeos de referência e comparação
video_ref_path = 'videos/Polichinelo.mp4'
video_comp_path = 'videos/PolichineloMato.mp4'

# Inicializar os modelos de detecção de pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Abrir os vídeos de referência e comparação
    cap_ref = cv2.VideoCapture(video_ref_path)
    cap_comp = cv2.VideoCapture(video_comp_path)

    # Criar as janelas para exibir os vídeos
    cv2.namedWindow('Video de Referência', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Video de Comparação', cv2.WINDOW_NORMAL)

    # Variável para armazenar a similaridade dos movimentos
    similarity_score = 0.0

    # Loop principal para processar os quadros dos dois vídeos
    while cap_ref.isOpened() and cap_comp.isOpened():
        # Ler os próximos quadros dos vídeos de referência e comparação
        ret_ref, frame_ref = cap_ref.read()
        ret_comp, frame_comp = cap_comp.read()

        # Parar o loop se algum dos vídeos chegou ao fim
        if not ret_ref or not ret_comp:
            break

        # Converter os quadros para RGB
        frame_ref_rgb = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)
        frame_comp_rgb = cv2.cvtColor(frame_comp, cv2.COLOR_BGR2RGB)

        # Detectar poses nos quadros
        results_ref = pose.process(frame_ref_rgb)
        results_comp = pose.process(frame_comp_rgb)

        # Verificar se uma pessoa foi detectada nos quadros de referência e comparação
        if results_ref.pose_landmarks is None or results_comp.pose_landmarks is None:
            # Pular para o próximo quadro se uma pessoa não for detectada em um dos vídeos
            continue

        # Mostrar o resultado com os pontos corporais desenhados para o vídeo de referência
        annotated_image_ref = frame_ref.copy()
        mp_drawing.draw_landmarks(
            annotated_image_ref, results_ref.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Video de Referência', annotated_image_ref)

        # Mostrar o resultado com os pontos corporais desenhados para o vídeo de comparação
        annotated_image_comp = frame_comp.copy()
        mp_drawing.draw_landmarks(
            annotated_image_comp, results_comp.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Comparar os movimentos dos vídeos de referência e comparação
        landmarks_ref = np.array([[lm.x, lm.y, lm.z] for lm in results_ref.pose_landmarks.landmark])
        landmarks_comp = np.array([[lm.x, lm.y, lm.z] for lm in results_comp.pose_landmarks.landmark])
        similarity_score = np.mean(np.abs(landmarks_comp - landmarks_ref))

        # Exibir o texto indicando a similaridade dos movimentos
        similarity_text = f"Similaridade: {similarity_score:.2f}"
        cv2.putText(annotated_image_comp, similarity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Video de Comparação', annotated_image_comp)

        # Parar o loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos
    cap_ref.release()
    cap_comp.release()
    cv2.destroyAllWindows()
