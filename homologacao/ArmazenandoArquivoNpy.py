import cv2
import mediapipe as mp
import numpy as np

# Inicializar o MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Configurar os vídeos de referência e comparação
# video_ref_path = 'videos/Polichinelo.mp4'
# video_comp_path = 'videos/PolichineloMato.mp4'

video_ref_path = 'videos/DesenvolvimentoHomem.mp4'
video_comp_path = 'videos/DesenvolvimentoMulher.mp4'

# Caminho do arquivo a ser lido ou gerado
# landmarks_file = 'referencias/Polichinelo.npy'

landmarks_file = 'referencias/Desenvolvimento.npy'

# Inicializar os modelos de detecção de pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Verificar se os dados de landmarks já foram salvos em um arquivo
    try:
        ref_landmarks = np.load(landmarks_file)
    except FileNotFoundError:
        ref_landmarks = None

    # Abrir o vídeo de referência
    cap_ref = cv2.VideoCapture(video_ref_path)

    # Armazenar os dados do vídeo de referência se ainda não estiverem presentes
    if ref_landmarks is None:
        ref_landmarks = []

        # Ler os quadros do vídeo de referência e armazenar os dados
        while cap_ref.isOpened():
            ret_ref, frame_ref = cap_ref.read()

            if not ret_ref:
                break

            frame_ref_rgb = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)
            results_ref = pose.process(frame_ref_rgb)

            if results_ref.pose_landmarks is None:
                continue

            landmarks_ref = np.array([[lm.x, lm.y, lm.z] for lm in results_ref.pose_landmarks.landmark])
            ref_landmarks.append(landmarks_ref)

            # Mostrar o resultado com os pontos corporais desenhados para o vídeo de referência
            annotated_image_ref = frame_ref.copy()
            mp_drawing.draw_landmarks(
                annotated_image_ref, results_ref.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Video de Referência', annotated_image_ref)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Converter os dados de referência para um array numpy
        ref_landmarks = np.array(ref_landmarks)

        # Salvar os dados em um arquivo
        np.save(landmarks_file, ref_landmarks)

    # Liberar os recursos do vídeo de referência
    cap_ref.release()
    cv2.destroyAllWindows()

    # Abrir o vídeo de comparação
    cap_comp = cv2.VideoCapture(video_comp_path)

    # Loop principal para processar os quadros do vídeo de comparação
    while cap_comp.isOpened():
        ret_comp, frame_comp = cap_comp.read()

        if not ret_comp:
            break

        frame_comp_rgb = cv2.cvtColor(frame_comp, cv2.COLOR_BGR2RGB)
        results_comp = pose.process(frame_comp_rgb)

        if results_comp.pose_landmarks is None:
            continue

        landmarks_comp = np.array([[lm.x, lm.y, lm.z] for lm in results_comp.pose_landmarks.landmark])

        # Comparar os movimentos do vídeo de comparação com os do vídeo de referência
        similarity_scores = np.mean(np.abs(landmarks_comp - ref_landmarks), axis=(1, 2))
        similarity_score = np.min(similarity_scores)

        # Mostrar o resultado com os pontos corporais desenhados para o vídeo de comparação
        annotated_image_comp = frame_comp.copy()
        mp_drawing.draw_landmarks(
            annotated_image_comp, results_comp.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Exibir o texto indicando a similaridade dos movimentos
        similarity_text = f"Similaridade: {similarity_score:.2f}"
        cv2.putText(annotated_image_comp, similarity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Video de Comparação', annotated_image_comp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos do vídeo de comparação
    cap_comp.release()
    cv2.destroyAllWindows()
