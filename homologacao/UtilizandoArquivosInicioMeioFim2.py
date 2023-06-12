import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Configurar os vídeos de referência e comparação
video_ref_path = 'videos/Polichinelo.mp4'
video_comp_path = 'videos/PolichineloMato.mp4'

# Caminho do arquivo a ser lido ou gerado
landmarks_file = 'referencias/Polichinelo.npy'
landmarks_file_point = 'referencias/PolichineloPoint.npy'

# Definir os segundos exatos para capturar os pontos corporais
segundo_initial = 4.2
segundo_middle = 5.6
segundo_final = 6.1

# Converter os segundos para milissegundos
milissegundos_initial = int(segundo_initial * 1000)
milissegundos_middle = int(segundo_middle * 1000)
milissegundos_final = int(segundo_final * 1000)

# Initialize pose detection models
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    try:
        ref_landmarks = np.load(landmarks_file)
        ref_landmarks_points = np.load(landmarks_file_point)
    except FileNotFoundError:
        ref_landmarks = None
        ref_landmarks_points = None

     # Armazenar os dados do vídeo de referência se ainda não estiverem presentes
    if ref_landmarks is None:
        # Open the reference video
        cap_ref = cv2.VideoCapture(video_ref_path)

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

        # Close the reference video
        cap_ref.release()
        cv2.destroyAllWindows()

    if  ref_landmarks_points is None:
        # Open the reference video
        cap_ref = cv2.VideoCapture(video_ref_path)

        ref_landmarks_points = []

        cap_ref.set(cv2.CAP_PROP_POS_MSEC, milissegundos_initial)
        ret_ref, frame_ref = cap_ref.read()
        results_ref = pose.process(cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB))

        if results_ref.pose_landmarks is not None:
            landmarks_initial = np.array([[lm.x, lm.y, lm.z] for lm in results_ref.pose_landmarks.landmark])
            ref_landmarks_points.append(landmarks_initial)

        cap_ref.set(cv2.CAP_PROP_POS_MSEC, milissegundos_middle)
        ret_ref, frame_ref = cap_ref.read()
        results_ref = pose.process(cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB))

        if results_ref.pose_landmarks is not None:
            landmarks_middle = np.array([[lm.x, lm.y, lm.z] for lm in results_ref.pose_landmarks.landmark])
            ref_landmarks_points.append(landmarks_middle)

        cap_ref.set(cv2.CAP_PROP_POS_MSEC, milissegundos_final)
        ret_ref, frame_ref = cap_ref.read()
        results_ref = pose.process(cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB))

        if results_ref.pose_landmarks is not None:
            landmarks_final = np.array([[lm.x, lm.y, lm.z] for lm in results_ref.pose_landmarks.landmark])
            ref_landmarks_points.append(landmarks_final)

        # Converter os dados de referência para um array numpy
        ref_landmarks_points = np.array(ref_landmarks_points)

        # Salvar os dados em um arquivo
        np.save(landmarks_file_point, ref_landmarks_points)

        # Close the reference video
        cap_ref.release()
        cv2.destroyAllWindows()

    cap_comp = cv2.VideoCapture(video_comp_path)

    exercises_count = 0
    exercise_started = True  # Set to True initially
    current_index = 0

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

        print("=================================")
        print(landmarks_comp)
        print("=================================")

        if not exercise_started:
            # Comparar os movimentos do vídeo de comparação com os do vídeo de referência
            similarity_scores = np.mean(np.abs(landmarks_comp - ref_landmarks), axis=(1, 2))
            similarity_score = np.min(similarity_scores)
            similarity_threshold = 0.5

            print("CAI SIM OTARIO")

            if similarity_score < similarity_threshold:
                exercise_started = True
        else:
            similarity_scores = np.mean(np.abs(landmarks_comp - ref_landmarks_points[current_index]), axis=1)
            similarity_score = np.min(similarity_scores)
            similarity_threshold = 0.5

            print("=================================")
            print(similarity_scores)
            print("=================================")

            if similarity_score >= similarity_threshold:
                current_index += 1
                exercise_started = False

                if current_index == len(ref_landmarks):
                    current_index = 0
                    exercises_count += 1

        # Mostrar o resultado com os pontos corporais desenhados para o vídeo de comparação
        annotated_image_comp = frame_comp.copy()
        mp_drawing.draw_landmarks(
            annotated_image_comp, results_comp.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Exibir o texto indicando a similaridade dos movimentos
        exercise_count_text = f"Exercises Completed: {exercises_count}"
        similarity_text = f"Similarity: {similarity_score:.2f}"
        cv2.putText(annotated_image_comp, exercise_count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_image_comp, similarity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Comparison Video', annotated_image_comp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_comp.release()
    cv2.destroyAllWindows()
