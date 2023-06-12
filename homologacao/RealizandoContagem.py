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

# Initialize pose detection models
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    try:
        ref_landmarks = np.load(landmarks_file)
    except FileNotFoundError:
        ref_landmarks = None

    # Open the reference video
    cap_ref = cv2.VideoCapture(video_ref_path)

    if ref_landmarks is None:
        ref_landmarks = []

        # Definir os índices dos landmarks para as posições de referência
        index_initial = 0  # Índice da posição inicial
        index_middle = 100  # Índice da posição média
        index_final = -1  # Índice da posição final

        while cap_ref.isOpened():
            ret_ref, frame_ref = cap_ref.read()

            if not ret_ref:
                break

            frame_ref_rgb = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)
            results_ref = pose.process(frame_ref_rgb)

            if results_ref.pose_landmarks is None:
                continue

            landmarks_ref = np.array([[lm.x, lm.y, lm.z] for lm in results_ref.pose_landmarks.landmark])

            # Adicionar os landmarks das posições de referência à lista
            ref_landmarks.append([
                landmarks_ref[index_initial],
                landmarks_ref[index_middle],
                landmarks_ref[index_final]
            ])

            annotated_image_ref = frame_ref.copy()
            mp_drawing.draw_landmarks(
                annotated_image_ref, results_ref.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Reference Video', annotated_image_ref)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        ref_landmarks = np.array(ref_landmarks)
        np.save(landmarks_file, ref_landmarks)

    cap_ref.release()
    cv2.destroyAllWindows()

    cap_comp = cv2.VideoCapture(video_comp_path)

    exercises_count = 0
    exercise_started = True  # Set to True initially

    while cap_comp.isOpened():
        ret_comp, frame_comp = cap_comp.read()

        if not ret_comp:
            break

        frame_comp_rgb = cv2.cvtColor(frame_comp, cv2.COLOR_BGR2RGB)
        results_comp = pose.process(frame_comp_rgb)

        if results_comp.pose_landmarks is None:
            continue

        landmarks_comp = np.array([[lm.x, lm.y, lm.z] for lm in results_comp.pose_landmarks.landmark])

        # Verificar a sequência correta das posições de referência
        similarity_scores = np.mean(np.abs(landmarks_comp - ref_landmarks), axis=(1, 2))
        similarity_score = np.min(similarity_scores)

        similarity_threshold = 0.5

        if similarity_score < similarity_threshold and exercise_started:
            exercise_started = False
            exercises_count += 1
        elif similarity_score >= similarity_threshold and not exercise_started:
            exercise_started = True

        annotated_image_comp = frame_comp.copy()
        mp_drawing.draw_landmarks(
            annotated_image_comp, results_comp.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        exercise_count_text = f"Exercises Completed: {exercises_count}"
        similarity_text = f"Similarity: {similarity_score:.2f}"
        cv2.putText(annotated_image_comp, exercise_count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_image_comp, similarity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Comparison Video', annotated_image_comp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_comp.release()
    cv2.destroyAllWindows()
