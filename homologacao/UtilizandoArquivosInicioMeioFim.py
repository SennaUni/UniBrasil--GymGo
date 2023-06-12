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
    except FileNotFoundError:
        ref_landmarks = None

    # Open the reference video
    cap_ref = cv2.VideoCapture(video_ref_path)

    if ref_landmarks is None:
        ref_landmarks = []

        cap_ref.set(cv2.CAP_PROP_POS_MSEC, milissegundos_initial)
        ret_ref, frame_ref = cap_ref.read()
        results_ref = pose.process(cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB))
        if results_ref.pose_landmarks is not None:
            landmarks_initial = np.array([[lm.x, lm.y, lm.z] for lm in results_ref.pose_landmarks.landmark])
            ref_landmarks.append(landmarks_initial)

        cap_ref.set(cv2.CAP_PROP_POS_MSEC, milissegundos_middle)
        ret_ref, frame_ref = cap_ref.read()
        results_ref = pose.process(cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB))
        if results_ref.pose_landmarks is not None:
            landmarks_middle = np.array([[lm.x, lm.y, lm.z] for lm in results_ref.pose_landmarks.landmark])
            ref_landmarks.append(landmarks_middle)

        cap_ref.set(cv2.CAP_PROP_POS_MSEC, milissegundos_final)
        ret_ref, frame_ref = cap_ref.read()
        results_ref = pose.process(cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB))
        if results_ref.pose_landmarks is not None:
            landmarks_final = np.array([[lm.x, lm.y, lm.z] for lm in results_ref.pose_landmarks.landmark])
            ref_landmarks.append(landmarks_final)

        annotated_image_ref = frame_ref.copy()
        mp_drawing.draw_landmarks(
            annotated_image_ref, results_ref.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Reference Video', annotated_image_ref)
        cv2.waitKey(0)

        ref_landmarks = np.array(ref_landmarks)
        np.save(landmarks_file, ref_landmarks)

    cap_ref.release()
    cv2.destroyAllWindows()

    cap_comp = cv2.VideoCapture(video_comp_path)

    exercises_count = 0
    exercise_started = True  # Set to True initially
    current_index = 0

    while cap_comp.isOpened():
        ret_comp, frame_comp = cap_comp.read()

        if not ret_comp:
            break

        frame_comp_rgb = cv2.cvtColor(frame_comp, cv2.COLOR_BGR2RGB)
        results_comp = pose.process(frame_comp_rgb)

        if results_comp.pose_landmarks is None:
            continue

        landmarks_comp = np.array([[lm.x, lm.y, lm.z] for lm in results_comp.pose_landmarks.landmark])

        if not exercise_started:
            similarity_scores = np.mean(np.abs(landmarks_comp - ref_landmarks[0]), axis=1)
            similarity_score = np.min(similarity_scores)
            similarity_threshold = 0.5

            if similarity_score < similarity_threshold:
                exercise_started = True
        else:
            similarity_scores = np.mean(np.abs(landmarks_comp - ref_landmarks[current_index]), axis=1)
            similarity_score = np.min(similarity_scores)
            similarity_threshold = 0.5

            if similarity_score >= similarity_threshold:
                current_index += 1
                exercise_started = False

                if current_index == len(ref_landmarks):
                    current_index = 0
                    exercises_count += 1

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
