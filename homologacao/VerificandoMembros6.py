import cv2
from openpose import pyopenpose as op
import numpy as np

# Configurar os vídeos de referência e comparação
video_ref_path = 'videos/Polichinelo.mp4'
video_comp_path = 'videos/PolichineloMato.mp4'

# Caminho do arquivo a ser lido ou gerado
landmarks_file = 'referencias/Polichinelo.npy'

# Inicializar OpenPose
params = {
    'model_folder': 'caminho/para/pasta/openpose/models'
}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Inicializar os modelos de detecção de pose
datum = op.Datum()

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

        datum.cvInputData = frame_ref
        opWrapper.emplaceAndPop([datum])

        if datum.poseKeypoints is None:
            continue

        landmarks_ref = datum.poseKeypoints[:, :, :2]
        ref_landmarks.append(landmarks_ref)

        # Mostrar o resultado com os pontos corporais desenhados para o vídeo de referência
        annotated_image_ref = datum.cvOutputData
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

    datum.cvInputData = frame_comp
    opWrapper.emplaceAndPop([datum])

    if datum.poseKeypoints is None:
        continue

    landmarks_comp = datum.poseKeypoints[:, :, :2]

    # Verificar se todos os membros estão presentes
    all_members_present = True
    missing_members = []
    # Verificar cada membro
    # ...

    if not all_members_present:
        missing_message = "Membros ausentes: "
        for member in missing_members:
            missing_message += f"{member}, "
        missing_message = missing_message[:-2]  # Remover a vírgula final e espaço
        cv2.putText(frame_comp, missing_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Video de Comparação', frame_comp)
        continue

    # Comparar os movimentos do vídeo de comparação com os do vídeo de referência
    similarity_scores = np.mean(np.abs(landmarks_comp - ref_landmarks), axis=(1, 2))
    similarity_score = np.min(similarity_scores)

    # Mostrar o resultado com os pontos corporais desenhados para o vídeo de comparação
    annotated_image_comp = datum.cvOutputData
    cv2.putText(annotated_image_comp, f"Similaridade: {similarity_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)
    cv2.imshow('Video de Comparação', annotated_image_comp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos do vídeo de comparação
cap_comp.release()
cv2.destroyAllWindows()
