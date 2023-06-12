import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Reshape, Activation

# Função para carregar o modelo de keypoints
def load_keypoint_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(None, None, 3))
    x = base_model.get_layer('block_16_project_BN').output
    x = Conv2D(256, 3, padding='same')(x)
    x = Conv2D(17, 1, activation='sigmoid', name='keypoints')(x)
    x = Reshape((-1, 17))(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Carregar o modelo de keypoints
keypoint_model = load_keypoint_model()

# Caminho dos vídeos
video_ref_path = 'caminho_do_video_referencia.mp4'
video_comp_path = 'caminho_do_video_comparacao.mp4'

# Carregar os vídeos
video_ref = cv2.VideoCapture(video_ref_path)
video_comp = cv2.VideoCapture(video_comp_path)

# Obter o tamanho esperado para redimensionar os frames
input_shape = keypoint_model.input_shape[1:3]

while True:
    # Ler um frame de cada vídeo
    ret_ref, frame_ref = video_ref.read()
    ret_comp, frame_comp = video_comp.read()

    # Verificar se os frames foram lidos corretamente
    if not ret_ref or not ret_comp:
        break

    # Redimensionar os frames para o tamanho esperado pelo modelo
    frame_ref_resized = cv2.resize(frame_ref, (input_shape[1], input_shape[0]))
    frame_comp_resized = cv2.resize(frame_comp, (input_shape[1], input_shape[0]))

    # Normalizar os pixels dos frames
    frame_ref_normalized = frame_ref_resized / 255.0
    frame_comp_normalized = frame_comp_resized / 255.0

    # Executar a inferência do modelo nos frames
    keypoints_ref = keypoint_model.predict(np.expand_dims(frame_ref_normalized, axis=0))[0]
    keypoints_comp = keypoint_model.predict(np.expand_dims(frame_comp_normalized, axis=0))[0]

    # Desenhar os keypoints nos frames
    for kp in keypoints_ref:
        x, y = kp
        cv2.circle(frame_ref, (int(x), int(y)), 3, (0, 0, 255), -1)
    for kp in keypoints_comp:
        x, y = kp
        cv2.circle(frame_comp, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Mostrar os resultados
    cv2.imshow("Video de Referencia", frame_ref)
    cv2.imshow("Video de Comparacao", frame_comp)

    # Verificar se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
video_ref.release()
video_comp.release()
