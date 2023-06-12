import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Caminho dos vídeos
video_ref_path = 'caminho_do_video_referencia.mp4'
video_comp_path = 'caminho_do_video_comparacao.mp4'

# Carregar os vídeos
video_ref = cv2.VideoCapture(video_ref_path)
video_comp = cv2.VideoCapture(video_comp_path)

# Configuração do detector de keypoints
detector = cv2.ORB_create()

while True:
    # Ler um frame de cada vídeo
    ret_ref, frame_ref = video_ref.read()
    ret_comp, frame_comp = video_comp.read()

    # Verificar se os frames foram lidos corretamente
    if not ret_ref or not ret_comp:
        break

    # Converter os frames para escala de cinza
    frame_ref_gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
    frame_comp_gray = cv2.cvtColor(frame_comp, cv2.COLOR_BGR2GRAY)

    # Detectar keypoints nos frames de referência e comparação
    keypoints_ref, descriptors_ref = detector.detectAndCompute(frame_ref_gray, None)
    keypoints_comp, descriptors_comp = detector.detectAndCompute(frame_comp_gray, None)

    # Emparelhamento dos keypoints
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_ref, descriptors_comp)

    # Ordenar os matches por distância
    matches = sorted(matches, key=lambda x: x.distance)

    # Desenhar os matches nos frames
    frame_matches = cv2.drawMatches(frame_ref, keypoints_ref, frame_comp, keypoints_comp, matches[:10], None)

    # Calcular o índice SSIM entre os frames
    ssim_score = ssim(frame_ref_gray, frame_comp_gray)

    # Mostrar o resultado
    cv2.imshow("Comparacao de Videos", frame_matches)
    print(f"Índice SSIM: {ssim_score}")

    # Verificar se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
video_ref.release()
video_comp.release()
cv2.destroyAllWindows()
