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

    # Selecionar os melhores matches
    num_matches = 10
    best_matches = matches[:num_matches]

    # Verificar se há número mínimo de correspondências
    if len(best_matches) < 4:
        print("Número insuficiente de correspondências de pontos.")
        continue

    # Obter os pontos correspondentes nos keypoints de referência e comparação
    src_points = np.float32([keypoints_ref[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints_comp[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

    # Calcular a matriz de transformação (homografia)
    M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Aplicar a transformação nos contornos do vídeo de referência
    h, w = frame_ref_gray.shape
    ref_contour = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    transformed_contour = cv2.perspectiveTransform(ref_contour, M)

    # Desenhar o contorno no frame de comparação
    frame_comp_with_contour = cv2.polylines(frame_comp, [np.int32(transformed_contour)], True, (0, 255, 0), 2)

    # Calcular o índice SSIM entre os frames
    ssim_score = ssim(frame_ref_gray, frame_comp_gray)

    # Mostrar os resultados
    cv2.imshow("Video de Referencia", frame_ref)
    cv2.imshow("Video de Comparacao", frame_comp_with_contour)
    print(f"Índice SSIM: {ssim_score}")

    # Verificar se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
video_ref.release()
video_comp.release()
cv2.destroyAllWindows()
