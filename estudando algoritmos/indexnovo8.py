import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

# Carregar o vídeo
video = cv2.VideoCapture('caminho_do_video.mp4')

# Ler o primeiro frame para definir a forma das máscaras
ret, frame = video.read()
height, width, _ = frame.shape

# Criar máscaras vazias com a mesma forma do vídeo
markers = np.zeros((height, width), dtype=np.int32)
mask = np.zeros((height, width), dtype=np.uint8)

while True:
    # Ler o frame do vídeo
    ret, frame = video.read()

    # Verificar se o frame foi lido corretamente
    if not ret:
        break

    # Converter o frame para escala de cinza
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar um limiar adaptativo para segmentar os objetos
    _, thresh = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Aplicar a segmentação de watershed
    distance_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    local_maxima = peak_local_max(distance_transform, min_distance=20, labels=thresh)
    markers.fill(0)  # Limpar as máscaras antes de atualizar
    for marker in local_maxima:
        markers[tuple(marker)] = 255  # Definir o valor máximo nas posições dos marcadores
    labels = watershed(-distance_transform, markers, mask=thresh)

    # Encontrar os contornos dos objetos
    contours, _ = cv2.findContours((labels > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar os contornos nos objetos encontrados
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Mostrar o resultado
    cv2.imshow("Rastreamento de Objetos", frame)

    # Verificar se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
video.release()
cv2.destroyAllWindows()
