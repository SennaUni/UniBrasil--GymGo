import cv2
import numpy as np

def comparar_polichinelos(polichinelos_ref, polichinelos_cmp, limite_distancia=5):
    # Comparação da distância entre os polichinelos
    distancias = np.linalg.norm(polichinelos_ref - polichinelos_cmp, axis=2)
    return np.all(distancias < limite_distancia)

# Carregar os vídeos de referência e comparação
video_ref = cv2.VideoCapture('caminho_do_video_referencia.mp4')
video_cmp = cv2.VideoCapture('caminho_do_video_comparacao.mp4')

while True:
    # Ler os frames dos vídeos de referência e comparação
    ret_ref, frame_ref = video_ref.read()
    ret_cmp, frame_cmp = video_cmp.read()

    # Verificar se ambos os frames foram lidos corretamente
    if not ret_ref or not ret_cmp:
        break

    # Converter os frames para escala de cinza
    frame_gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
    frame_gray_cmp = cv2.cvtColor(frame_cmp, cv2.COLOR_BGR2GRAY)

    # Parâmetros para o algoritmo Lucas-Kanade
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Detectar os pontos de interesse no frame de referência
    corners_ref = cv2.goodFeaturesToTrack(frame_gray_ref, maxCorners=100, qualityLevel=0.3, minDistance=7)

    if corners_ref is not None:
        # Rastrear os pontos de interesse no frame de comparação
        corners_cmp, status, _ = cv2.calcOpticalFlowPyrLK(frame_gray_ref, frame_gray_cmp, corners_ref, None, **lk_params)

        # Selecionar apenas os pontos que foram rastreados corretamente
        corners_ref = corners_ref[status == 1]
        corners_cmp = corners_cmp[status == 1]

        # Desenhar os pontos de interesse no frame de comparação
        for corner_ref, corner_cmp in zip(corners_ref, corners_cmp):
            x_ref, y_ref = corner_ref.ravel().astype(int)
            x_cmp, y_cmp = corner_cmp.ravel().astype(int)
            cv2.circle(frame_cmp, (x_cmp, y_cmp), 3, (0, 0, 255), -1)

        # Comparação dos polichinelos detectados
        polichinelos_ref = corners_ref.reshape(-1, 1, 2).astype(np.float32)
        polichinelos_cmp = corners_cmp.reshape(-1, 1, 2).astype(np.float32)
        execucao_correta = comparar_polichinelos(polichinelos_ref, polichinelos_cmp)
    else:
        execucao_correta = False

    # Desenho do resultado da comparação
    if execucao_correta:
        cv2.putText(frame_cmp, "Execução Correta", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame_cmp, "Execução Incorreta", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Redimensionar a janela de exibição
    frame_ref = cv2.resize(frame_ref, (640, 480))
    frame_cmp = cv2.resize(frame_cmp, (640, 480))

    # Exibir os frames lado a lado
    frame_comparacao = np.hstack((frame_ref, frame_cmp))

    # Mostrar o resultado
    cv2.imshow("Comparação de Polichinelos", frame_comparacao)

    # Verificar se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
video_ref.release()
video_cmp.release()
cv2.destroyAllWindows()
