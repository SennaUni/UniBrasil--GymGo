import cv2
import numpy as np

# Função para comparar os polichinelos detectados
def comparar_polichinelos(polichinelos_ref, polichinelos_cam):
    if len(polichinelos_ref) != len(polichinelos_cam):
        return False

    for i in range(len(polichinelos_ref)):
        (x_ref, y_ref, w_ref, h_ref) = cv2.boundingRect(polichinelos_ref[i])
        (x_cam, y_cam, w_cam, h_cam) = cv2.boundingRect(polichinelos_cam[i])

        if abs(x_ref - x_cam) > 10 or abs(y_ref - y_cam) > 10:
            return False

    return True

# Caminhos dos vídeos de referência e comparação
video_referencia = 'videos/Polichinelo.mp4'
video_comparacao = 'videos/Polichinelo.mp4'

# Inicialização dos vídeos de referência e comparação
video_ref = cv2.VideoCapture(video_referencia)
video_cmp = cv2.VideoCapture(video_comparacao)

# Verificação se os vídeos foram abertos corretamente
if not video_ref.isOpened():
    print("Erro ao abrir o vídeo de referência.")
    exit()

if not video_cmp.isOpened():
    print("Erro ao abrir o vídeo de comparação.")
    exit()

# Inicialização do rastreador Optical Flow (Lucas-Kanade)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Dimensões desejadas para a tela de exibição (ajuste conforme necessário)
largura_tela = 800
altura_tela = 600

while True:
    # Leitura do próximo frame do vídeo de referência
    ret_ref, frame_ref = video_ref.read()

    # Verificação se não há mais frames para ler no vídeo de referência
    if not ret_ref:
        print("Fim do vídeo de referência.")
        break

    # Leitura do próximo frame do vídeo de comparação
    ret_cmp, frame_cmp = video_cmp.read()

    # Verificação se não há mais frames para ler no vídeo de comparação
    if not ret_cmp:
        print("Fim do vídeo de comparação.")
        break

    # Redimensionar os frames para o tamanho desejado
    frame_ref = cv2.resize(frame_ref, (largura_tela // 2, altura_tela))
    frame_cmp = cv2.resize(frame_cmp, (largura_tela // 2, altura_tela))

    # Convertendo os frames para escala de cinza
    frame_gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
    frame_gray_cmp = cv2.cvtColor(frame_cmp, cv2.COLOR_BGR2GRAY)

    # Detecção de pontos-chave nos dois frames
    keypoints_ref = cv2.goodFeaturesToTrack(frame_gray_ref, maxCorners=100, qualityLevel=0.3, minDistance=7)
    keypoints_cmp, status, _ = cv2.calcOpticalFlowPyrLK(frame_gray_ref, frame_gray_cmp, keypoints_ref, None, **lk_params)

    # Filtrando os pontos-chave que foram bem rastreados
    keypoints_ref = keypoints_ref[status.ravel() == 1]
    keypoints_cmp = keypoints_cmp[status.ravel() == 1]

    # Desenho dos pontos-chave nos frames de referência e comparação
    for kp_ref, kp_cmp in zip(keypoints_ref, keypoints_cmp):
        x_ref, y_ref = kp_ref.ravel().astype(int)
        x_cmp, y_cmp = kp_cmp.ravel().astype(int)
        cv2.circle(frame_ref, (x_ref, y_ref), 3, (0, 0, 255), -1)
        cv2.circle(frame_cmp, (x_cmp, y_cmp), 3, (0, 0, 255), -1)

    # Comparação dos polichinelos detectados
    polichinelos_ref = keypoints_ref.reshape(-1, 1, 2).astype(np.float32)
    polichinelos_cmp = keypoints_cmp.reshape(-1, 1, 2).astype(np.float32)
    execucao_correta = comparar_polichinelos(polichinelos_ref, polichinelos_cmp)

    # Desenho do resultado da comparação
    if execucao_correta:
        cv2.putText(frame_cmp, "Execução Correta", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame_cmp, "Execução Incorreta", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Criação da tela de exibição
    tela_exibicao = np.hstack((frame_ref, frame_cmp))

    # Redimensionar a tela de exibição para o tamanho desejado
    tela_exibicao = cv2.resize(tela_exibicao, (largura_tela, altura_tela))

    # Exibição da tela de exibição
    cv2.imshow('Comparação: Vídeo de Referência x Vídeo de Comparação', tela_exibicao)

    # Verificação de tecla de saída (pressione 'q' para encerrar)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberação dos recursos
video_ref.release()
video_cmp.release()
cv2.destroyAllWindows()
