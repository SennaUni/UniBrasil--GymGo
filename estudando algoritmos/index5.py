import cv2
import numpy as np

# Função para detectar polichinelos
def detectar_polichinelos(frame, bg_subtractor):
    # Aplica a subtração de fundo no frame atual
    mask = bg_subtractor.apply(frame)
    
    # Realiza operações de limiarização e contorno na máscara
    _, thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Encontra os contornos com área suficiente para serem considerados polichinelos
    min_area = 1000  # Defina um valor apropriado para sua aplicação
    polichinelos = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            polichinelos.append(contour)
    
    return polichinelos

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

# Definição do tamanho desejado para a tela de exibição
largura_tela = 800
altura_tela = 400

# Inicialização do subtrator de fundo
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

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

    # Espelhamento horizontal do frame do vídeo de comparação (opcional)
    frame_cmp = cv2.flip(frame_cmp, 1)

    # Redimensionamento dos frames para o tamanho desejado
    frame_ref = cv2.resize(frame_ref, (largura_tela // 2, altura_tela))
    frame_cmp = cv2.resize(frame_cmp, (largura_tela // 2, altura_tela))

    # Detecção de polichinelos no vídeo de referência
    polichinelos_ref = detectar_polichinelos(frame_ref, bg_subtractor)

    # Desenho dos polichinelos detectados no frame do vídeo de referência
    cv2.drawContours(frame_ref, polichinelos_ref, -1, (0, 255, 0), 2)

    # Detecção de polichinelos no vídeo de comparação
    polichinelos_cmp = detectar_polichinelos(frame_cmp, bg_subtractor)

    # Desenho dos polichinelos detectados no frame do vídeo de comparação
    cv2.drawContours(frame_cmp, polichinelos_cmp, -1, (0, 255, 0), 2)

    # Comparação dos polichinelos detectados
    execucao_correta = comparar_polichinelos(polichinelos_ref, polichinelos_cmp)

    # Desenho do resultado da comparação
    if execucao_correta:
        cv2.putText(frame_cmp, "Execução Correta", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame_cmp, "Execução Incorreta", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Criação da tela de exibição com tamanho reduzido
    tela_exibicao = np.zeros((altura_tela, largura_tela, 3), dtype=np.uint8)
    tela_exibicao[:, :largura_tela // 2] = frame_ref
    tela_exibicao[:, largura_tela // 2:] = frame_cmp

    # Exibição da tela de exibição
    cv2.imshow('Comparação: Vídeo de Referência x Vídeo de Comparação', tela_exibicao)

    # Verificação de tecla de saída (pressione 'q' para encerrar)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberação dos recursos
video_ref.release()
video_cmp.release()
cv2.destroyAllWindows()
