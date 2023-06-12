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

# Caminho do vídeo de referência
video_referencia = 'videos/Polichinelo.mp4'

# Inicialização do vídeo de referência
video_ref = cv2.VideoCapture(video_referencia)

# Verificação se o vídeo de referência foi aberto corretamente
if not video_ref.isOpened():
    print("Erro ao abrir o vídeo de referência.")
    exit()

# Inicialização da câmera
video_cam = cv2.VideoCapture(0)  # Utilize 0 para a câmera padrão, ou especifique um número de câmera

# Verificação se a câmera foi aberta corretamente
if not video_cam.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Inicialização do subtrator de fundo
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Dimensões da tela de exibição
largura_tela = 800
altura_tela = 600

while True:
    # Leitura do próximo frame do vídeo de referência
    ret_ref, frame_ref = video_ref.read()

    # Verificação se não há mais frames para ler no vídeo de referência
    if not ret_ref:
        print("Fim do vídeo de referência.")
        break

    # Leitura do próximo frame da câmera
    ret_cam, frame_cam = video_cam.read()

    # Verificação se não há mais frames para ler na câmera
    if not ret_cam:
        print("Erro ao ler o frame da câmera.")
        break

    # Espelhamento horizontal do frame da câmera (opcional)
    frame_cam = cv2.flip(frame_cam, 1)

    # Redimensionamento do frame da câmera
    frame_cam = cv2.resize(frame_cam, (largura_tela // 2, altura_tela))

    # Redimensionamento do frame de referência
    frame_ref = cv2.resize(frame_ref, (largura_tela // 2, altura_tela))

    # Detecção de polichinelos no vídeo de referência
    polichinelos_ref = detectar_polichinelos(frame_ref, bg_subtractor)

    # Desenho dos polichinelos detectados no frame do vídeo de referência
    cv2.drawContours(frame_ref, polichinelos_ref, -1, (0, 255, 0), 2)

    # Detecção de polichinelos na captura da câmera
    polichinelos_cam = detectar_polichinelos(frame_cam, bg_subtractor)

    # Desenho dos polichinelos detectados no frame da câmera
    cv2.drawContours(frame_cam, polichinelos_cam, -1, (0, 255, 0), 2)

    # Comparação dos polichinelos detectados
    execucao_correta = comparar_polichinelos(polichinelos_ref, polichinelos_cam)

    # Desenho do resultado da comparação
    if execucao_correta:
        cv2.putText(frame_cam, "Execução Correta", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame_cam, "Execução Incorreta", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Criação da tela de exibição
    tela_exibicao = np.zeros((altura_tela, largura_tela, 3), dtype=np.uint8)
    tela_exibicao[:altura_tela, :largura_tela // 2] = frame_ref
    tela_exibicao[:altura_tela, largura_tela // 2:] = frame_cam

    # Exibição da tela de exibição
    cv2.imshow('Comparação: Vídeo de Referência x Câmera', tela_exibicao)

    # Verificação de tecla de saída (pressione 'q' para encerrar)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberação dos recursos
video_ref.release()
video_cam.release()
cv2.destroyAllWindows()
