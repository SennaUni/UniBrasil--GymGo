import cv2

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

# Caminho do vídeo de referência
video_referencia = "videos/Polichinelo.mp4"

# Inicialização do vídeo de referência
video = cv2.VideoCapture(video_referencia)

# Verificação se o vídeo de referência foi aberto corretamente
if not video.isOpened():
    print("Erro ao abrir o vídeo de referência.")
    exit()

# Inicialização do subtrator de fundo
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Leitura do próximo frame do vídeo de referência
    ret, frame = video.read()

    # Verificação se não há mais frames para ler
    if not ret:
        print("Fim do vídeo de referência.")
        break

    # Espelhamento horizontal do frame (opcional)
    frame = cv2.flip(frame, 1)

    # Detecção de polichinelos
    polichinelos = detectar_polichinelos(frame, bg_subtractor)

    # Desenho dos polichinelos detectados no frame
    cv2.drawContours(frame, polichinelos, -1, (0, 255, 0), 2)

    # Exibição do frame com os polichinelos detectados
    cv2.imshow('Detecção de Polichinelos', frame)

    # Verificação de tecla de saída (pressione 'q' para encerrar)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberação dos recursos
video.release()
cv2.destroyAllWindows()
