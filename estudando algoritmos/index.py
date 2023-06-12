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

# Inicialização do vídeo
video = cv2.VideoCapture('videos/Polichinelo.mp4')  # Utilize 0 para a câmera padrão, ou especifique um caminho de arquivo

# Inicialização do subtrator de fundo
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Leitura do próximo frame
    ret, frame = video.read()
    
    if not ret:
        break
    
    # Espelhamento horizontal do frame (opcional)
    frame = cv2.flip(frame, 1)
    
    # Detecção de polichinelos
    polichinelos = detectar_polichinelos(frame, bg_subtractor)
    
    # Desenho dos polichinelos detectados no frame
    cv2.drawContours(frame, polichinelos, -1, (0, 255, 0), 2)
    
    # Exibição do frame resultante
    cv2.imshow('Detecção de Polichinelos', frame)
    
    # Verificação de tecla de saída (pressione 'q' para encerrar)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberação dos recursos
video.release()
cv2.destroyAllWindows()