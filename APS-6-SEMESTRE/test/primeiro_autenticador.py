import cv2
import face_recognition
import pickle
import numpy as np

# Carregar os dados de codificação salvos
print("Carregando base de dados de rostos...")
with open("codificacoes.pkl", "rb") as f:
    dados_codificados = pickle.load(f)

codificacoes_conhecidas = dados_codificados["codificacoes"]
metadados_conhecidos = dados_codificados["metadados"]
print("Base de dados carregada.")

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    sucesso, frame = cap.read()
    if not sucesso:
        break
    
    # É mais eficiente processar um frame menor, então vamos redimensioná-lo
    # Isso acelera muito o reconhecimento
    frame_pequeno = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Converter de BGR (OpenCV) para RGB (face_recognition)
    frame_rgb = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)

    # Detectar a localização de todos os rostos no frame atual
    locs_rostos_frame = face_recognition.face_locations(frame_rgb)
    # Codificar os rostos detectados
    cods_rostos_frame = face_recognition.face_encodings(frame_rgb, locs_rostos_frame)

    # Loop em cada rosto encontrado no frame
    for cod_rosto, loc_rosto in zip(cods_rostos_frame, locs_rostos_frame):
        # Comparar o rosto atual com todos os rostos conhecidos
        matches = face_recognition.compare_faces(codificacoes_conhecidas, cod_rosto)
        
        # Calcular a "distância" facial. Quanto menor a distância, mais parecidos são os rostos.
        distancias_rosto = face_recognition.face_distance(codificacoes_conhecidas, cod_rosto)
        
        nome = "Desconhecido"
        nivel_acesso = "Nenhum"
        cor_retangulo = (0, 0, 255) # Vermelho para acesso negado
        
        # Se houver alguma correspondência, encontrar a melhor
        if True in matches:
            melhor_match_idx = np.argmin(distancias_rosto)
            if matches[melhor_match_idx]:
                metadados = metadados_conhecidos[melhor_match_idx]
                nome = metadados["nome"]
                nivel_acesso = metadados["nivel"]
                cor_retangulo = (0, 255, 0) # Verde para acesso concedido

        # Desenhar um retângulo e o texto no frame original (lembre-se que redimensionamos)
        top, right, bottom, left = loc_rosto
        # Escalar de volta para o tamanho original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenhar o retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), cor_retangulo, 2)

        # Preparar o texto a ser exibido
        texto_acesso = f"Acesso: {nivel_acesso.replace('_', ' ').title()}"
        if nome == "Desconhecido":
            texto_acesso = "ACESSO NEGADO"
        
        # Desenhar o fundo para o texto
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), cor_retangulo, cv2.FILLED)
        # Escrever o nome e o nível de acesso
        cv2.putText(frame, nome, (left + 6, bottom - 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(frame, texto_acesso, (left + 6, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    # Exibir o resultado
    cv2.imshow("Sistema de Seguranca do Cofre", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar as janelas
cap.release()
cv2.destroyAllWindows()