import sys
import os
import cv2
import face_recognition
import pickle
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

class CofreApp(QWidget):
    """
    Classe principal da aplicação do Cofre de Segurança Facial.
    
    Esta classe gerencia a interface do usuário (PyQt5), a captura
    de vídeo (OpenCV) e a lógica de reconhecimento facial (face_recognition).
    """
    
    # === Constantes de Estilo ===
    STYLE_JANELA = "background-color: #101010; color: white;"
    STYLE_BTN = """
        QPushButton {
            background-color: #444; color: white;
            border-radius: 10px; padding: 10px 20px;
        }
        QPushButton:hover { background-color: #666; }
    """
    STYLE_STATUS_INICIAL = "background-color: #222; border-radius: 8px; padding: 12px;"
    STYLE_STATUS_NEGADO = "background-color: #8b0000; color: white; padding: 10px; border-radius: 8px;"
    STYLE_STATUS_LIBERADO = "background-color: #006400; color: white; padding: 10px; border-radius: 8px;"
    
    FONT_STATUS = QFont("Arial", 20, QFont.Weight.Bold)
    FONT_BTN = QFont("Arial", 14)

    def __init__(self):
        super().__init__()

        # Inicializa variáveis de estado
        self.cap = None
        self.codificacoes_conhecidas = []
        self.metadados_conhecidos = []
        self.erros_face_seq = 0 # Contador de erros de reconhecimento

        # Configura os componentes
        self.init_ui()
        self.carregar_codificacoes()
        self.init_camera()

        # Inicia o loop de processamento
        self.timer = QTimer()
        self.timer.timeout.connect(self.atualizar_frame_loop)
        self.timer.start(40) # ~25 FPS

    # ------------------------------------------------------------------
    # 1. MÉTODOS DE INICIALIZAÇÃO (setup)
    # ------------------------------------------------------------------

    def init_ui(self):
        """Configura a janela principal e todos os widgets da interface."""
        
        # === Config Janela ===
        self.setWindowTitle("🔒 Sistema de Segurança do Cofre")
        self.resize(1000, 700)
        self.setStyleSheet(self.STYLE_JANELA)

        # === Widgets ===
        self.video_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.status_label = QLabel("Inicializando reconhecimento facial...")
        self.status_label.setFont(self.FONT_STATUS)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(self.STYLE_STATUS_INICIAL)

        self.img_status = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.img_status.setPixmap(QPixmap("aguardando.png").scaled(250, 250, Qt.AspectRatioMode.KeepAspectRatio))

        self.btn_sair = QPushButton("Encerrar Sistema")
        self.btn_sair.setFont(self.FONT_BTN)
        self.btn_sair.setStyleSheet(self.STYLE_BTN)
        self.btn_sair.clicked.connect(self.close)

        # === Layout ===
        layout_principal = QVBoxLayout()
        layout_principal.addWidget(self.video_label, stretch=3)
        layout_principal.addWidget(self.status_label)
        layout_principal.addWidget(self.img_status, stretch=1)

        layout_botao = QHBoxLayout()
        layout_botao.addStretch(1)
        layout_botao.addWidget(self.btn_sair)
        layout_botao.addStretch(1)
        
        layout_principal.addLayout(layout_botao)
        self.setLayout(layout_principal)

    def carregar_codificacoes(self):
        """Carrega o arquivo .pkl com os rostos conhecidos."""
        print("Carregando banco de dados de rostos...")
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            pkl_path = os.path.join(base_dir, "..", "codificacoes.pkl")

            if not os.path.exists(pkl_path):
                raise FileNotFoundError

            with open(pkl_path, "rb") as f:
                dados = pickle.load(f)

            self.codificacoes_conhecidas = dados["codificacoes"]
            self.metadados_conhecidos = dados["metadados"]
            print("Base carregada com sucesso!")

        except FileNotFoundError:
            QMessageBox.critical(self, "Erro", f"Arquivo de codificações não encontrado:\n{pkl_path}")
            sys.exit(1)
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar codificações: {e}")
            sys.exit(1)

    def init_camera(self):
        """Inicializa a captura de vídeo, testando múltiplos índices."""
        # Tenta abrir câmera 0 (padrão)
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            # Tenta outras câmeras (1–4) se a 0 falhar
            for i in range(1, 5):
                print(f"Tentando câmera índice {i}...")
                temp_cap = cv2.VideoCapture(i)
                if temp_cap.isOpened():
                    self.cap = temp_cap
                    break

        # Se ainda assim falhar, mostra alerta e fecha
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Erro", "❌ Não foi possível acessar a câmera.\nVerifique as permissões.")
            sys.exit(1)
        
        print("Câmera inicializada com sucesso.")

    # ------------------------------------------------------------------
    # 2. MÉTODOS DO LOOP PRINCIPAL (processamento)
    # ------------------------------------------------------------------

    def atualizar_frame_loop(self):
        """Método principal chamado pelo Timer. Orquestra o processamento."""
        
        # 1. Obter frame da câmera
        frame_rgb = self._get_frame_rgb()
        if frame_rgb is None:
            return # Pula este ciclo se o frame for inválido

        # 2. Processar o reconhecimento facial
        nome, acesso = self._processar_faces(frame_rgb)

        # 3. Atualizar a interface gráfica
        self._atualizar_ui(frame_rgb, nome, acesso)

    def _get_frame_rgb(self):
        """Lê um frame da câmera e converte para RGB."""
        sucesso, frame = self.cap.read()

        if not sucesso or frame is None:
            print("⚠️ Frame vazio — câmera não retornou imagem.")
            return None

        try:
            # Garante tipo e converte BGR (OpenCV) -> RGB (PyQt/face_rec)
            frame = np.array(frame, dtype=np.uint8)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
            return frame_rgb
        except Exception as e:
            print(f"Erro ao converter frame: {e}")
            return None

    def _processar_faces(self, frame_rgb):
        """Executa o reconhecimento facial no frame."""
        
        # Reduz imagem para acelerar
        frame_peq = cv2.resize(frame_rgb, (0, 0), fx=0.25, fy=0.25)
        frame_peq = np.ascontiguousarray(frame_peq, dtype=np.uint8)

        try:
            locs = face_recognition.face_locations(frame_peq)
            encs = face_recognition.face_encodings(frame_peq, locs)
            self.erros_face_seq = 0  # Zera contador se der certo
        except Exception as e:
            self.erros_face_seq += 1
            print(f"Erro no face_recognition ({self.erros_face_seq}): {e}")
            if self.erros_face_seq > 5:
                print("❌ Muitos erros seguidos — pausando temporariamente.")
                self.timer.stop() # Para o timer se o erro for persistente
            return "Desconhecido", "NEGADO"

        # Compara os rostos encontrados com a base de dados
        nome = "Desconhecido"
        acesso = "NEGADO"
        for enc in encs:
            matches = face_recognition.compare_faces(self.codificacoes_conhecidas, enc)
            dist = face_recognition.face_distance(self.codificacoes_conhecidas, enc)
            
            if True in matches:
                i = np.argmin(dist)
                meta = self.metadados_conhecidos[i]
                nome = meta["nome"]
                acesso = meta["nivel"].replace("_", " ").title()
                break # Encontrou um rosto, para o loop
        
        return nome, acesso

    def _atualizar_ui(self, frame_rgb, nome, acesso):
        """Atualiza os labels de status, imagem e vídeo na tela."""
        
        # 1. Atualiza Status (Texto e Imagem)
        if nome == "Desconhecido":
            self.status_label.setText("🚫 Acesso Negado")
            self.status_label.setStyleSheet(self.STYLE_STATUS_NEGADO)
            self.img_status.setPixmap(QPixmap("acesso_negado.png").scaled(250, 250, Qt.AspectRatioMode.KeepAspectRatio))
        else:
            self.status_label.setText(f"✅ Acesso Liberado - {nome} ({acesso})")
            self.status_label.setStyleSheet(self.STYLE_STATUS_LIBERADO)
            self.img_status.setPixmap(QPixmap("acesso_liberado.png").scaled(250, 250, Qt.AspectRatioMode.KeepAspectRatio))

        # 2. Atualiza o feed de Vídeo
        h, w, ch = frame_rgb.shape
        bytes_por_linha = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_por_linha, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    # ------------------------------------------------------------------
    # 3. MÉTODOS DE EVENTOS (fechamento)
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Chamado quando a janela é fechada."""
        print("Encerrando o sistema...")
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()

# ----------------------------------------------------------------------
# PONTO DE ENTRADA DA APLICAÇÃO
# ----------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    janela = CofreApp()
    janela.show()
    sys.exit(app.exec())