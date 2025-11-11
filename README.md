# APS_6_SEMESTRE
Projeto de Visão Computacional para a disciplina de Processamento de Imagem e Visão Computacional


🔒 Projeto: Cofre de Segurança com Reconhecimento Facial


Este é um projeto acadêmico para a disciplina de Visão Computacional, desenvolvido como um protótipo de um cofre de segurança de múltiplos níveis. O sistema utiliza reconhecimento facial em tempo real para conceder ou negar acesso.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

📝 Descrição
O sistema simula um cofre de alta segurança com três níveis hierárquicos de permissão:

Nível 1: Acesso geral (operadores).

Nível 2: Acesso restrito (diretores de divisão).

Nível 3: Acesso exclusivo (autoridade máxima, ex: Ministro).

A aplicação utiliza a webcam para capturar o vídeo, identificar um rosto e compará-lo com um banco de dados de rostos autorizados. Uma interface gráfica (GUI) exibe o status da operação em tempo real (Aguardando, Acesso Liberado, Acesso Negado), o nome da pessoa reconhecida e seu nível de permissão.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

✨ Principais Funcionalidades
Reconhecimento em Tempo Real: Identificação facial processada frame a frame.

Banco de Dados de Rostos: Sistema de codificação que processa imagens de pessoas autorizadas e armazena suas "assinaturas" faciais.

Níveis de Acesso: Lógica para atribuir diferentes permissões com base no usuário.

Interface Gráfica (GUI): Tela amigável construída com PyQt5 que exibe o feed da câmera e o status do sistema.

Feedback Visual: Imagens de status (liberado.png, negado.png) atualizadas dinamicamente.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔧 Tecnologias Utilizadas
Python 3.10

Anaconda: Gerenciador de pacotes e ambientes.

OpenCV: Para captura e processamento de imagem da webcam.

dlib: A biblioteca C++ que serve de motor para o reconhecimento facial.

face_recognition: Uma biblioteca Python que simplifica o uso da dlib para encontrar, codificar e comparar rostos.

PyQt5: A biblioteca utilizada para construir a interface gráfica do usuário.

Numpy: Para manipulação eficiente de arrays de imagem.

Pickle: Para serializar e salvar o banco de dados de codificações faciais.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 Instalação (O Caminho Correto)
Para evitar os conflitos de DLL que encontramos, a instalação deste projeto deve ser feita de uma maneira muito específica.

💡 O Desafio: pip vs. conda-forge
Durante o desenvolvimento, enfrentamos um erro crítico e persistente: ImportError: DLL load failed while importing QtWidgets.

Causa: Este erro acontece devido a um conflito entre as bibliotecas C++ compiladas pelo pip e as compiladas pelo conda.

O Conflito: Tentar instalar o PyQt6 (via pip) e o opencv (via conda-forge) no mesmo ambiente causa uma falha na inicialização das DLLs do PyQt.

A Solução Definitiva: A única solução 100% estável foi usar exclusivamente o conda-forge para instalar todos os pacotes pesados (dlib, opencv, pyqt). Como o PyQt6 não está no conda-forge, migramos o projeto para o PyQt5, que está disponível e é mantido no conda-forge.

Passos para a Instalação
Siga estes passos no Terminal Anaconda (Anaconda Prompt).

1. Crie um Ambiente Anaconda Limpo

Este comando único cria um novo ambiente chamado "cofre" e já instala todas as dependências necessárias de forma compatível, direto do conda-forge.

conda create -n cofre -c conda-forge python=3.10 dlib opencv face_recognition pyqt=5

2. Ative o Ambiente

Sempre que for trabalhar no projeto, ative o ambiente com:

conda activate cofre_final

Nenhuma outra instalação via pip é necessária, pois a biblioteca face_recognition já foi instalada pelo conda (ela vem junto com dlib e opencv no conda-forge).

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

📁 Estrutura do Projeto
O projeto deve seguir esta organização de pastas para que os caminhos (../codificacoes.pkl) funcionem corretamente.

/APS-6-SEMESTRE/
│
├── database/
│   ├── nivel-1/
│   │   ├── (fotos .jpg/.png)
│   ├── nivel-2/
│   │   ├── (fotos .jpg/.png)
│   ├── nivel-3/
│   │   ├── (fotos .jpg/.png)
│   │
│   ├── autenticador_de_permissao.py  # 👈 Script principal da GUI
│   └── codificador_de_faces.py     # 👈 Script para processar a database
│
├── test/
│   └── primeiro_autenticador.py      # Pasta de testes (ignorar na produção)
│
└── codificacoes.pkl

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

▶️ Modo de Uso
Passo 1: Alimentar o Banco de Dados

Adicione fotos nítidas (preferencialmente .jpg ou .png) das pessoas autorizadas dentro das pastas correspondentes (database/nivel_1, database/nivel_2, database/nivel_3). O nome do arquivo será usado como o nome da pessoa.

Passo 2: Gerar as Codificações

Antes de executar o cofre pela primeira vez (ou sempre que adicionar novas fotos), você deve rodar o script de codificação.

Ative o ambiente: conda activate cofre

Navegue até a pasta raiz (APS-6-SEMESTRE/)

Execute: python codificador_de_faces.py

Isso irá criar (ou atualizar) o arquivo codificacoes.pkl, que contém as "assinaturas" faciais.

Passo 3: Executar o Sistema

Com o banco de dados gerado e os arquivos .png de status na pasta raiz, execute a aplicação principal:

python database/autenticador_de_permissao.py
O sistema será iniciado, a câmera será ativada e o reconhecimento começará.
