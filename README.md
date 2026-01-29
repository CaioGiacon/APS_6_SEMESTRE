# 🔒 Cofre de Segurança com Reconhecimento Facial

> **APS 6º Semestre** - Disciplina de Processamento de Imagem e Visão Computacional.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Conda](https://img.shields.io/badge/Manager-Anaconda-green?logo=anaconda&logoColor=white)
![OpenCV](https://img.shields.io/badge/Lib-OpenCV-red?logo=opencv&logoColor=white)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green?logo=qt&logoColor=white)

Este projeto acadêmico simula um cofre de alta segurança que utiliza **reconhecimento facial em tempo real** para gerenciar o acesso. O sistema identifica usuários e concede permissões baseadas em uma hierarquia de três níveis.

---

## 📋 Funcionalidades

* **Reconhecimento Facial em Tempo Real:** Identificação instantânea processada frame a frame via webcam.
* **Controle de Acesso Hierárquico:**
    * 🔓 **Nível 1:** Acesso Geral (Operadores).
    * 🔐 **Nível 2:** Acesso Restrito (Diretores).
    * ⛔ **Nível 3:** Acesso Exclusivo (Autoridade Máxima).
* **Interface Gráfica (GUI):** Desenvolvida em **PyQt5**, exibe o feed da câmera, nome do usuário e status de acesso.
* **Feedback Visual:** Indicadores dinâmicos de "Acesso Liberado" ou "Acesso Negado".
* **Banco de Dados Persistente:** Armazena as "assinaturas faciais" (encodings) para verificação rápida.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10
* **Visão Computacional:** OpenCV, Dlib, Face_recognition
* **Interface:** PyQt5
* **Gerenciamento de Dados:** Numpy, Pickle
* **Ambiente:** Anaconda (Essencial para este projeto)

---

## ⚠️ Instalação (Importante)

> **Nota Crítica:** Este projeto possui dependências sensíveis (especialmente `dlib` e `PyQt`). Para evitar conflitos de DLL conhecidos, siga estritamente os passos abaixo utilizando o **Anaconda**.

### 1. Criar o Ambiente
Abra o **Anaconda Prompt** e execute o comando abaixo. Isso criará um ambiente limpo chamado `cofre` e instalará todas as dependências via `conda-forge` para garantir compatibilidade.

```bash
conda create -n cofre -c conda-forge python=3.10 dlib opencv face_recognition pyqt=5
conda activate cofre
```
## 📂 Estrutura do Projeto
Certifique-se de que seus arquivos estejam organizados desta forma para que os scripts encontrem o banco de dados:

```APS-6-SEMESTRE/
│
├── database/
│   ├── nivel-1/             # Fotos dos Operadores (.jpg/.png)
│   ├── nivel-2/             # Fotos dos Diretores
│   ├── nivel-3/             # Fotos das Autoridades
│   │
│   ├── autenticador_de_permissao.py   # 🏁 Script Principal (GUI)
│   └── codificador_de_faces.py        # ⚙️ Script para gerar banco de dados
│
└── codificacoes.pkl         # Arquivo gerado com as assinaturas faciais
```

## ▶️ Como Usar

Passo 1: Cadastrar Usuários
Coloque fotos nítidas (frontal, boa iluminação) das pessoas nas pastas correspondentes dentro de database/ (nivel-1, nivel-2, etc.). O nome do arquivo será usado como o nome do usuário na tela.

Passo 2: Gerar Codificações
Antes de rodar o programa pela primeira vez (ou após adicionar novas fotos), processe as imagens para criar o arquivo de reconhecimento:

```bash
python database/codificador_de_faces.py
```
Isso criará/atualizará o arquivo codificacoes.pkl na raiz.

Passo 3: Executar o Cofre
Com o ambiente ativado e as codificações geradas, inicie o sistema:
```bash
python database/autenticador_de_permissao.py
```
A interface abrirá e a câmera começará a buscar por rostos autorizados.

## 👤 Autor
Desenvolvido por Caio Giacon.
Projeto desenvolvido para fins educacionais na disciplina de Visão Computacional.
