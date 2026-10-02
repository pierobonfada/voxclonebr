# 🐟 VoxCloneBR — Fish Audio S2-Pro (Português do Brasil)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Fish Audio](https://img.shields.io/badge/Fish--Audio-S2--Pro-success.svg)](https://huggingface.co/fishaudio/s2-pro)
[![VRAM](https://img.shields.io/badge/VRAM-≤10GB-orange.svg)](#arquitetura-híbrida-cross-memory)
[![Whisper](https://img.shields.io/badge/Whisper-24%20Cores%20CPU-blue.svg)](#transcrição-automática-com-whisper)

**VoxCloneBR** é uma solução de síntese de fala (TTS) e clonagem de voz instantânea (*zero-shot*) em **Português do Brasil**, baseada no modelo de ponta **Fish Audio S2-Pro**.

Desenvolvido especificamente para rodar com eficiência em máquinas com **12 GB de VRAM** (com teto de segurança em **10 GB**) e **64 GB de RAM**, utilizando arquitetura híbrida de memória (*Cross-Memory*) e processamento paralelo do Whisper em até **24 núcleos de CPU**.

---

## ✨ Principais Funcionalidades

- 🇧🇷 **Foco em Português do Brasil:** Otimizado com tags fonéticas e prosódia natural para o português brasileiro.
- 🎧 **Entrada Universal de Áudio (.ogg, .mp3, .wav, etc.):**
  - Aceita áudios diretamente de mensagens do **WhatsApp**, Telegram e gravações de voz sem erros de rejeição de formato.
  - Pré-processamento e normalização automática via FFmpeg.
- ⚡ **Transcrição Automática (ASR) com 24 Núcleos de CPU:**
  - Decifra automaticamente o que foi falado no áudio de amostra via **Faster-Whisper (int8)**.
  - Não consome nenhum byte de VRAM da GPU para transcrição.
  - Preenche automaticamente uma caixa de texto editável para correção manual de pontuação antes da clonagem.
- 🚀 **Arquitetura Híbrida Cross-Memory (GPU VRAM ≤ 10 GB):**
  - Mantém o uso da placa de vídeo entre 7.2 GB e 8.7 GB de pico, sem risco de Out-of-Memory (OOM).
  - Faz o balanceamento inteligente enviando camadas iniciais, embeddings e o codec de áudio DAC para a memória RAM do computador.
- 🎭 **Tags de Expressão e Emoção em 1 Clique:**
  - Botões para inserir pausas e emoções diretamente no texto: `[pause]`, `[whisper]`, `[laughing]`, `[chuckle]`, `[excited]`, `[angry]`, `[sad]`, `[sigh]`, `[emphasis]`, `[low voice]`.
- 🎵 **Saída em MP3 128 kbps:**
  - Conversão direta e automática para MP3 leve e de alta qualidade sonora.
- 📊 **Telemetria de VRAM em Tempo Real:**
  - Monitor visual na interface exibindo alocação e reserva de memória de vídeo.

---

## 🚀 Como Clonar e Rodar

### 1. Clonar o Repositório
```bash
git clone git@github.com:pierobonfada/voxclonebr.git
cd voxclonebr
```

### 2. Instalação Automática
O script `setup.sh` detecta se você possui `uv` ou `pip`, cria o ambiente virtual isolado, instala todas as dependências e baixa os pesos dos modelos (`fishaudio/s2-pro` e `faster-whisper`):

```bash
chmod +x setup.sh iniciar.sh
./setup.sh
```

*(Ou instale manualmente via `pip install -r requirements.txt` e execute `python download_models.py`)*

### 3. Iniciar a Aplicação
Para rodar a interface web:
```bash
./iniciar.sh
```
Acesse no seu navegador: **http://127.0.0.1:7860** (ou através da sua rede local via `http://<IP-DO-PC>:7860`).

Para rodar em uma porta personalizada:
```bash
./iniciar.sh 8080
```

---

## 📁 Estrutura do Projeto

```
voxclonebr/
├── .gitignore             # Ignora checkpoints grandes, áudios gerados e venvs
├── requirements.txt       # Dependências exatas e pinagens estáveis
├── setup.sh               # Script de instalação do ambiente e dependências
├── iniciar.sh             # Script de inicialização em 0.0.0.0:7860
├── download_models.py     # Download dos checkpoints s2-pro e whisper
├── engine_cross.py        # Motor de inferência híbrido GPU/CPU (Cross-Memory)
├── transcriber.py         # Motor ASR Faster-Whisper multi-thread (CPU)
├── webui_pt_br.py         # Interface Gradio em Português com telemetria
└── README.md              # Documentação completa
```

---

## ⚙️ Configurações de Geração

Na interface gráfica você pode ajustar:
- **Comprimento de Chunk (Iterativo):** Padrão `200`. Mantém coerência e ritmo em textos longos.
- **Temperatura:** Padrão `0.7`. Valores menores produzem fala mais neutra e estável; valores maiores trazem mais expressividade.
- **Top-P:** Padrão `0.8`. Amostragem de probabilidade dos tokens.
- **Penalidade de Repetição:** Padrão `1.2`. Evita repetições e gagueira.
- **Semente (Seed):** Fixe um número inteiro para reproduzir exatamente a mesma entonação, ou deixe `0` para gerar variações a cada execução.

---

## 🛠️ Requisitos de Sistema Recomendados

- **Sistema Operacional:** Linux (Ubuntu/Debian recomendado)
- **Placa de Vídeo:** NVIDIA RTX com 12 GB de VRAM (ex: RTX 3060, RTX 4070, etc.)
- **Memória RAM:** 32 GB a 64 GB
- **Processador:** CPU multi-core (utiliza até 24 threads no Faster-Whisper)
- **Drivers:** CUDA 12.4+ / PyTorch 2.4+
- **Dependência do Sistema:** `ffmpeg` instalado no sistema (`sudo apt install ffmpeg`)

---

## 📄 Licença

Este projeto é desenvolvido para fins de pesquisa e uso pessoal com modelos da comunidade Fish Audio.
Consulte as diretrizes e licenças originais do [Fish Audio](https://github.com/fishaudio/fish-speech).
