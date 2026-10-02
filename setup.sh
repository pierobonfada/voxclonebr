#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

export PATH="$HOME/.local/bin:$PATH"

echo "=========================================================="
echo "🛠️  Configurando ambiente do VoxCloneBR..."
echo "=========================================================="

# 0. Instala dependências essenciais do sistema se estiver em Debian/Ubuntu
install_system_deps() {
    local missing_pkgs=()
    if ! command -v ffmpeg &> /dev/null; then
        missing_pkgs+=("ffmpeg")
    fi
    if ! command -v gcc &> /dev/null; then
        missing_pkgs+=("build-essential")
    fi
    if [ ! -f "/usr/include/portaudio.h" ] && [ ! -f "/usr/local/include/portaudio.h" ]; then
        missing_pkgs+=("portaudio19-dev")
    fi
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        missing_pkgs+=("curl")
    fi

    if [ ${#missing_pkgs[@]} -gt 0 ]; then
        echo "📦 Instalando pacotes do sistema necessários: ${missing_pkgs[*]}..."
        if command -v apt-get &> /dev/null; then
            if [ "$(id -u)" -eq 0 ]; then
                apt-get update -qq && apt-get install -y -qq "${missing_pkgs[@]}"
            elif command -v sudo &> /dev/null; then
                sudo apt-get update -qq && sudo apt-get install -y -qq "${missing_pkgs[@]}"
            else
                echo "⚠️ Permissão root/sudo não encontrada. Por favor instale manualmente: apt-get install -y ${missing_pkgs[*]}"
            fi
        else
            echo "⚠️ Gerenciador de pacotes apt não encontrado. Certifique-se de que os seguintes pacotes estejam instalados: ${missing_pkgs[*]}"
        fi
    else
        echo "✅ Dependências do sistema (ffmpeg, build-essential, portaudio19-dev) verificadas!"
    fi
}

install_system_deps

# 1. Garante que o uv está instalado
if ! command -v uv &> /dev/null; then
    echo "⚡ Instalando 'uv' (gerenciador de ambiente e dependências ultrarrápido)..."
    if command -v curl &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &> /dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Cria virtualenv e instala dependências
if command -v uv &> /dev/null; then
    echo "⚡ Usando 'uv' para ambiente virtual e instalação ultrarrápida..."
    if [ ! -d ".venv" ]; then
        uv venv .venv --python 3.12 || uv venv .venv
    fi
    source .venv/bin/activate
    echo "📦 Instalando dependências..."
    uv pip install -r requirements.txt
else
    echo "🐍 Usando venv e pip padrão do Python..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    python -m pip install --upgrade pip
    echo "📦 Instalando dependências..."
    pip install -r requirements.txt
fi

# 3. Garante arquivo de marcação .project-root (necessário pelo pyrootutils do fish-speech)
touch .project-root
python -c "import fish_speech, pathlib; (pathlib.Path(fish_speech.__file__).parent / '.project-root').touch(exist_ok=True)" 2>/dev/null || true

# 4. Baixa modelos se necessário
if [ ! -d "checkpoints/s2-pro" ] || [ ! -f "checkpoints/s2-pro/codec.pth" ]; then
    echo "📥 Baixando modelos do Fish Audio S2-Pro e Whisper..."
    python download_models.py
else
    echo "✅ Modelos já encontrados em checkpoints/s2-pro!"
fi

chmod +x setup.sh iniciar.sh

echo "=========================================================="
echo "🎉 Instalação concluída com sucesso!"
echo "Para rodar o VoxCloneBR:"
echo "   ./iniciar.sh [porta]"
echo "=========================================================="
