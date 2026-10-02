#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=========================================================="
echo "🛠️ Configurando ambiente do VoxCloneBR..."
echo "=========================================================="

# 1. Cria virtualenv e instala dependências
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

# 2. Baixa modelos se necessário
if [ ! -d "checkpoints/s2-pro" ] || [ ! -f "checkpoints/s2-pro/codec.pth" ]; then
    echo "📥 Baixando modelos do Fish Audio S2-Pro e Whisper..."
    python download_models.py
else
    echo "✅ Modelos já encontrados em checkpoints/s2-pro!"
fi

echo "=========================================================="
echo "🎉 Instalação concluída com sucesso!"
echo "Para rodar o VoxCloneBR:"
echo "   ./iniciar.sh [porta]"
echo "=========================================================="
