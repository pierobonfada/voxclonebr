#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

export PATH="$HOME/.local/bin:$PATH"
[ -f ".project-root" ] || touch .project-root

# Detecta interpretador Python
if [ -d ".venv" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -d "venv" ]; then
    VENV_PYTHON="venv/bin/python"
elif [ -d "../fish-tts/.venv" ]; then
    VENV_PYTHON="../fish-tts/.venv/bin/python"
else
    VENV_PYTHON="python3"
fi

# Verifica se os modelos existem
if [ ! -d "checkpoints/s2-pro" ] || [ ! -f "checkpoints/s2-pro/codec.pth" ]; then
    if [ -d "../fish-tts/checkpoints/s2-pro" ]; then
        echo "🔗 Vinculando checkpoints já existentes..."
        mkdir -p checkpoints
        ln -sfn "$(pwd)/../fish-tts/checkpoints/s2-pro" checkpoints/s2-pro
    else
        echo "⚠️ Modelos não encontrados em checkpoints/s2-pro!"
        echo "📥 Baixando modelos do Fish Audio S2-Pro..."
        $VENV_PYTHON download_models.py
    fi
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-24}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-24}
PORT="${1:-7860}"

echo "=========================================================="
echo "🐟 VoxCloneBR — Fish Audio S2-Pro (Português do Brasil)"
echo "🚀 Modo Híbrido Cross-Memory (GPU VRAM ≤ 10GB + PC RAM 64GB)"
echo "⚡ CPU Whisper ASR: 24 núcleos de alto desempenho"
echo "🎧 Saída de áudio: MP3 128 kbps"
echo "🌐 Acessível em: http://0.0.0.0:${PORT}"
echo "=========================================================="

exec $VENV_PYTHON webui_pt_br.py --host 0.0.0.0 --port "$PORT" --cpu-layers 4
