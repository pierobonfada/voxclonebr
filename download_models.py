#!/usr/bin/env python3
"""
download_models.py
Script para baixar os pesos do Fish Audio S2-Pro e pré-carregar o Whisper.
"""

import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def download_s2_pro():
    target_dir = Path("checkpoints/s2-pro")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Baixando / verificando pesos de 'fishaudio/s2-pro' em {target_dir}...")
    snapshot_download(
        repo_id="fishaudio/s2-pro",
        local_dir=str(target_dir),
        resume_download=True,
    )
    print("✅ Pesos do Fish Audio S2-Pro verificados com sucesso!")

def preload_whisper(model_size: str = "small"):
    print(f"📥 Pré-carregando modelo faster-whisper '{model_size}'...")
    try:
        from faster_whisper import WhisperModel
        WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"✅ Modelo Faster-Whisper '{model_size}' pronto para uso offline!")
    except Exception as e:
        print(f"⚠️ Aviso ao pré-carregar Faster-Whisper: {e}")

if __name__ == "__main__":
    download_s2_pro()
    preload_whisper("small")
    print("\n🎉 Todos os modelos necessários estão baixados e prontos!")
