import os
import tempfile
import subprocess
from pathlib import Path
from loguru import logger
from faster_whisper import WhisperModel

class AudioTranscriber:
    """
    Motor de Reconhecimento Automático de Fala (ASR) para Português do Brasil usando faster-whisper na CPU.
    Aceita QUALQUER formato de áudio (.ogg do WhatsApp, mp3, wav, m4a, flac, opus, etc.)
    sem consumir VRAM da GPU e utilizando todos os núcleos de CPU disponíveis.
    """

    def __init__(self, model_size: str = "small", cpu_threads: int = None):
        self.model_size = model_size
        threads = cpu_threads or os.cpu_count() or 24
        logger.info(f"Carregando faster-whisper '{model_size}' na CPU (int8) usando {threads} threads...")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
            num_workers=4,
        )
        logger.info(f"Transcriber pronto com {threads} threads de CPU.")

    def transcribe(self, audio_path: str, language: str = "pt") -> str:
        if not audio_path or not os.path.exists(audio_path):
            return ""

        input_path = audio_path
        tmp_converted_path = None

        # Pré-converte para WAV 16kHz via FFmpeg para garantir compatibilidade 100% universal (.ogg, etc.)
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                tmp_converted_path = tmp_f.name

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                tmp_converted_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            input_path = tmp_converted_path
        except Exception as e:
            logger.warning(f"Aviso na conversão FFmpeg fallback: {e}. Tentando leitura direta.")
            input_path = audio_path

        try:
            segments, info = self.model.transcribe(
                input_path,
                language=language,
                beam_size=5,
                vad_filter=True,
            )
            text_parts = [s.text.strip() for s in segments]
            transcription = " ".join(text_parts).strip()
            logger.info(f"Transcrição concluída ({len(transcription)} caracteres): {transcription}")
            return transcription
        finally:
            if tmp_converted_path and os.path.exists(tmp_converted_path):
                os.remove(tmp_converted_path)
