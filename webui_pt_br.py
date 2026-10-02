import os
import time
import tempfile
import argparse
import subprocess
from pathlib import Path
from loguru import logger

import gradio as gr
import torch

from engine_cross import CrossMemoryS2ProEngine
from transcriber import AudioTranscriber

# Instâncias globais carregadas sob demanda
ENGINE: CrossMemoryS2ProEngine = None
TRANSCRIBER: AudioTranscriber = None


def get_engine(num_cpu_layers: int = 4) -> CrossMemoryS2ProEngine:
    global ENGINE
    if ENGINE is None:
        logger.info(f"Carregando CrossMemoryS2ProEngine com {num_cpu_layers} camadas na CPU (32 na GPU)...")
        ENGINE = CrossMemoryS2ProEngine(
            checkpoint_dir="checkpoints/s2-pro",
            decoder_checkpoint="checkpoints/s2-pro/codec.pth",
            num_cpu_layers=num_cpu_layers,
            max_seq_len=4096,
            max_vram_gb=10.0,
        )
    return ENGINE


def get_transcriber(model_size: str = "small") -> AudioTranscriber:
    global TRANSCRIBER
    if TRANSCRIBER is None or TRANSCRIBER.model_size != model_size:
        logger.info(f"Carregando AudioTranscriber ('{model_size}' na CPU)...")
        TRANSCRIBER = AudioTranscriber(model_size=model_size)
    return TRANSCRIBER


def format_vram_status() -> str:
    """Retorna status atual da GPU formatado em HTML com barra de progresso."""
    if not torch.cuda.is_available():
        return "⚠️ GPU não detectada. Rodando em CPU."

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    pct = (allocated / total) * 100
    color = "#22c55e" if allocated <= 8.5 else ("#f59e0b" if allocated <= 10.0 else "#ef4444")

    return f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 600; font-size: 14px;">🎮 Monitor de VRAM (Limite Seguro: 10.0 GB):</span>
            <span style="font-weight: 700; color: {color}; font-size: 15px;">{allocated:.2f} GB / {total:.1f} GB ({pct:.1f}%)</span>
        </div>
        <div style="background: #334155; border-radius: 4px; height: 8px; width: 100%; margin-top: 8px; overflow: hidden;">
            <div style="background: {color}; width: {min(100, pct):.1f}%; height: 100%;"></div>
        </div>
        <div style="font-size: 11px; color: #94a3b8; margin-top: 6px;">
            VRAM Reservada: {reserved:.2f} GB | PC RAM (64 GB): Offload de 4 camadas + Embeddings + Codec DAC ativos.
        </div>
    </div>
    """


def normalize_and_transcribe_audio(file_path: str, model_size: str):
    """
    Recebe qualquer arquivo de áudio (.ogg, .mp3, .wav, .m4a, etc.),
    converte via FFmpeg para WAV/MP3 limpo e executa a transcrição Whisper.
    """
    if not file_path or not os.path.exists(file_path):
        return None, "", "Nenhum áudio recebido."

    logger.info(f"Processando áudio recebido: {file_path}")
    t0 = time.time()

    output_dir = Path("outputs/normalized_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    norm_wav = str(output_dir / f"amostra_{timestamp}.wav")
    preview_mp3 = str(output_dir / f"preview_{timestamp}.mp3")

    try:
        # Converte para WAV padrão 44.1kHz mono (compatível com qualquer codec .ogg do WhatsApp/Telegram/Discord)
        subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, "-vn", "-ar", "44100", "-ac", "1", norm_wav],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Gera MP3 leve para preview no navegador
        subprocess.run(
            ["ffmpeg", "-y", "-i", norm_wav, "-b:a", "128k", preview_mp3],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        audio_for_model = norm_wav
        audio_for_preview = preview_mp3
    except Exception as e:
        logger.warning(f"Aviso na conversão FFmpeg: {e}. Usando arquivo original.")
        audio_for_model = file_path
        audio_for_preview = file_path

    # Transcreve com Whisper na CPU
    transcriber = get_transcriber(model_size)
    try:
        text = transcriber.transcribe(audio_for_model, language="pt")
        elapsed = time.time() - t0
        status_msg = f"✅ Áudio processado e transcrito com sucesso em {elapsed:.1f}s!"
    except Exception as e:
        logger.error(f"Erro na transcrição Whisper: {e}")
        text = ""
        status_msg = f"⚠️ Áudio carregado, mas erro ao transcrever: {e}"

    return audio_for_preview, audio_for_model, text, status_msg


def generate_speech(
    text: str,
    processed_audio_path: str,
    reference_text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    chunk_length: int,
    seed: int,
    progress=gr.Progress(),
):
    """Gera a fala clonada em Português do Brasil e exporta em MP3 128k."""
    if not text or not text.strip():
        raise gr.Error("Por favor, digite o texto a ser falado.")

    if not processed_audio_path or not os.path.exists(processed_audio_path):
        raise gr.Error("Por favor, envie um áudio de amostra antes de gerar.")

    progress(0.1, desc="Iniciando motor de síntese...")
    engine = get_engine()

    progress(0.3, desc="Processando voz de referência e gerando tokens...")
    t0 = time.time()

    try:
        audio_data, sample_rate = engine.generate_tts(
            text=text.strip(),
            reference_audio_path=processed_audio_path,
            reference_text=reference_text.strip() if reference_text else None,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=int(max_new_tokens),
            chunk_length=int(chunk_length),
            seed=int(seed) if seed > 0 else None,
        )
    except Exception as e:
        logger.error(f"Erro na geração de TTS: {e}")
        raise gr.Error(f"Erro na geração de áudio: {e}")

    progress(0.85, desc="Codificando áudio em MP3 128 kbps...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = int(time.time() * 1000)
    mp3_output_path = str(output_dir / f"fala_gerada_{timestamp}.mp3")

    engine.export_mp3_128k(audio_data, mp3_output_path)
    total_time = time.time() - t0
    audio_dur = len(audio_data) / sample_rate

    vram_status = format_vram_status()
    info_msg = (
        f"🎉 **Áudio gerado com sucesso!**\n\n"
        f"- **Duração do áudio:** {audio_dur:.2f} segundos\n"
        f"- **Tempo de síntese:** {total_time:.2f} segundos\n"
        f"- **Formato:** MP3 128 kbps\n"
        f"- **VRAM Alocada:** {engine.get_vram_allocated_gb():.2f} GB (Seguro: < 10 GB)"
    )

    progress(1.0, desc="Concluído!")
    return mp3_output_path, info_msg, vram_status


def insert_tag(current_text: str, tag: str) -> str:
    """Insere uma tag de emoção ou prosódia no texto."""
    current_text = current_text or ""
    if current_text.endswith(" ") or not current_text:
        return current_text + f"{tag} "
    return current_text + f" {tag} "


def build_app(initial_cpu_layers: int = 4) -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
    )

    custom_css = """
    .tag-btn {
        min-width: 90px !important;
        font-size: 13px !important;
        padding: 4px 8px !important;
        margin: 2px !important;
    }
    .header-box {
        text-align: center;
        padding: 16px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    """

    with gr.Blocks(theme=theme, css=custom_css, title="Fish Audio S2-Pro — Clonagem de Voz PT-BR") as app:
        processed_audio_state = gr.State(value="")

        gr.HTML(
            """
            <div class="header-box">
                <h1 style="margin: 0; font-size: 26px;">🐟 Fish Audio S2-Pro — Síntese e Clonagem de Voz</h1>
                <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 14px;">
                    Português do Brasil 🇧🇷 • Arquitetura Híbrida Cross-Memory (GPU VRAM ≤ 10 GB + PC RAM 64 GB) • Saída MP3 128 kbps
                </p>
            </div>
            """
        )

        vram_monitor = gr.HTML(value=format_vram_status())

        with gr.Row():
            # Coluna da esquerda: Áudio de Amostra e Transcrição
            with gr.Column(scale=5):
                gr.Markdown("### 1️⃣ Áudio de Amostra (Voz a ser Clonada)")
                gr.Markdown(
                    "Envie qualquer áudio (5 a 30 seg). **Aceita qualquer formato: `.ogg` (WhatsApp, Opus, Vorbis), `.mp3`, `.wav`, etc.**"
                )

                with gr.Tab("📁 Upload de Arquivo (.ogg, .mp3, .wav, etc.)"):
                    file_input = gr.File(
                        label="Clique ou arraste seu áudio aqui (qualquer formato é aceito!)",
                        type="filepath",
                    )

                with gr.Tab("🎤 Gravar com Microfone"):
                    mic_input = gr.Audio(
                        label="Grave diretamente pelo microfone",
                        sources=["microphone"],
                        type="filepath",
                    )

                audio_preview = gr.Audio(
                    label="🎧 Prévia do Áudio de Amostra",
                    interactive=False,
                )

                with gr.Row():
                    whisper_model_choice = gr.Dropdown(
                        label="Modelo Whisper (CPU)",
                        choices=["small", "large-v3"],
                        value="small",
                        scale=2,
                    )
                    btn_retranscribe = gr.Button("🔄 Re-transcrever Áudio", scale=2, variant="secondary")

                transcribe_status = gr.Markdown(value="*Envie um áudio para transcrever automaticamente com IA.*")

                ref_text = gr.Textbox(
                    label="📝 Texto Falado no Áudio de Amostra (Editável)",
                    placeholder="O texto falado no áudio acima aparecerá aqui automaticamente. Você pode editá-lo para corrigir pontuação ou pronúncia.",
                    lines=3,
                    interactive=True,
                )

                with gr.Accordion("⚙️ Configurações Avançadas da Síntese", open=False):
                    chunk_length = gr.Slider(
                        label="Comprimento de Chunk (Prompt Iterativo)",
                        minimum=0,
                        maximum=400,
                        value=200,
                        step=20,
                        info="0 desativa o particionamento. 200 mantém estabilidade de frases longas.",
                    )
                    max_new_tokens = gr.Slider(
                        label="Máximo de Novos Tokens por Batch",
                        minimum=64,
                        maximum=2048,
                        value=1024,
                        step=64,
                        info="Controla a extensão máxima da fala gerada.",
                    )
                    temperature = gr.Slider(
                        label="Temperatura (Criatividade / Variação)",
                        minimum=0.1,
                        maximum=1.2,
                        value=0.7,
                        step=0.05,
                        info="Valores menores = mais estável; valores maiores = mais expressivo.",
                    )
                    top_p = gr.Slider(
                        label="Top-P (Amostragem de Núcleo)",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                    )
                    repetition_penalty = gr.Slider(
                        label="Penalidade de Repetição",
                        minimum=1.0,
                        maximum=1.8,
                        value=1.2,
                        step=0.05,
                        info="Evita repetições e gagueira no áudio gerado.",
                    )
                    seed = gr.Number(
                        label="Semente (Seed)",
                        value=0,
                        precision=0,
                        info="0 para aleatório a cada geração, ou fixe um número para reprodutibilidade.",
                    )

            # Coluna da direita: Texto para fala e saída
            with gr.Column(scale=6):
                gr.Markdown("### 2️⃣ Texto a ser Falado (Português do Brasil)")
                input_text = gr.Textbox(
                    label="✍️ Texto a ser Falado",
                    placeholder="Digite o texto que a voz clonada deve falar em português do Brasil...",
                    lines=6,
                )

                gr.Markdown("#### ✨ Tags de Expressão & Prosódia (Clique para inserir no texto):")
                with gr.Row():
                    btn_pause = gr.Button("[pause]", elem_classes=["tag-btn"])
                    btn_whisper = gr.Button("[whisper]", elem_classes=["tag-btn"])
                    btn_laughing = gr.Button("[laughing]", elem_classes=["tag-btn"])
                    btn_chuckle = gr.Button("[chuckle]", elem_classes=["tag-btn"])
                    btn_excited = gr.Button("[excited]", elem_classes=["tag-btn"])
                with gr.Row():
                    btn_angry = gr.Button("[angry]", elem_classes=["tag-btn"])
                    btn_sad = gr.Button("[sad]", elem_classes=["tag-btn"])
                    btn_sigh = gr.Button("[sigh]", elem_classes=["tag-btn"])
                    btn_emphasis = gr.Button("[emphasis]", elem_classes=["tag-btn"])
                    btn_low_voice = gr.Button("[low voice]", elem_classes=["tag-btn"])

                btn_generate = gr.Button(
                    "🚀 Gerar Áudio (Voz Clonada - MP3 128k)",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("### 3️⃣ Áudio Gerado (MP3 128 kbps)")
                output_audio = gr.Audio(
                    label="🎧 Reproduzir / Baixar Áudio Final",
                    type="filepath",
                    interactive=False,
                )

                output_info = gr.Markdown(value="*Envie o áudio de amostra, digite o texto e clique no botão acima.*")

        # Conexão dos Eventos
        for btn, tag in [
            (btn_pause, "[pause]"),
            (btn_whisper, "[whisper]"),
            (btn_laughing, "[laughing]"),
            (btn_chuckle, "[chuckle]"),
            (btn_excited, "[excited]"),
            (btn_angry, "[angry]"),
            (btn_sad, "[sad]"),
            (btn_sigh, "[sigh]"),
            (btn_emphasis, "[emphasis]"),
            (btn_low_voice, "[low voice]"),
        ]:
            btn.click(insert_tag, inputs=[input_text, gr.State(tag)], outputs=[input_text])

        file_input.change(
            normalize_and_transcribe_audio,
            inputs=[file_input, whisper_model_choice],
            outputs=[audio_preview, processed_audio_state, ref_text, transcribe_status],
        )

        mic_input.change(
            normalize_and_transcribe_audio,
            inputs=[mic_input, whisper_model_choice],
            outputs=[audio_preview, processed_audio_state, ref_text, transcribe_status],
        )

        def retranscribe_action(curr_audio, model_sz):
            return normalize_and_transcribe_audio(curr_audio, model_sz)

        btn_retranscribe.click(
            retranscribe_action,
            inputs=[processed_audio_state, whisper_model_choice],
            outputs=[audio_preview, processed_audio_state, ref_text, transcribe_status],
        )

        btn_generate.click(
            generate_speech,
            inputs=[
                input_text,
                processed_audio_state,
                ref_text,
                temperature,
                top_p,
                repetition_penalty,
                max_new_tokens,
                chunk_length,
                seed,
            ],
            outputs=[output_audio, output_info, vram_monitor],
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fish Audio S2-Pro TTS PT-BR WebUI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=7860, help="Porta para rodar o Gradio")
    parser.add_argument("--cpu-layers", type=int, default=4, help="Número de camadas na CPU (Cross-Memory)")
    parser.add_argument("--share", action="store_true", help="Criar link público do Gradio")
    args = parser.parse_args()

    # Pré-carrega engine e transcriber
    get_engine(num_cpu_layers=args.cpu_layers)
    get_transcriber("small")

    app = build_app(initial_cpu_layers=args.cpu_layers)
    logger.info(f"Iniciando Gradio WebUI em http://{args.host}:{args.port}")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
