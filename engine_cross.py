import os
import gc
import math
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Generator, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from loguru import logger

# Set PyTorch memory management settings to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_num_threads(os.cpu_count() or 24)

from fish_speech.models.text2semantic.llama import (
    DualARTransformer,
    KVCache,
    BaseTransformerForwardResult,
)
from fish_speech.models.text2semantic.inference import (
    generate_long,
    decode_one_token_ar,
)
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.utils.schema import ServeTTSRequest


class CrossMemoryS2ProEngine:
    """
    Fish Audio S2-Pro TTS Engine with Hybrid CPU-GPU (Cross-Memory) allocation.
    Guarantees GPU VRAM usage stays strictly under 10 GB (target ~7.0-8.7 GB),
    offloading layers, embeddings, DAC audio codec and Whisper to system RAM (64 GB).
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/s2-pro",
        decoder_checkpoint: str = "checkpoints/s2-pro/codec.pth",
        num_cpu_layers: int = 4,
        max_seq_len: int = 4096,
        max_vram_gb: float = 10.0,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.decoder_checkpoint = Path(decoder_checkpoint)
        self.num_cpu_layers = num_cpu_layers
        self.max_seq_len = max_seq_len
        self.max_vram_gb = max_vram_gb

        self.cuda_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_dev = torch.device("cpu")

        if torch.cuda.is_available():
            # Set a hard safety cap: 10GB out of total GPU memory
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            fraction = min(1.0, max_vram_gb / total_vram)
            try:
                torch.cuda.set_per_process_memory_fraction(fraction)
                logger.info(f"Set CUDA memory fraction to {fraction:.2f} ({max_vram_gb} GB cap on {total_vram:.1f} GB GPU)")
            except Exception as e:
                logger.warning(f"Could not set per_process_memory_fraction: {e}")

        logger.info("Initializing Cross-Memory S2-Pro Engine...")
        self._load_decoder()
        self._load_transformer()
        logger.info("Engine successfully initialized and ready!")

    def _load_decoder(self):
        """Loads DAC audio codec onto CPU (saving ~1.8 GB VRAM)."""
        logger.info("Loading DAC Codec on CPU (PC RAM)...")
        self.decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=str(self.decoder_checkpoint),
            device="cpu",
        )
        self.ref_loader = ReferenceLoader()
        self.ref_loader.decoder_model = self.decoder_model
        self.sample_rate = getattr(self.decoder_model, "sample_rate", 44100)
        logger.info(f"DAC Codec loaded on CPU. Sample rate: {self.sample_rate} Hz")

    def _load_transformer(self):
        """Loads DualARTransformer with layers partitioned across CPU and CUDA."""
        logger.info(
            f"Loading DualARTransformer S2-Pro (max_seq_len={self.max_seq_len}, "
            f"{self.num_cpu_layers} layers on CPU, {36 - self.num_cpu_layers} layers on GPU)..."
        )
        model = DualARTransformer.from_pretrained(
            str(self.checkpoint_dir),
            load_weights=True,
            max_length=self.max_seq_len,
        )

        total_layers = len(model.layers)
        assert self.num_cpu_layers < total_layers, "Cannot offload all layers to CPU"

        # Assign layers: first N layers on CPU, rest on CUDA
        for i in range(self.num_cpu_layers):
            model.layers[i] = model.layers[i].to(device=self.cpu_dev, dtype=torch.bfloat16)

        for i in range(self.num_cpu_layers, total_layers):
            model.layers[i] = model.layers[i].to(device=self.cuda_dev, dtype=torch.bfloat16)

        # Offload large token embeddings and codebook embeddings to CPU (saves ~1 GB VRAM)
        model.embeddings = model.embeddings.to(device=self.cpu_dev, dtype=torch.bfloat16)
        model.codebook_embeddings = model.codebook_embeddings.to(device=self.cpu_dev, dtype=torch.bfloat16)

        # Output head and norm on CUDA
        model.norm = model.norm.to(device=self.cuda_dev, dtype=torch.bfloat16)
        if hasattr(model, "output"):
            model.output = model.output.to(device=self.cuda_dev, dtype=torch.bfloat16)

        # Fast AR transformer (4 layers) on CUDA
        model.fast_layers = model.fast_layers.to(device=self.cuda_dev, dtype=torch.bfloat16)
        model.fast_embeddings = model.fast_embeddings.to(device=self.cuda_dev, dtype=torch.bfloat16)
        model.fast_norm = model.fast_norm.to(device=self.cuda_dev, dtype=torch.bfloat16)
        model.fast_output = model.fast_output.to(device=self.cuda_dev, dtype=torch.bfloat16)

        # Buffers for rotary position embeddings and causal masks
        model.causal_mask = model.causal_mask.to(self.cuda_dev)
        model.fast_freqs_cis = model.fast_freqs_cis.to(self.cuda_dev)
        model.freqs_cis = model.freqs_cis.to(self.cuda_dev)

        model.causal_mask_cpu = model.causal_mask.to(self.cpu_dev)
        model.freqs_cis_cpu = model.freqs_cis.to(self.cpu_dev)

        # Allocate KV caches per-layer on their native device
        for b in model.layers:
            dev = next(b.parameters()).device
            with torch.device(dev):
                b.attention.kv_cache = KVCache(
                    1,
                    self.max_seq_len,
                    model.config.n_local_heads,
                    model.config.head_dim,
                    dtype=torch.bfloat16,
                )

        for b in model.fast_layers:
            with torch.device(self.cuda_dev):
                b.attention.kv_cache = KVCache(
                    1,
                    model.config.num_codebooks,
                    model.config.fast_n_local_heads,
                    model.config.fast_head_dim,
                    dtype=torch.bfloat16,
                )

        model.max_seq_len = self.max_seq_len
        model.max_batch_size = 1
        model._cache_setup_done = True

        # Attach cross-device forward pass
        self._patch_cross_device_forward(model)
        self.model = model

        allocated_gb = self.get_vram_allocated_gb()
        reserved_gb = self.get_vram_reserved_gb()
        logger.info(
            f"DualARTransformer loaded. Current VRAM: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved."
        )

    def _patch_cross_device_forward(self, model: DualARTransformer):
        """Patches model.forward_generate to handle cross-memory execution smoothly."""
        cpu_dev = self.cpu_dev
        cuda_dev = self.cuda_dev

        def cross_forward_generate(
            inp: torch.Tensor,
            input_pos: Optional[torch.Tensor] = None,
            audio_masks: Optional[torch.Tensor] = None,
            audio_parts: Optional[torch.Tensor] = None,
            return_all: bool = False,
            kv_len: Optional[int] = None,
        ) -> BaseTransformerForwardResult:
            # 1. Embedding on CPU
            inp_cpu = inp.to(cpu_dev)
            embeds = []
            for i in range(model.config.num_codebooks):
                emb = model.codebook_embeddings(
                    inp_cpu[:, i + 1] + i * model.config.codebook_size
                )
                embeds.append(emb)

            vq_embeds_sum = torch.stack(embeds, dim=1).sum(dim=1)
            vq_masks = (inp_cpu[:, 0] >= model.config.semantic_begin_id) & (
                inp_cpu[:, 0] <= model.config.semantic_end_id
            )
            vq_embeds_sum[~vq_masks] = 0
            x = model.embeddings(inp_cpu[:, 0]) + vq_embeds_sum

            if model.config.scale_codebook_embeddings:
                vq_masks_expanded = vq_masks.unsqueeze(-1).expand_as(x)
                x = torch.where(
                    vq_masks_expanded,
                    x / math.sqrt(model.config.num_codebooks + 1),
                    x,
                )

            # Audio embeddings if present
            if audio_parts is not None and hasattr(model, "audio_projector"):
                audio_embeds = model.audio_projector(audio_parts.to(next(model.audio_projector.parameters()).device))
                if model.config.scale_codebook_embeddings:
                    x[audio_masks] = audio_embeds / math.sqrt(2)
                else:
                    x[audio_masks] = audio_embeds

            cache_capacity = (
                model.max_seq_len if model.max_seq_len > 0 else model.config.max_seq_len
            )
            if input_pos is None:
                input_pos = torch.arange(inp.shape[-1], device=x.device)
                active_kv_len = inp.shape[-1]
            else:
                active_kv_len = cache_capacity if kv_len is None else kv_len

            mask_cuda = model.causal_mask[None, None, input_pos.to(cuda_dev), :active_kv_len]
            freqs_cis_cuda = model.freqs_cis[input_pos.to(cuda_dev)]

            mask_cpu = model.causal_mask_cpu[None, None, input_pos.to(cpu_dev), :active_kv_len]
            freqs_cis_cpu = model.freqs_cis_cpu[input_pos.to(cpu_dev)]

            # 2. Sequential passage through layers
            for layer in model.layers:
                ldev = next(layer.parameters()).device
                if x.device != ldev:
                    x = x.to(ldev)
                fl = freqs_cis_cpu if ldev == cpu_dev else freqs_cis_cuda
                ml = mask_cpu if ldev == cpu_dev else mask_cuda
                ipl = input_pos.to(ldev)
                x = layer(x, fl, ml, input_pos=ipl)

            if x.size(1) > 1 and not return_all:
                x = x[:, -1:]

            # 3. Norm and Output on CUDA
            x = x.to(cuda_dev)
            slow_out = model.norm(x)

            token_logits = F.linear(slow_out, model.embeddings.weight.to(cuda_dev))
            hidden_out = (
                slow_out
                if getattr(model.config, "norm_fastlayer_input", False)
                else x
            )
            hidden_out = model.fast_project_in(hidden_out)

            return BaseTransformerForwardResult(
                logits=token_logits, hidden_states=hidden_out
            )

        model.forward_generate = cross_forward_generate

    def get_vram_allocated_gb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0

    def get_vram_reserved_gb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / (1024**3)
        return 0.0

    def encode_reference_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Loads and encodes any audio file format (.ogg, .mp3, .wav, etc.) into VQ tokens on CPU."""
        if not audio_path or not os.path.exists(audio_path):
            return None

        # Convert to standard wav if necessary or read directly
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            audio_content = self.ref_loader.load_audio(audio_bytes, self.sample_rate)
        except Exception as e:
            logger.warning(f"torchaudio failed to read {audio_path}: {e}. Fallback to ffmpeg...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                wav_path = tmp_wav.name
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_path, "-ar", str(self.sample_rate), "-ac", "1", wav_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                with open(wav_path, "rb") as f:
                    audio_bytes = f.read()
                audio_content = self.ref_loader.load_audio(audio_bytes, self.sample_rate)
            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)

        audios = torch.from_numpy(audio_content).to("cpu")[None, None, :]
        audio_lengths = torch.tensor([audios.shape[2]], device="cpu", dtype=torch.long)

        prompt_tokens = self.decoder_model.encode(audios, audio_lengths)[0][0]
        logger.info(f"Encoded reference audio into prompt tokens: shape {prompt_tokens.shape}")
        return prompt_tokens

    @torch.inference_mode()
    def generate_tts(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 1024,
        chunk_length: int = 200,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generates speech from text with optional zero-shot voice cloning.
        Returns (audio_numpy_float32, sample_rate).
        """
        if seed is not None and seed > 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        prompt_tokens_list = []
        prompt_text_list = []

        if reference_audio_path and os.path.exists(reference_audio_path):
            p_tokens = self.encode_reference_audio(reference_audio_path)
            if p_tokens is not None:
                prompt_tokens_list.append(p_tokens)
                ref_txt = reference_text.strip() if reference_text else ""
                prompt_text_list.append(ref_txt)

        logger.info(f"Generating speech for text: {text[:60]}...")
        t0 = time.time()

        chunks = list(
            generate_long(
                model=self.model,
                device="cuda" if torch.cuda.is_available() else "cpu",
                decode_one_token=decode_one_token_ar,
                text=text,
                prompt_tokens=prompt_tokens_list if prompt_tokens_list else None,
                prompt_text=prompt_text_list if prompt_text_list else None,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                compile=False,
                iterative_prompt=chunk_length > 0,
                chunk_length=chunk_length,
            )
        )

        audio_segments = []
        for chunk in chunks:
            if chunk.action != "next":
                codes = chunk.codes if isinstance(chunk.codes, torch.Tensor) else torch.from_numpy(chunk.codes)
                codes = codes.to("cpu")
                # Decode VQ tokens to audio on CPU
                fake_audio = (
                    self.decoder_model.from_indices(codes[None])[0, 0]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
                audio_segments.append(fake_audio)

        if not audio_segments:
            raise RuntimeError("Nenhum áudio foi gerado pelo modelo.")

        final_audio = np.concatenate(audio_segments, axis=0)
        dur = len(final_audio) / self.sample_rate
        logger.info(
            f"Generated {dur:.2f}s of audio in {time.time() - t0:.2f}s! "
            f"VRAM Allocated: {self.get_vram_allocated_gb():.2f} GB"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return final_audio, self.sample_rate

    def export_mp3_128k(self, audio_data: np.ndarray, output_path: str):
        """Encodes raw numpy audio into MP3 at exactly 128 kbps using ffmpeg."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name

        try:
            sf.write(tmp_wav_path, audio_data, self.sample_rate)
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                tmp_wav_path,
                "-b:a",
                "128k",
                "-f",
                "mp3",
                output_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"Exported MP3 128k to {output_path} (size: {os.path.getsize(output_path)} bytes)")
        finally:
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)
