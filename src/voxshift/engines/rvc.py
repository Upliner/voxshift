"""RVC v2: per-voice trained model, 48 kHz output, language independent.

Content comes from HuBERT phonetic features and pitch from RMVPE, neither tied
to a language. The streaming logic follows upstream's realtime_gui.py.
"""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from ..paths import RVC_DIR
from ..voices import Voice
from .base import Engine

ZC_DIVISOR = 100  # upstream sizes every buffer in 10 ms units
INDEX_NPROBE = 8  # at upstream's 1, sparse clusters return too few neighbours and whole blocks skip the index


def _load_upstream():
    # Upstream reads its locale files relative to the working directory at import.
    if str(RVC_DIR) not in sys.path:
        sys.path.insert(0, str(RVC_DIR))
    with contextlib.chdir(RVC_DIR):
        from infer.rmvpe import RMVPE
        from infer.rtrvc import RVC
    return RVC, RMVPE


class RvcEngine(Engine):
    name = "rvc"

    def __init__(self, voice: Voice, pitch: float = 12.0, block_ms: float = 40.0,
                 crossfade_ms: float = 40.0, context_ms: float = 1000.0, index_rate: float = 0.3,
                 f0_method: str = "rmvpe", device: str = "cuda", half: bool = True, **_):
        torch.backends.cudnn.enabled = True  # MeanVC2 turns this off process-wide
        RVC, RMVPE = _load_upstream()
        self.device = torch.device(device)
        self.f0_method = f0_method
        config = SimpleNamespace(device=self.device, is_half=half)
        with contextlib.chdir(RVC_DIR):
            self.rvc = RVC(pitch, 0, str(voice.model), str(voice.index), index_rate, config)
        if self.rvc.net_g is None:
            raise RuntimeError(f"Could not load RVC model {voice.model}: see the traceback above")
        self.rvc.index.nprobe = INDEX_NPROBE
        sys.modules["infer.rtrvc"].printt = lambda *_: None  # timing logs would break the live status line
        self.rvc.model_rmvpe = RMVPE(str(RVC_DIR / "assets/rmvpe/rmvpe.pt"), is_half=half,
                                     device=self.device)

        rate = self.rvc.tgt_sr
        self.input_rate = self.output_rate = rate
        zc = self.zc = rate // ZC_DIVISOR

        def frames(ms: float) -> int:
            return int(round(ms * rate / 1000 / zc)) * zc

        self.block_samples = frames(block_ms)
        self.block_16k = 160 * self.block_samples // zc
        crossfade = frames(crossfade_ms)
        self.sola_buffer_frames = min(crossfade, 4 * zc)
        self.sola_search_frames = zc
        extra = frames(context_ms)

        size = extra + crossfade + self.sola_search_frames + self.block_samples
        self.input_wav = torch.zeros(size, device=self.device, dtype=torch.float32)
        self.input_16k = torch.zeros(160 * size // zc, device=self.device, dtype=torch.float32)
        self.sola_buffer = torch.zeros(self.sola_buffer_frames, device=self.device, dtype=torch.float32)
        self.sola_den_kernel = torch.ones(1, 1, self.sola_buffer_frames, device=self.device)
        self.skip_head = extra // zc
        self.return_length = (self.block_samples + self.sola_buffer_frames + self.sola_search_frames) // zc
        self.fade_in = torch.sin(0.5 * np.pi * torch.linspace(
            0.0, 1.0, self.sola_buffer_frames, device=self.device, dtype=torch.float32)) ** 2
        self.fade_out = 1 - self.fade_in
        self.resampler = Resample(rate, 16000, dtype=torch.float32).to(self.device)

        self.process(np.zeros(self.block_samples, dtype=np.float32))  # first call builds caches
        self.reset()

    @property
    def algorithmic_latency_ms(self) -> float:
        held = self.block_samples + self.sola_buffer_frames + self.sola_search_frames
        return held * 1000 / self.input_rate

    def set_pitch(self, semitones: float) -> None:
        self.rvc.change_key(semitones)

    def reset(self) -> None:
        self.input_wav.zero_()
        self.input_16k.zero_()
        self.sola_buffer.zero_()
        self.rvc.cache_pitch.zero_()
        self.rvc.cache_pitchf.zero_()

    @torch.no_grad()
    def process(self, block: np.ndarray) -> np.ndarray:
        n, n16, zc = self.block_samples, self.block_16k, self.zc
        self.input_wav[:-n] = self.input_wav[n:].clone()
        self.input_wav[-n:] = torch.from_numpy(block).to(self.device)
        self.input_16k[:-n16] = self.input_16k[n16:].clone()
        self.input_16k[-160 * (n // zc + 1):] = self.resampler(self.input_wav[-n - 2 * zc:])[160:]

        wav = self.rvc.infer(self.input_16k, n16, self.skip_head, self.return_length, self.f0_method)

        # SOLA: align the new block to the tail of the previous one, then crossfade.
        head = wav[None, None, : self.sola_buffer_frames + self.sola_search_frames]
        nom = F.conv1d(head, self.sola_buffer[None, None, :])
        den = torch.sqrt(F.conv1d(head**2, self.sola_den_kernel) + 1e-8)
        offset = int(torch.argmax(nom[0, 0] / den[0, 0]))
        wav = wav[offset:]
        wav[: self.sola_buffer_frames] *= self.fade_in
        wav[: self.sola_buffer_frames] += self.sola_buffer * self.fade_out
        self.sola_buffer[:] = wav[n: n + self.sola_buffer_frames]
        return wav[:n].float().cpu().numpy()
