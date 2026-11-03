"""MeanVC 2: zero-shot streaming conversion, 16 kHz output.

Any reference clip works as the target with no training, but the content
encoder comes from a Mandarin ASR model, so quality in other languages is
unproven, and its pitch can't be steered directly.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi

from ..paths import MEANVC2_DIR
from ..voices import Voice
from .base import Engine

RATE = 16000
# (ms of audio per process() call, algorithmic latency in ms). The "40ms" model
# uses 80 ms ASR chunks; the upstream realtime loop always feeds 160 ms, and
# intelligibility came out the same at both. The latencies are estimates.
MODELS = {"40ms": (80, 130), "120ms": (160, 210)}


def _load_upstream():
    runtime = MEANVC2_DIR / "runtime"
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    spec = importlib.util.spec_from_file_location("meanvc2_run_rt", runtime / "run_rt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_runner(voice: Voice, device: str, model: str, chunk: int):
    base = _load_upstream().VCRunner

    class Runner(base):
        def _encode_chunk(self, samples: np.ndarray) -> torch.Tensor | None:
            """Upstream's method with the BN upsampling sized to CHUNK, not fixed at 160 ms."""
            with torch.no_grad():
                padded = (np.concatenate((self.samples_cache, samples))
                          if self.samples_cache.size else samples)
                if len(padded) < 400:
                    self.samples_cache = padded
                    return None
                fbanks = kaldi.fbank(torch.from_numpy(padded * (1 << 15)).unsqueeze(0),
                                     frame_length=25, frame_shift=10, snip_edges=True,
                                     num_mel_bins=80, energy_floor=0.0, dither=0.0,
                                     sample_frequency=RATE)
                self.samples_cache = padded[160 * fbanks.shape[0]:]
                if self.frame_cache is not None:
                    fbanks = torch.cat([self.frame_cache, fbanks], dim=0)

                if self.asr_offset >= 4000:
                    cache_t = self.asr_att_cache.size(2) if self.asr_att_cache.dim() > 0 else 0
                    self.asr_offset = max(cache_t, self.asr_offset - 4000)

                bns = []
                i = 0
                while i + self._bn_window <= fbanks.shape[0]:
                    out, self.asr_att_cache, self.asr_cnn_cache = self.asr(
                        fbanks[i:i + self._bn_window].unsqueeze(0),
                        torch.tensor(self.asr_offset, dtype=torch.int64),
                        torch.tensor(self._required_cache_size, dtype=torch.int64),
                        self.asr_att_cache, self.asr_cnn_cache,
                    )
                    bns.append(out.squeeze(0).detach())
                    self.asr_offset += self._asr_offset_step
                    i += self._bn_stride
                self.frame_cache = fbanks[i:]
                if not bns:
                    return None

                bn = torch.cat(bns, dim=0).unsqueeze(0)
                if self.encoder_output_cache is not None:
                    bn = torch.cat([self.encoder_output_cache, bn], dim=1)
                self.encoder_output_cache = bn[:, -1:, :]
                if bn.shape[1] < 2:
                    return None
                frames = self.CHUNK // 160
                up = F.interpolate(bn.transpose(1, 2), size=frames + 1, mode="linear",
                                   align_corners=True)
                return up.transpose(1, 2)[:, 1:, :]

    runner = Runner(str(voice.reference), device=device, model=model)
    runner.CHUNK = chunk  # upstream's __init__ sets its own
    return runner


class MeanVC2Engine(Engine):
    name = "meanvc2"
    input_rate = output_rate = RATE

    def __init__(self, voice: Voice, model: str = "40ms", device: str = "cuda", **_):
        chunk_ms, self._latency = MODELS[model]
        self.block_samples = chunk_ms * RATE // 1000
        self.runner = _make_runner(voice, device, model, self.block_samples)
        self.process(np.zeros(self.block_samples, dtype=np.float32))
        self.reset()

    @property
    def algorithmic_latency_ms(self) -> float:
        return self._latency

    def reset(self) -> None:
        self.runner._init_cache()

    def process(self, block: np.ndarray) -> np.ndarray:
        out = self.runner.process_chunk(block)
        return np.zeros(0, dtype=np.float32) if out is None else out.astype(np.float32)
