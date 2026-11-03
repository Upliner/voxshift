"""The realtime loop: capture, gate, convert, fan out to sinks."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from .audio import Capture
from .engines.base import Engine

MAX_BACKLOG_BLOCKS = 2


class Sink(Protocol):
    def write(self, samples: np.ndarray) -> None: ...
    def close(self) -> None: ...


class FileSink:
    """Writes to an audio file from its own thread, so disk stalls can't hit the audio loop."""

    def __init__(self, path: Path, rate: int):
        self.file = sf.SoundFile(path, "w", samplerate=rate, channels=1)
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while (samples := self.queue.get()) is not None:
            self.file.write(samples)
        self.file.close()

    def write(self, samples: np.ndarray) -> None:
        self.queue.put(samples)

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()


class NoiseGate:
    """Mutes the input between words: voice models turn room noise into breathy artefacts.

    Works in 10 ms frames. It opens at once, holds through short pauses, and
    ramps so the gain never steps.
    """

    def __init__(self, rate: int, threshold_db: float = -45.0, hold_ms: float = 250.0):
        self.frame = rate // 100
        self.threshold_db = threshold_db
        self.hold_frames = int(hold_ms / 10)
        self.remaining = 0
        self.gain = 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        out = block.copy()
        for start in range(0, len(block), self.frame):
            frame = out[start:start + self.frame]
            level = 10 * np.log10(np.mean(frame**2) + 1e-12)
            self.remaining = self.hold_frames if level > self.threshold_db else max(self.remaining - 1, 0)
            target = 1.0 if self.remaining else 0.0
            frame *= np.linspace(self.gain, target, len(frame), dtype=np.float32)
            self.gain = target
        return out


@dataclass
class Stats:
    blocks: int = 0
    overruns: int = 0
    last_ms: float = 0.0
    peak_ms: float = 0.0
    input_db: float = -120.0
    output_db: float = -120.0


def level_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(x**2) + 1e-12)) if len(x) else -120.0


class Pipeline:
    """Runs an engine on a live stream until stop().

    One thread only drains the capture pipe. If conversion falls behind, the
    oldest blocks are discarded: an audible skip is better than latency that
    grows for the rest of the session.
    """

    def __init__(self, engine: Engine, capture: Capture, sinks: list[Sink], gate: NoiseGate | None):
        self.engine = engine
        self.capture = capture
        self.sinks = sinks
        self.gate = gate
        self.stats = Stats()
        self.running = False
        self.eof = False
        self.error: Exception | None = None
        self.blocks: deque[np.ndarray] = deque()
        self.ready = threading.Condition()
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        self.running = True
        self.threads = [threading.Thread(target=self._read, daemon=True),
                        threading.Thread(target=self._convert, daemon=True)]
        for t in self.threads:
            t.start()

    def _read(self) -> None:
        while self.running:
            block = self.capture.read(self.engine.block_samples)
            with self.ready:
                if block is None:
                    self.eof = True
                    self.ready.notify()
                    return
                self.blocks.append(block)
                while len(self.blocks) > MAX_BACKLOG_BLOCKS:
                    self.blocks.popleft()
                    self.stats.overruns += 1
                self.ready.notify()

    def _convert(self) -> None:
        try:
            while True:
                with self.ready:
                    while self.running and not self.blocks and not self.eof:
                        self.ready.wait(0.5)
                    if not self.running or not self.blocks:
                        return
                    block = self.blocks.popleft()
                self._step(block)
        except Exception as e:
            self.error = e
        finally:
            self.running = False

    def _step(self, block: np.ndarray) -> None:
        stats = self.stats
        start = time.perf_counter()
        stats.input_db = level_db(block)
        gated = self.gate.process(block) if self.gate else block
        out = self.engine.process(gated)
        stats.last_ms = (time.perf_counter() - start) * 1000
        stats.peak_ms = max(stats.peak_ms, stats.last_ms)
        stats.blocks += 1
        if len(out):
            stats.output_db = level_db(out)
            for sink in self.sinks:
                sink.write(out)

    def stop(self) -> None:
        self.running = False
        with self.ready:
            self.ready.notify_all()
        self.capture.close()
        for t in self.threads:
            t.join(timeout=3)
        for sink in self.sinks:
            sink.close()


def convert_stream(engine: Engine, audio: np.ndarray, gate: NoiseGate | None = None) -> np.ndarray:
    """Run a whole clip through the streaming path, removing the engine's delay.

    Files take the same code path as live audio, so what you hear from `convert`
    is what the live stream produces.
    """
    engine.reset()
    n = engine.block_samples
    delay = int(engine.algorithmic_latency_ms * engine.output_rate / 1000)
    padded = np.concatenate([audio, np.zeros(delay * engine.input_rate // engine.output_rate + n, np.float32)])
    parts = []
    for start in range(0, len(padded) - n + 1, n):
        block = padded[start:start + n]
        parts.append(engine.process(gate.process(block) if gate else block))
    parts.append(engine.flush())
    out = np.concatenate(parts)
    want = len(audio) * engine.output_rate // engine.input_rate
    return out[delay:delay + want]
