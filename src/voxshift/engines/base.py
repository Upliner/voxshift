from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Engine(ABC):
    """A streaming converter: fixed input blocks in, audio out.

    Callers feed exactly `block_samples` at `input_rate` to each process() call.
    The output length may vary per call, and its rate is `output_rate`.
    """

    name: str
    input_rate: int
    output_rate: int
    block_samples: int

    @property
    @abstractmethod
    def algorithmic_latency_ms(self) -> float:
        """Delay inherent to the model's buffering, excluding inference and devices."""

    @abstractmethod
    def process(self, block: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def reset(self) -> None:
        """Forget streaming state, as if starting a new utterance."""

    def flush(self) -> np.ndarray:
        """Return any audio still held at the end of a finite input."""
        return np.zeros(0, dtype=np.float32)

    def set_pitch(self, semitones: float) -> None:
        """Change the pitch shift live. Engines without pitch control ignore it."""


def make_engine(name: str, **options) -> Engine:
    if name == "rvc":
        from .rvc import RvcEngine

        return RvcEngine(**options)
    if name == "meanvc2":
        from .meanvc2 import MeanVC2Engine

        return MeanVC2Engine(**options)
    if name == "passthrough":
        from .passthrough import PassthroughEngine

        return PassthroughEngine(**options)
    raise SystemExit(f"Unknown engine '{name}'")


ENGINES = ("rvc", "meanvc2", "passthrough")
