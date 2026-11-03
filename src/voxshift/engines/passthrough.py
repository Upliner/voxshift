from __future__ import annotations

import numpy as np

from .base import Engine


class PassthroughEngine(Engine):
    """Copies input to output. Used to test and measure the audio path alone."""

    name = "passthrough"

    def __init__(self, rate: int = 48000, block_ms: float = 20.0, **_):
        self.input_rate = self.output_rate = rate
        self.block_samples = int(rate * block_ms / 1000)

    @property
    def algorithmic_latency_ms(self) -> float:
        return self.block_samples * 1000 / self.input_rate

    def process(self, block: np.ndarray) -> np.ndarray:
        return block

    def reset(self) -> None:
        pass
