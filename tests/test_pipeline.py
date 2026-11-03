import time

import numpy as np
import soundfile as sf

from voxshift.engines.passthrough import PassthroughEngine
from voxshift.pipeline import FileSink, NoiseGate, Pipeline, convert_stream, level_db

RATE = 48000


def tone(seconds, amplitude=0.3):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeCapture:
    def __init__(self, audio, block_interval=0.0):
        self.audio = audio
        self.position = 0
        self.closed = False
        self.block_interval = block_interval

    def read(self, samples):
        if self.position + samples > len(self.audio):
            return None
        time.sleep(self.block_interval)
        block = self.audio[self.position:self.position + samples]
        self.position += samples
        return block

    def close(self):
        self.closed = True


class ListSink:
    def __init__(self):
        self.parts = []
        self.closed = False

    def write(self, samples):
        self.parts.append(samples)

    def close(self):
        self.closed = True


class DelayEngine(PassthroughEngine):
    """Mimics a model that returns its input a fixed time later."""

    def __init__(self, delay_samples):
        super().__init__()
        self.delay = delay_samples
        self.reset()

    @property
    def algorithmic_latency_ms(self):
        return self.delay * 1000 / self.output_rate

    def reset(self):
        self.held = np.zeros(self.delay, dtype=np.float32)

    def process(self, block):
        joined = np.concatenate([self.held, block])
        self.held = joined[len(block):]
        return joined[:len(block)]


def test_gate_silences_noise_and_keeps_speech():
    gate = NoiseGate(RATE, threshold_db=-40)
    assert np.abs(gate.process(tone(0.5, 0.001))).max() == 0
    gate.process(tone(0.1))  # opens during the first frames
    assert level_db(gate.process(tone(0.1))) > -14


def test_gate_holds_through_short_pause_then_closes():
    gate = NoiseGate(RATE, threshold_db=-40, hold_ms=100)
    gate.process(tone(0.1))
    assert np.abs(gate.process(np.ones(int(0.05 * RATE), np.float32) * 0.001)).max() > 0
    assert np.abs(gate.process(np.ones(int(0.3 * RATE), np.float32) * 0.001)[-480:]).max() == 0


def test_gate_gain_never_steps():
    gate = NoiseGate(RATE, threshold_db=-40, hold_ms=0)
    signal = np.concatenate([np.zeros(4800), np.full(4800, 0.5), np.zeros(4800)]).astype(np.float32)
    assert np.abs(np.diff(gate.process(signal))).max() < 0.01


def test_convert_stream_removes_engine_delay_and_keeps_length():
    audio = tone(1.0)
    out = convert_stream(DelayEngine(2400), audio)
    assert len(out) == len(audio)
    assert np.allclose(out, audio, atol=1e-6)


def test_pipeline_delivers_all_blocks_to_every_sink():
    audio = tone(1.0)
    capture = FakeCapture(audio, block_interval=0.002)  # paced, like a device
    first, second = ListSink(), ListSink()
    pipeline = Pipeline(PassthroughEngine(), capture, [first, second], None)
    pipeline.start()
    while pipeline.running:
        pass
    pipeline.stop()
    assert capture.closed and first.closed and second.closed
    assert pipeline.error is None
    delivered = np.concatenate(first.parts)
    assert pipeline.stats.overruns == 0
    assert np.array_equal(delivered, audio[:len(delivered)])
    assert len(audio) - len(delivered) < pipeline.engine.block_samples
    assert len(second.parts) == len(first.parts)


def test_pipeline_reports_engine_failure():
    class Broken(PassthroughEngine):
        def process(self, block):
            raise RuntimeError("out of memory")

    pipeline = Pipeline(Broken(), FakeCapture(tone(1.0)), [ListSink()], None)
    pipeline.start()
    pipeline.threads[1].join(timeout=2)
    pipeline.stop()
    assert "out of memory" in str(pipeline.error)


def test_pipeline_drops_old_blocks_when_conversion_falls_behind():
    import threading

    release = threading.Event()

    class Stalled(PassthroughEngine):
        def process(self, block):
            release.wait(2)
            return block

    pipeline = Pipeline(Stalled(), FakeCapture(tone(1.0)), [ListSink()], None)
    pipeline.start()
    pipeline.threads[0].join(timeout=2)
    release.set()
    pipeline.stop()
    assert pipeline.stats.overruns > 0


def test_file_sink_writes_everything(tmp_path):
    path = tmp_path / "out.flac"
    sink = FileSink(path, RATE)
    audio = tone(0.5)
    for start in range(0, len(audio), 960):
        sink.write(audio[start:start + 960])
    sink.close()
    written, rate = sf.read(path, dtype="float32")
    assert rate == RATE and len(written) == len(audio)
    assert np.allclose(written, audio, atol=1e-3)
