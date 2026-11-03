"""Assembles an engine and its audio streams into a running pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from . import audio
from .engines.base import Engine, make_engine
from .pipeline import FileSink, NoiseGate, Pipeline, Sink
from .voices import Voice, load_voice

CONFIG = Path.home() / ".config" / "voxshift" / "config.json"
DEFAULT_PITCH = 12.0
DEVICE_LATENCY_MS = 10
PREBUFFER_MS = 40


def load_config() -> dict:
    return json.loads(CONFIG.read_text()) if CONFIG.exists() else {}


def save_user_f0(f0: float) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({**load_config(), "user_f0": round(f0, 1)}, indent=2))


def pitch_for(voice: Voice) -> float:
    """Semitones from the calibrated speaker's median pitch to the voice's."""
    user_f0 = load_config().get("user_f0")
    return round(12 * math.log2(voice.median_f0 / user_f0), 1) if user_f0 else DEFAULT_PITCH


@dataclass
class Options:
    engine: str = "rvc"
    voice: str | None = None
    pitch: float | None = None
    mic: str | None = None
    monitor: bool = False
    monitor_device: str | None = None
    virtual_mic: bool = False
    record: Path | None = None
    gate_db: float | None = -45.0
    block_ms: float | None = None
    model: str | None = None
    device: str = "cuda"
    extra: dict = field(default_factory=dict)


def build_engine(opts: Options) -> Engine:
    kwargs = {**opts.extra, "device": opts.device}
    if opts.engine != "passthrough":
        if not opts.voice:
            raise SystemExit("--voice is required for this engine")
        voice = load_voice(opts.voice)
        kwargs.update(voice=voice, pitch=pitch_for(voice) if opts.pitch is None else opts.pitch)
    if opts.block_ms:
        kwargs["block_ms"] = opts.block_ms
    if opts.model:
        kwargs["model"] = opts.model
    return make_engine(opts.engine, **kwargs)


def start_session(engine: Engine, opts: Options) -> tuple[Pipeline, bool]:
    """Open the streams and start converting. Returns the pipeline and whether
    this call created the virtual mic (so the caller knows to remove it)."""
    sinks: list[Sink] = []
    created_mic = False
    if opts.virtual_mic:
        created_mic = not audio.virtual_mic_exists()
        audio.create_virtual_mic()
        sinks.append(audio.Playback(engine.output_rate, audio.virtual_mic_playback_target(),
                                    DEVICE_LATENCY_MS, PREBUFFER_MS, name="voxshift_mic_feed"))
    if opts.monitor:
        sinks.append(audio.Playback(engine.output_rate, opts.monitor_device,
                                    DEVICE_LATENCY_MS, PREBUFFER_MS, name="voxshift_monitor"))
    if opts.record:
        sinks.append(FileSink(opts.record, engine.output_rate))
    if not sinks:
        raise SystemExit("No output selected: use --monitor, --virtual-mic and/or --record")

    capture = audio.Capture(engine.input_rate, opts.mic, DEVICE_LATENCY_MS)
    gate = None if opts.gate_db is None else NoiseGate(engine.input_rate, opts.gate_db)
    pipeline = Pipeline(engine, capture, sinks, gate)
    pipeline.start()
    return pipeline, created_mic
