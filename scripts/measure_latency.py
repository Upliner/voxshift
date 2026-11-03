#!/usr/bin/env python3
"""Measure real end-to-end latency and continuity through the audio stack.

Plays clicks into a temporary null sink, runs the pipeline with that sink's
monitor as its "microphone" and the virtual mic as output, then records the
virtual mic the way another app would. No physical device is used.

    measure_latency.py [--engine passthrough] [--voice NAME] [--seconds 10]
                       [--speech CLIP]

Voice models only reproduce speech, so clicks and tones vanish in them. With
--speech the test signal is word onsets cut from CLIP, separated by silence.
"""

from __future__ import annotations

import argparse
import subprocess
import threading
import time

import numpy as np

import soundfile as sf
import soxr

from voxshift import audio
from voxshift.pipeline import Pipeline
from voxshift.session import Options, build_engine

SINK = "voxshift_latency_test"
RATE = 48000
PERIOD = RATE // 2
MEASURE_BLOCK = RATE // 100


def click_train(seconds: float) -> np.ndarray:
    x = np.zeros(int(seconds * RATE), np.float32)
    for start in range(RATE // 2, len(x) - 480, PERIOD):
        x[start:start + 240] = 0.8 * np.sin(2 * np.pi * 1000 * np.arange(240) / RATE)
    return x


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def load_speech(path: str, seconds: float) -> np.ndarray:
    x, rate = sf.read(path, dtype="float32")
    return soxr.resample(x, rate, RATE).astype(np.float32)[:int(seconds * RATE)]


def word_onsets(speech: np.ndarray, seconds: float) -> np.ndarray:
    """Half-second cuts of loud speech, each after a second of silence."""
    frame = RATE // 100
    loud = [i for i in range(0, len(speech) - RATE, frame) if np.abs(speech[i:i + frame]).max() > 0.2]
    x = np.zeros(int(seconds * RATE), np.float32)
    burst = np.linspace(1, 0, RATE // 2, dtype=np.float32) ** 0.25
    burst[:96] *= np.linspace(0, 1, 96, dtype=np.float32)
    slots = range(RATE // 2, len(x) - RATE, 3 * PERIOD)
    for n, slot in enumerate(slots):
        start = loud[n * len(loud) // len(slots)]
        x[slot:slot + len(burst)] = speech[start:start + len(burst)] * burst
    return x


def onsets(x: np.ndarray, threshold: float = 0.1, spacing: int = RATE // 8) -> np.ndarray:
    hits = np.where(np.abs(x) > threshold)[0]
    return hits[np.insert(np.diff(hits) > spacing, 0, True)] if len(hits) else hits


def count_gaps(x: np.ndarray) -> tuple[int, float]:
    """Silent stretches inside a recorded tone, ignoring its start and end."""
    active = np.where(np.abs(x) > 0.01)[0]
    if len(active) < RATE:
        return -1, 0.0
    body = x[active[0] + 4800:active[-1] - 4800]
    envelope = np.convolve(np.abs(body), np.ones(96) / 96, "same")
    quiet = (envelope[96:-96] < 0.05).astype(int)
    edges = np.diff(np.concatenate([[0], quiet, [0]]))
    lengths = np.where(edges == -1)[0] - np.where(edges == 1)[0]
    return len(lengths), float(lengths.sum() * 1000 / RATE)


def dropouts(x: np.ndarray) -> int:
    """Stretches of 1 ms or more of exact silence: a model never outputs those, a dry stream does."""
    active = np.where(np.abs(x) > 0.01)[0]
    body = (x[active[0]:active[-1]] == 0).astype(int)
    edges = np.diff(np.concatenate([[0], body, [0]]))
    return int((np.where(edges == -1)[0] - np.where(edges == 1)[0] >= RATE // 1000).sum())


def run(engine, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    """Returns (what was played, what the virtual mic produced, pipeline), time-aligned."""
    capture = audio.Capture(engine.input_rate, SINK, audio_latency, monitor=True)
    feed = audio.Playback(engine.output_rate, audio.virtual_mic_playback_target(), audio_latency,
                          prebuffer, name="voxshift_mic_feed")
    pipeline = Pipeline(engine, capture, [feed], None)
    pipeline.start()

    tap = audio.Capture(RATE, audio.VIRTUAL_MIC, audio_latency)
    tapped: list[np.ndarray] = []

    def read_tap() -> None:
        while (block := tap.read(MEASURE_BLOCK)) is not None:
            tapped.append(block)

    threading.Thread(target=read_tap, daemon=True).start()
    time.sleep(0.5)

    # A second tap on the test sink records what the pipeline was sent. Both taps
    # share one graph clock, so the two recordings can be compared sample for sample.
    reference = audio.Capture(RATE, SINK, audio_latency, monitor=True)
    ref_blocks: list[np.ndarray] = []

    def read_reference() -> None:
        while (block := reference.read(MEASURE_BLOCK)) is not None:
            ref_blocks.append(block)

    threading.Thread(target=read_reference, daemon=True).start()
    time.sleep(0.2)
    tapped.clear()
    ref_blocks.clear()

    player = audio.Playback(RATE, SINK, 5, 0, max_backlog_ms=1000, name="voxshift_test_signal")
    for start in range(0, len(signal), 2400):
        player.write(signal[start:start + 2400])
        time.sleep(0.049)
    time.sleep(1.0)

    played = np.concatenate(ref_blocks)
    produced = np.concatenate(tapped)
    n = min(len(played), len(produced))
    player.close()
    pipeline.stop()
    tap.close()
    reference.close()
    return played[:n], produced[:n], pipeline


def main() -> None:
    global audio_latency, prebuffer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", default="passthrough")
    p.add_argument("--voice")
    p.add_argument("--speech", help="audio clip to use as the test signal")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--block-ms", type=float)
    p.add_argument("--device-latency", type=int, default=10)
    p.add_argument("--prebuffer", type=int, default=40)
    a = p.parse_args()
    audio_latency, prebuffer = a.device_latency, a.prebuffer

    default_sink = subprocess.check_output(["pactl", "get-default-sink"], text=True).strip()
    default_source = subprocess.check_output(["pactl", "get-default-source"], text=True).strip()
    module = subprocess.check_output(["pactl", "load-module", "module-null-sink", f"sink_name={SINK}",
                                      "channel_map=mono", f"rate={RATE}"], text=True).strip()
    created = not audio.virtual_mic_exists()
    audio.create_virtual_mic()
    try:
        time.sleep(0.5)
        engine = build_engine(Options(engine=a.engine, voice=a.voice, block_ms=a.block_ms))

        speech = load_speech(a.speech, a.seconds) if a.speech else None
        signal = click_train(a.seconds) if speech is None else word_onsets(speech, a.seconds * 1.5)
        spacing = RATE // 8 if speech is None else RATE
        played, produced, pipeline = run(engine, signal)
        sent, received = onsets(played, spacing=spacing), onsets(produced, spacing=spacing)
        delays = [(r - s) * 1000 / RATE for s in sent
                  if len(later := received[received >= s]) and (r := later[0]) - s < (RATE // 3 if speech is None else RATE)]
        print(f"clicks sent {len(sent)}, received {len(received)}, matched {len(delays)}")
        if delays:
            print(f"latency: median {np.median(delays):.1f} ms, min {min(delays):.1f}, max {max(delays):.1f}, "
                  f"last {delays[-1]:.1f}")

        engine.reset()
        played, produced, pipeline = run(engine, tone(a.seconds) if speech is None else speech)
        s = pipeline.stats
        if speech is None:
            gaps, gap_ms = count_gaps(produced)
            print(f"continuity: {gaps} gaps, {gap_ms:.1f} ms total over {a.seconds:.0f} s; {s.overruns} overruns")
        else:
            print(f"continuity: {dropouts(produced)} dropouts over {a.seconds:.0f} s; {s.overruns} overruns, "
                  f"{pipeline.sinks[0].dropped} samples dropped; infer peak {s.peak_ms:.1f} ms")
    finally:
        if created:
            audio.remove_virtual_mic()
        subprocess.run(["pactl", "unload-module", module], check=True)
    after = subprocess.check_output(["pactl", "get-default-sink"], text=True).strip()
    after_source = subprocess.check_output(["pactl", "get-default-source"], text=True).strip()
    if (after, after_source) != (default_sink, default_source):
        print(f"WARNING: default devices changed: {after} / {after_source}")


if __name__ == "__main__":
    main()
