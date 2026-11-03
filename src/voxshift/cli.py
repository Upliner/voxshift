"""Command line interface: `voxshift --help`."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import audio
from .engines.base import ENGINES
from .session import Options, build_engine, save_user_f0, start_session
from .voices import list_voices


def add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--engine", choices=ENGINES, default="rvc")
    p.add_argument("--voice", help="voice profile name (see `voxshift voices`)")
    p.add_argument("--pitch", type=float, help="semitones; default comes from `voxshift calibrate`, else +12")
    p.add_argument("--block-ms", type=float, help="RVC block size: lower is faster but needs a faster GPU")
    p.add_argument("--model", choices=("40ms", "120ms"), help="MeanVC2 model")
    p.add_argument("--device", default="cuda")


def options(a: argparse.Namespace, **extra) -> Options:
    return Options(engine=a.engine, voice=a.voice, pitch=a.pitch, block_ms=a.block_ms,
                   model=a.model, device=a.device, **extra)


def cmd_devices(_) -> None:
    for d in audio.list_devices():
        print(f"{d.kind:6}  {d.name}\n        {d.description}")


def cmd_voices(_) -> None:
    for v in list_voices():
        print(f"{v.name:16} median F0 {v.median_f0:5.0f} Hz  {v.license:12} {v.source}")


def cmd_mic(a) -> None:
    if a.action == "create":
        audio.create_virtual_mic()
        print(f"Virtual microphone ready: {audio.VIRTUAL_MIC}")
    else:
        audio.remove_virtual_mic()
        print("Virtual microphone removed")


def cmd_run(a) -> None:
    opts = options(a, mic=a.mic, monitor=a.monitor, monitor_device=a.monitor_device,
                   virtual_mic=a.virtual_mic, record=a.record,
                   gate_db=None if a.no_gate else a.gate_db)
    print("Loading engine...")
    engine = build_engine(opts)
    pipeline, created_mic = start_session(engine, opts)
    block_ms = engine.block_samples * 1000 / engine.input_rate
    print(f"Running {engine.name}: block {block_ms:.0f} ms, algorithmic latency "
          f"{engine.algorithmic_latency_ms:.0f} ms. Ctrl+C to stop.")
    try:
        while pipeline.running:
            time.sleep(1)
            s = pipeline.stats
            print(f"\rinfer {s.last_ms:5.1f} ms (peak {s.peak_ms:5.1f}/{block_ms:.0f})  overruns {s.overruns}  "
                  f"in {s.input_db:6.1f} dB  out {s.output_db:6.1f} dB ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()
    pipeline.stop()
    if created_mic and not a.keep_mic:
        audio.remove_virtual_mic()
    if pipeline.error:
        raise SystemExit(f"Conversion failed: {pipeline.error}")


def cmd_convert(a) -> None:
    import soundfile as sf
    import soxr

    from .pipeline import convert_stream

    engine = build_engine(options(a))
    x, rate = sf.read(a.input, dtype="float32", always_2d=True)
    x = soxr.resample(x.mean(axis=1), rate, engine.input_rate)
    start = time.perf_counter()
    y = convert_stream(engine, x.astype(np.float32))
    elapsed = time.perf_counter() - start
    sf.write(a.output, y, engine.output_rate)
    print(f"{a.output}: {len(x) / engine.input_rate:.1f} s converted in {elapsed:.1f} s")


def cmd_calibrate(a) -> None:
    import librosa

    print(f"Speak normally for {a.seconds} s...")
    capture = audio.Capture(16000, a.mic)
    block = capture.read(a.seconds * 16000)
    capture.close()
    if block is None:
        raise SystemExit("Could not read from the microphone")
    f0, voiced, _ = librosa.pyin(block, fmin=60, fmax=400, sr=16000)
    if voiced.sum() < 100:
        raise SystemExit("Not enough speech heard: check the microphone and try again")
    median = float(np.nanmedian(f0[voiced]))
    save_user_f0(median)
    print(f"Your median pitch is {median:.0f} Hz. Saved.")


def cmd_train(a) -> None:
    from .train import train

    train(a.name, a.dataset, a.minutes, a.epochs, a.batch_size, a.source, a.license)


def cmd_gui(_) -> None:
    from .gui import main

    main()


def main() -> None:
    parser = argparse.ArgumentParser(prog="voxshift", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list audio devices").set_defaults(fn=cmd_devices)
    sub.add_parser("voices", help="list voice profiles").set_defaults(fn=cmd_voices)
    sub.add_parser("gui", help="open the graphical app").set_defaults(fn=cmd_gui)

    p = sub.add_parser("mic", help="manage the virtual microphone")
    p.add_argument("action", choices=("create", "remove"))
    p.set_defaults(fn=cmd_mic)

    p = sub.add_parser("run", help="convert the microphone live")
    add_engine_args(p)
    p.add_argument("--mic", help="input device name (default: system default)")
    p.add_argument("--monitor", action="store_true", help="play the result (use headphones)")
    p.add_argument("--monitor-device", help="output device name")
    p.add_argument("--virtual-mic", action="store_true", help="feed the virtual microphone")
    p.add_argument("--record", type=Path, help="write the result to this audio file")
    p.add_argument("--gate-db", type=float, default=-45.0, help="noise gate threshold")
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--keep-mic", action="store_true", help="leave the virtual mic in place on exit")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("convert", help="convert an audio file")
    add_engine_args(p)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("calibrate", help="measure your pitch to set the shift automatically")
    p.add_argument("--mic")
    p.add_argument("--seconds", type=int, default=15)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("train", help="train an RVC voice from single-speaker audio")
    p.add_argument("name")
    p.add_argument("dataset", type=Path)
    p.add_argument("--minutes", type=float, default=30)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--source", default="")
    p.add_argument("--license", default="")
    p.set_defaults(fn=cmd_train)

    a = parser.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
