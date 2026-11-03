#!/usr/bin/env python3
"""Objective checks on converted speech across languages.

Converts each clip in data/test/, then reports pitch, formants against the
target voice, and intelligibility. There are no ground-truth transcripts, so
character error rate compares Whisper's reading of the output with its reading
of the source. None of this says whether the voice sounds convincing: listen to
the files for that.

    evaluate.py --voice lada --block-ms 30 --pitch auto [--langs uk,ru]
    evaluate.py --engine meanvc2 --voice lada
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import jiwer
import numpy as np
import parselmouth
import soundfile as sf
import soxr
from faster_whisper import WhisperModel

from voxshift.paths import ROOT
from voxshift.pipeline import NoiseGate, convert_stream
from voxshift.session import Options, build_engine
from voxshift.voices import load_voice

TEST_DIR = ROOT / "data" / "test"
OUT_DIR = ROOT / "data" / "out"


def voice_stats(y: np.ndarray, rate: int) -> dict[str, float]:
    """Median F0 and median F1-F3 over voiced frames."""
    sound = parselmouth.Sound(y.astype(np.float64), sampling_frequency=rate)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=600)
    f0 = pitch.selected_array["frequency"]
    times = pitch.xs()[f0 > 0]
    # 5500 Hz is Praat's recommended ceiling for female voices; 5000 for male.
    ceiling = 5500 if np.median(f0[f0 > 0]) > 165 else 5000
    formants = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=ceiling)
    stats = {"f0": float(np.median(f0[f0 > 0])), "voiced": float((f0 > 0).mean())}
    for n in (1, 2, 3):
        values = np.array([formants.get_value_at_time(n, t) for t in times])
        stats[f"f{n}"] = float(np.nanmedian(values))
    return stats


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", default="rvc")
    p.add_argument("--voice", default="lada")
    p.add_argument("--block-ms", type=float, default=100)
    p.add_argument("--pitch", default="auto", help='semitones, or "auto" to match the voice per clip')
    p.add_argument("--langs", help="comma-separated, default every clip in data/test")
    p.add_argument("--whisper", default="medium")
    p.add_argument("--whisper-device", default="cpu")
    a = p.parse_args()

    voice = load_voice(a.voice)
    engine = build_engine(Options(engine=a.engine, voice=a.voice, pitch=12.0, block_ms=a.block_ms))
    y, rate = sf.read(voice.reference, dtype="float32")
    target = voice_stats(y, rate)
    print("target", {k: round(v, 1) for k, v in target.items()})

    whisper = WhisperModel(a.whisper, device=a.whisper_device,
                           compute_type="int8" if a.whisper_device == "cpu" else "float16")

    def read(x: np.ndarray, source_rate: int, lang: str) -> str:
        audio = soxr.resample(x, source_rate, 16000).astype(np.float32)
        segments, _ = whisper.transcribe(audio, language=lang, beam_size=5, vad_filter=False)
        return normalise(" ".join(s.text for s in segments))

    tag = f"{a.voice}_{a.block_ms:g}ms_{a.pitch}" if a.engine == "rvc" else f"{a.voice}_{a.engine}"
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    langs = a.langs.split(",") if a.langs else sorted(p.stem for p in TEST_DIR.glob("*.wav"))
    rows = []
    print("lang  shift  F0 in>out   F1 in>out   F2 in>out   F3 in>out   voiced   CER")
    for lang in langs:
        x, rate = sf.read(TEST_DIR / f"{lang}.wav", dtype="float32")
        x = x[:int(30 * rate)]
        source = voice_stats(x, rate)
        shift = 12 * math.log2(target["f0"] / source["f0"]) if a.pitch == "auto" else float(a.pitch)
        engine.set_pitch(shift)
        resampled = soxr.resample(x, rate, engine.input_rate).astype(np.float32)
        y = convert_stream(engine, resampled, NoiseGate(engine.input_rate))
        sf.write(out_dir / f"{lang}.wav", y, engine.output_rate)
        result = voice_stats(y, engine.output_rate)
        reference, hypothesis = read(x, rate, lang), read(y, engine.output_rate, lang)
        cer = jiwer.cer(reference, hypothesis) if reference else float("nan")
        rows.append({"lang": lang, "shift": shift, "source": source, "result": result, "cer": cer,
                     "reference": reference, "hypothesis": hypothesis})
        print(f"{lang:4s} {shift:+5.1f}  {source['f0']:3.0f}>{result['f0']:3.0f}     "
              f"{source['f1']:3.0f}>{result['f1']:3.0f}     {source['f2']:4.0f}>{result['f2']:4.0f}  "
              f"{source['f3']:4.0f}>{result['f3']:4.0f}  {source['voiced']:.2f}>{result['voiced']:.2f}  "
              f"{cer:.3f}", flush=True)
    print(f"mean CER {np.mean([r['cer'] for r in rows]):.3f}; files in {out_dir}")
    Path(out_dir / "report.json").write_text(json.dumps({"target": target, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
