"""Train an RVC v2 voice from a directory of clean single-speaker audio.

Drives the upstream RVC scripts the same way its WebUI does, then packages the
result as a voice profile under voices/.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .paths import RVC_DIR, VOICES_DIR
from .voices import Voice

SAMPLE_RATE = "48k"
SLICE_SECONDS = 3.7
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".opus"}


def _run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    # webui.py sets these for itself. -P stops train/ landing on sys.path, where
    # train/train.py would shadow the train package the scripts import from the repo root.
    env = {"PYTHONPATH": str(RVC_DIR), "weight_root": "assets/weights", "index_root": "logs",
           "outside_index_root": "assets/indices", "rmvpe_root": "assets/rmvpe", **os.environ}
    subprocess.run([sys.executable, "-P", *args], cwd=RVC_DIR, env=env, check=True)


def select_subset(src: Path, dest: Path, minutes: float, seed: int = 0) -> list[Path]:
    """Link a random selection of clips totalling `minutes` of audio into dest."""
    clips = sorted(p for p in src.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if not clips:
        raise SystemExit(f"No audio files under {src}")
    random.Random(seed).shuffle(clips)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    total = 0.0
    chosen = []
    for clip in clips:
        (dest / clip.name).symlink_to(clip.resolve())
        chosen.append(clip)
        total += sf.info(clip).duration
        if total >= minutes * 60:
            break
    print(f"Selected {len(chosen)} clips, {total / 60:.1f} min", flush=True)
    return chosen


def write_filelist(exp: Path) -> None:
    gt, feat, f0, f0nsf = (exp / d for d in ("0_gt_wavs", "3_feature768", "2a_f0", "2b-f0nsf"))
    names = {p.stem for p in gt.iterdir()} & {p.stem for p in feat.iterdir()} \
        & {p.name[:-len(".wav.npy")] for p in f0.iterdir()} \
        & {p.name[:-len(".wav.npy")] for p in f0nsf.iterdir()}
    if not names:
        raise SystemExit("Feature extraction produced no usable clips")
    lines = [f"{gt}/{n}.wav|{feat}/{n}.npy|{f0}/{n}.wav.npy|{f0nsf}/{n}.wav.npy|0" for n in sorted(names)]
    mute = RVC_DIR / "logs" / "mute"
    lines += [f"{mute}/0_gt_wavs/mute{SAMPLE_RATE}.wav|{mute}/3_feature768/mute.npy|"
              f"{mute}/2a_f0/mute.wav.npy|{mute}/2b-f0nsf/mute.wav.npy|0"] * 2
    random.shuffle(lines)
    (exp / "filelist.txt").write_text("\n".join(lines))
    shutil.copyfile(RVC_DIR / "configs" / "v2" / f"{SAMPLE_RATE}.json", exp / "config.json")


def measure_median_f0(clips: list[Path], sample: int = 40) -> float:
    values = []
    for clip in clips[:sample]:
        y, sr = librosa.load(clip, sr=16000, mono=True)
        f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=500, sr=sr)
        values.append(f0[voiced])
    return float(np.nanmedian(np.concatenate(values)))


def pick_reference(clips: list[Path], dest: Path, seconds: float = 15.0) -> None:
    """Concatenate the first clips into the reference used by zero-shot engines."""
    parts, total = [], 0.0
    for clip in clips:
        y, sr = sf.read(clip, always_2d=False)
        parts.append(y if y.ndim == 1 else y.mean(axis=1))
        total += len(y) / sr
        if total >= seconds:
            break
    sf.write(dest, np.concatenate(parts), sr)


def train(name: str, dataset: Path, minutes: float, epochs: int, batch_size: int,
          source: str, license: str) -> Voice:
    rvc_data = RVC_DIR / "logs"
    exp = rvc_data / name
    stage = rvc_data / f"{name}_dataset"
    exp.mkdir(parents=True, exist_ok=True)
    (RVC_DIR / "assets" / "weights").mkdir(exist_ok=True)
    clips = select_subset(dataset, stage, minutes)

    n_cpu = str(max(os.cpu_count() // 2, 1))
    if not (exp / "3_feature768").exists():
        _run("train/preprocess.py", str(stage), "48000", n_cpu, str(exp), "False", str(SLICE_SECONDS))
        _run("train/dataset/extract_f0.py", "cuda", "1", "0", "0", str(exp), "True")
        _run("train/dataset/extract_hubert_feature.py", "cuda", "1", "0", "0", str(exp), "v2", "True")
    write_filelist(exp)
    _run("train/train.py", "-e", name, "-sr", SAMPLE_RATE, "-f0", "1", "-bs", str(batch_size),
         "-g", "0", "-te", str(epochs), "-se", str(max(epochs // 4, 1)),
         "-pg", f"assets/pretrained_v2/f0G{SAMPLE_RATE}.pth",
         "-pd", f"assets/pretrained_v2/f0D{SAMPLE_RATE}.pth",
         "-l", "1", "-c", "0", "-sw", "1", "-v", "v2")
    _run("train/train_index.py", name, "v2", str(exp), n_cpu, "auto")
    return package(name, clips, source, license)


def package(name: str, clips: list[Path], source: str, license: str) -> Voice:
    weights = RVC_DIR / "assets" / "weights"
    latest = max(weights.glob(f"{name}*.pth"), key=lambda p: p.stat().st_mtime)
    indexes = sorted((RVC_DIR / "logs" / name).rglob("added*.index"))
    if not indexes:
        raise SystemExit(f"No index produced for {name}")
    dest = VOICES_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(latest, dest / "model.pth")
    shutil.copyfile(indexes[-1], dest / "model.index")
    pick_reference(clips, dest / "reference.wav")
    voice = Voice(name=name, path=dest, median_f0=measure_median_f0(clips), source=source, license=license)
    voice.save()
    print(json.dumps({"voice": name, "median_f0": voice.median_f0, "model": latest.name}), flush=True)
    return voice
