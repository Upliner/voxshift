#!/usr/bin/env python3
"""Build a multilingual male-speech test set from public-domain LibriVox audio.

Reader gender isn't in the metadata, so candidates are kept only when their
measured median F0 is in the typical male range. Clips land in data/test/ with
a manifest.json recording provenance and pitch.
"""

from __future__ import annotations

import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "test"
LANGS = {"uk": "ukr", "ru": "rus", "en": "eng", "de": "deu", "es": "spa", "fr": "fra", "pl": "pol", "it": "ita"}
MALE_F0_MAX = 155.0
CLIP_SECONDS = 40
SKIP_SECONDS = 75  # past the LibriVox spoken preamble


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def candidates(lang: str, rows: int = 25) -> list[str]:
    q = urllib.parse.quote(f"collection:librivoxaudio AND language:{lang}")
    url = f"https://archive.org/advancedsearch.php?q={q}&fl%5B%5D=identifier&rows={rows}&output=json"
    return [d["identifier"] for d in get_json(url)["response"]["docs"]]


def first_mp3(identifier: str) -> str | None:
    meta = get_json(f"https://archive.org/metadata/{identifier}")
    names = sorted(f["name"] for f in meta.get("files", []) if f["name"].endswith("_64kb.mp3"))
    return f"https://archive.org/download/{identifier}/{urllib.parse.quote(names[0])}" if names else None


def slice_clip(url: str, dest: Path) -> bool:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(SKIP_SECONDS), "-t", str(CLIP_SECONDS),
           "-i", url, "-ac", "1", "-ar", "48000", str(dest)]
    return subprocess.run(cmd, timeout=120).returncode == 0 and dest.exists()


def median_f0(path: Path) -> float:
    y, sr = librosa.load(path, sr=16000, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
    return float(np.nanmedian(f0[voiced])) if voiced.any() else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for code, lang in LANGS.items():
        if code in manifest:
            continue
        dest = OUT / f"{code}.wav"
        for identifier in candidates(lang):
            try:
                url = first_mp3(identifier)
                if not url or not slice_clip(url, dest):
                    continue
                dur = sf.info(dest).duration
                f0 = median_f0(dest)
            except Exception as e:  # one dead item shouldn't stop the language
                print(f"  {identifier}: {e}")
                continue
            print(f"{code} {identifier}: {dur:.0f}s median F0 {f0:.0f} Hz")
            if dur >= CLIP_SECONDS - 1 and f0 < MALE_F0_MAX:
                manifest[code] = {"source": url, "identifier": identifier, "median_f0": round(f0, 1),
                                  "license": "public domain (LibriVox)"}
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
                break
        else:
            dest.unlink(missing_ok=True)
            print(f"{code}: no male reader found")


if __name__ == "__main__":
    main()
