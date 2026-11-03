"""Voice profiles: a directory under voices/ describing one target voice.

voice.json holds the metadata; model.pth and model.index are the RVC model,
reference.wav is the clip zero-shot engines take the speaker from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import VOICES_DIR


@dataclass
class Voice:
    name: str
    path: Path
    median_f0: float
    source: str = ""
    license: str = ""

    @property
    def model(self) -> Path:
        return self.path / "model.pth"

    @property
    def index(self) -> Path:
        return self.path / "model.index"

    @property
    def reference(self) -> Path:
        return self.path / "reference.wav"

    def save(self) -> None:
        data = asdict(self)
        del data["path"]
        (self.path / "voice.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_voice(name: str) -> Voice:
    path = VOICES_DIR / name
    meta = path / "voice.json"
    if not meta.exists():
        known = ", ".join(v.name for v in list_voices()) or "none"
        raise SystemExit(f"Unknown voice '{name}' (available: {known})")
    return Voice(path=path, **json.loads(meta.read_text()))


def list_voices() -> list[Voice]:
    if not VOICES_DIR.exists():
        return []
    return [load_voice(p.parent.name) for p in sorted(VOICES_DIR.glob("*/voice.json"))]
