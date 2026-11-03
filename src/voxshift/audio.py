"""Audio I/O through PipeWire's pw-cat, with PulseAudio tools as a fallback.

Streams run as child processes exchanging raw mono float32 over pipes, so
PipeWire handles device routing and sample-rate conversion. Python can't
meet realtime deadlines itself; the pipes and a jitter prebuffer absorb its
scheduling noise.
"""

from __future__ import annotations

import array
import fcntl
import os
import shutil
import subprocess
import termios
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BYTES_PER_SAMPLE = 4
F_SETPIPE_SZ = 1031

VIRTUAL_MIC = "voxshift_mic"


def backend() -> str:
    """Pick the audio toolset: 'pipewire' when pw-cat exists, else 'pulse'."""
    if shutil.which("pw-cat"):
        return "pipewire"
    if shutil.which("parec") and shutil.which("pacat"):
        return "pulse"
    raise SystemExit("Neither pw-cat nor parec/pacat found: install PipeWire or PulseAudio tools")


def capture_cmd(rate: int, target: str | None, latency_ms: int, monitor: bool = False) -> list[str]:
    if backend() == "pipewire":
        cmd = ["pw-cat", "--record", "--raw", "--format", "f32", "--rate", str(rate),
               "--channels", "1", "--latency", f"{latency_ms}ms",
               "-P", "node.name=voxshift_capture"]
        if monitor:
            cmd += ["-P", "stream.capture.sink=true"]
        if target:
            cmd += ["--target", target]
        return cmd + ["-"]
    cmd = ["parec", "--format=float32le", f"--rate={rate}", "--channels=1",
           f"--latency-msec={latency_ms}", "--client-name=voxshift_capture"]
    if target:
        cmd.append(f"--device={target}.monitor" if monitor else f"--device={target}")
    return cmd


def playback_cmd(rate: int, target: str | None, latency_ms: int, name: str) -> list[str]:
    if backend() == "pipewire":
        cmd = ["pw-cat", "--playback", "--raw", "--format", "f32", "--rate", str(rate),
               "--channels", "1", "--latency", f"{latency_ms}ms", "-P", f"node.name={name}"]
        if target:
            cmd += ["--target", target]
        return cmd + ["-"]
    cmd = ["pacat", "--playback", "--format=float32le", f"--rate={rate}", "--channels=1",
           f"--latency-msec={latency_ms}", f"--client-name={name}"]
    if target:
        cmd.append(f"--device={target}")
    return cmd


def stop_process(proc: subprocess.Popen) -> None:
    """End a stream. pw-cat ignores SIGTERM while blocked on its pipe, hence the kill."""
    proc.terminate()
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class Capture:
    """A microphone (or sink monitor) stream read in blocks."""

    def __init__(self, rate: int, target: str | None = None, latency_ms: int = 10, monitor: bool = False):
        self.proc = subprocess.Popen(capture_cmd(rate, target, latency_ms, monitor),
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.rate = rate

    def read(self, samples: int) -> np.ndarray | None:
        """Block until `samples` have arrived; None once the stream has ended."""
        want = samples * BYTES_PER_SAMPLE
        buf = bytearray()
        while len(buf) < want:
            chunk = self.proc.stdout.read(want - len(buf))
            if not chunk:
                return None
            buf += chunk
        return np.frombuffer(bytes(buf), dtype=np.float32).copy()

    def close(self) -> None:
        stop_process(self.proc)


class Playback:
    """A playback stream fed without blocking, holding latency near a target.

    pw-cat glitches on a pipe that runs dry, so whenever a write finds the pipe
    empty (at startup or after an underrun) `prebuffer_ms` of silence goes ahead
    of it as jitter margin. Priming earlier would be wasted: the stream plays it
    out before any audio arrives. The two ends run on different clocks and can
    drift, so writes past `max_backlog_ms` are dropped rather than letting
    latency creep up.
    """

    def __init__(self, rate: int, target: str | None = None, latency_ms: int = 10,
                 prebuffer_ms: int = 40, max_backlog_ms: int = 200, name: str = "voxshift_output"):
        self.proc = subprocess.Popen(playback_cmd(rate, target, latency_ms, name),
                                     stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.fd = self.proc.stdin.fileno()
        self.rate = rate
        self.dropped = 0
        self.max_backlog = max_backlog_ms * rate // 1000 * BYTES_PER_SAMPLE
        limit = int(Path("/proc/sys/fs/pipe-max-size").read_text())
        fcntl.fcntl(self.fd, F_SETPIPE_SZ, min(max(self.max_backlog * 2, 65536), limit))
        os.set_blocking(self.fd, False)
        self.silence = bytes(prebuffer_ms * rate // 1000 * BYTES_PER_SAMPLE)

    def backlog_bytes(self) -> int:
        n = array.array("i", [0])
        fcntl.ioctl(self.fd, termios.FIONREAD, n)
        return n[0]

    def write(self, samples: np.ndarray) -> None:
        data = samples.astype(np.float32).tobytes()
        backlog = self.backlog_bytes()
        if backlog == 0:
            data = self.silence + data
        if backlog + len(data) > self.max_backlog:
            self.dropped += len(samples)
            return
        try:
            written = os.write(self.fd, data)
        except BlockingIOError:
            written = 0
        except BrokenPipeError:
            raise RuntimeError("Audio output stream exited") from None
        self.dropped += (len(data) - written) // BYTES_PER_SAMPLE

    def close(self) -> None:
        self.proc.stdin.close()
        stop_process(self.proc)


@dataclass
class Device:
    name: str
    description: str
    kind: str  # "source" or "sink"


def list_devices() -> list[Device]:
    """Sources and sinks known to the audio server, monitors excluded."""
    devices = []
    for kind, section in (("source", "Source"), ("sink", "Sink")):
        out = subprocess.check_output(["pactl", "list", f"{kind}s"], text=True, errors="replace",
                                      env={**os.environ, "LC_ALL": "C"})
        for block in out.split(f"\n{section} #"):
            fields = dict(line.strip().split(": ", 1) for line in block.splitlines() if ": " in line)
            name = fields.get("Name")
            if name and not name.endswith(".monitor"):
                devices.append(Device(name, fields.get("Description", name), kind))
    return devices


def _pactl(*args: str) -> str:
    return subprocess.check_output(["pactl", *args], text=True).strip()


def _virtual_mic_modules() -> list[str]:
    ids = []
    for line in _pactl("list", "short", "modules").splitlines():
        parts = line.split("\t")
        if VIRTUAL_MIC in line and parts[1] in ("module-null-sink", "module-remap-source"):
            ids.append(parts[0])
    return ids


def virtual_mic_exists() -> bool:
    return VIRTUAL_MIC in {d.name for d in list_devices()}


def create_virtual_mic() -> None:
    """Expose a device other apps can select as their microphone. Idempotent."""
    if virtual_mic_exists():
        return
    props = "device.description=VoxShift_Virtual_Mic"
    if backend() == "pipewire":
        # A null sink with the Source class appears as a plain input to apps and
        # still accepts our playback stream by name.
        _pactl("load-module", "module-null-sink", "media.class=Audio/Source/Virtual",
               f"sink_name={VIRTUAL_MIC}", "channel_map=mono", "rate=48000",
               f"sink_properties={props}")
    else:
        _pactl("load-module", "module-null-sink", f"sink_name={VIRTUAL_MIC}_sink",
               "channel_map=mono", "rate=48000")
        _pactl("load-module", "module-remap-source", f"master={VIRTUAL_MIC}_sink.monitor",
               f"source_name={VIRTUAL_MIC}", "channel_map=mono",
               f"source_properties={props}")


def remove_virtual_mic() -> None:
    for module in reversed(_virtual_mic_modules()):
        _pactl("unload-module", module)


def virtual_mic_playback_target() -> str:
    """The node our output stream connects to in order to feed the virtual mic."""
    return VIRTUAL_MIC if backend() == "pipewire" else f"{VIRTUAL_MIC}_sink"
