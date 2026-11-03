import numpy as np
import pytest

from voxshift import audio

RATE = 48000
BLOCK = np.zeros(960, dtype=np.float32)  # 20 ms
BLOCK_BYTES = len(BLOCK) * audio.BYTES_PER_SAMPLE


@pytest.fixture
def playback(monkeypatch):
    """A Playback whose child never reads, so the pipe contents can be inspected."""
    monkeypatch.setattr(audio, "playback_cmd", lambda *_: ["sleep", "30"])
    created = []

    def make(**kwargs):
        created.append(audio.Playback(RATE, **kwargs))
        return created[-1]

    yield make
    for p in created:
        p.close()


def test_prebuffer_goes_in_with_first_write_not_at_open(playback):
    p = playback(prebuffer_ms=40)
    assert p.backlog_bytes() == 0
    p.write(BLOCK)
    assert p.backlog_bytes() == BLOCK_BYTES + 40 * RATE // 1000 * audio.BYTES_PER_SAMPLE


def test_prebuffer_not_repeated_while_pipe_has_data(playback):
    p = playback(prebuffer_ms=40)
    p.write(BLOCK)
    level = p.backlog_bytes()
    p.write(BLOCK)
    assert p.backlog_bytes() == level + BLOCK_BYTES


def test_writes_past_max_backlog_are_dropped(playback):
    p = playback(prebuffer_ms=0, max_backlog_ms=50)
    for _ in range(5):
        p.write(BLOCK)
    assert p.backlog_bytes() == 2 * BLOCK_BYTES
    assert p.dropped == 3 * len(BLOCK)


def test_close_ends_a_child_that_ignores_sigterm(monkeypatch):
    # pw-cat blocked on its pipe behaves like this.
    monkeypatch.setattr(audio, "playback_cmd",
                        lambda *_: ["sh", "-c", "trap '' TERM; sleep 30"])
    p = audio.Playback(RATE)
    p.close()
    assert p.proc.returncode is not None
