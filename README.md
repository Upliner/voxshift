# VoxShift

Realtime AI voice conversion for Linux, built for convincing male-to-female
conversion. Output can be played live, recorded to a file, or sent to a virtual
microphone that other apps select as their input.

## Status

| Part | State |
|---|---|
| Audio path, virtual mic, recording, GUI, CLI | Tested with the passthrough engine |
| RVC training | Works: 20 min of data, 30 epochs, about 20 min on an RTX 3060 |
| RVC engine | Works on files and through the virtual mic; `voxshift run` and the GUI not yet tried with it |
| MeanVC 2 engine | Works, but slower, lower-pitched and less intelligible than RVC here |
| How convincing the voice sounds | **Not judged**: needs listening |

## Engines

| Engine | Target voice | Output | Languages | Licence |
|---|---|---|---|---|
| `rvc` (default) | Needs a trained model | 48 kHz | Language independent by design | MIT |
| `meanvc2` | Any reference clip, no training | 16 kHz | Trained on Mandarin only, usable on the 8 languages tested | Apache-2.0 per its README; the repo has no licence file |
| `passthrough` | None | Same as input | Any | For testing |

## Setup

Requires Python 3.11, an NVIDIA GPU, and PipeWire or PulseAudio.

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python -e '.[dev]'
uv pip install --python .venv/bin/python --no-deps s3prl   # MeanVC 2 speaker model only

git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI third_party/rvc
git -C third_party/rvc checkout 81eed5e
git clone https://github.com/ASLP-lab/MeanVC2 third_party/meanvc2
git -C third_party/meanvc2 checkout 13acf84

.venv/bin/python scripts/fetch_models.py
```

## Usage

```sh
voxshift train lada data/raw/lada/dataset_lada_trimmed/accept --minutes 30 --epochs 150 \
    --source "https://doi.org/10.5281/zenodo.7396774" --license Apache-2.0
voxshift calibrate                                # measures your pitch to set the shift
voxshift run --voice lada --virtual-mic           # then pick "VoxShift Virtual Mic" in the other app
voxshift run --voice lada --monitor --record out.flac
voxshift convert --voice lada in.wav out.wav
voxshift gui
```

Use headphones with `--monitor`: a speaker feeding back into the microphone
makes an echo loop. Without `calibrate`, the pitch shift defaults to +12 semitones.

## Data

| Dataset | Used for | Licence |
|---|---|---|
| [Lada](https://doi.org/10.5281/zenodo.7396774) | Target voice: Ukrainian, female, 48 kHz, 10 h 37 m | Apache-2.0 |
| [LibriVox](https://librivox.org) clips | Test speech (`scripts/fetch_testset.py`) | Public domain |
| [Hi-Fi TTS](https://openslr.org/109) | Not used yet: English target voices, 41 GB | CC BY 4.0 |

MeanVC 2's fine-tuned WavLM speaker weights come from a Google Drive link in its
upstream repo, and no licence for them is stated. Check before redistributing them.

## Tests

```sh
.venv/bin/python -m pytest
.venv/bin/python scripts/measure_latency.py [--engine rvc --voice lada --speech data/test/uk.wav]
.venv/bin/python scripts/evaluate.py --voice lada --block-ms 40 --pitch auto --whisper-device cuda
```

`measure_latency.py` runs clicks and a tone through temporary virtual devices, so
it needs no microphone and plays nothing aloud. Passthrough results:

| Block | Median latency | Gaps in 8 s | Overruns |
|---|---|---|---|
| 20 ms | 102 ms | 5 | 0 |
| 40 ms | 86 ms | 1 | 0 |
| 100 ms | 153 ms | 2 | 0 |

These include the 40 ms jitter prebuffer, and exclude inference and any
physical device latency.

RVC with the short test voice on an RTX 3060, using speech because the model
erases clicks and tones. The block default is 40 ms, for its inference headroom.

| Block | Median latency, each run | Inference peak | Dropouts in 20 s |
|---|---|---|---|
| 100 ms | 289 ms | 68 ms | 0 |
| 60 ms | 218 ms | 38 ms | 0 |
| 40 ms | 246, 234, 221, 194 ms | 30-33 ms | 0 |
| 30 ms | 184, 219, 204, 192 ms | 27 ms | 0 |
| 20 ms | 172 ms | 23 ms | 0 |

Latency shifts by tens of milliseconds between runs. Conversion runs about six
times faster than realtime at 100 ms blocks.

`evaluate.py` over the 8 clips in `data/test/`, 30 s each. CER compares Whisper's
reading of the output with its reading of the source, so it is a relative check:

| Block | Pitch | Mean CER | Output F0 |
|---|---|---|---|
| 100 ms | matched to voice | 0.094 | 288-302 Hz |
| 30 ms | matched to voice | 0.115 | 288-301 Hz |
| 30 ms | +12 | 0.101 | 199-275 Hz |

MeanVC 2 with the same voice's reference clip, 80 ms blocks, no training:

| Median latency, each run | Inference | Dropouts in 20 s | Mean CER | Output F0 |
|---|---|---|---|---|
| 417, 455, 430 ms | Typically 42 ms offline, peaks 87-116 ms live | 37, 2, 2 | 0.163 | 210-239 Hz |

Its live peaks overrun the 80 ms block, which fills the output backlog. It has
no pitch control, so its output stays about 70 Hz below the target voice.

The target voice measures 302 Hz here, and 288 Hz by the pYIN method training
records. Matching it took +13.6 to +19.2 semitones.
