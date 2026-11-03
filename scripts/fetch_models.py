#!/usr/bin/env python3
"""Download the pretrained weights the engines need into third_party/.

Idempotent: anything already on disk is skipped. Run from the project venv:

    .venv/bin/python scripts/fetch_models.py [--rvc] [--rvc-train] [--meanvc2]

With no flags everything is fetched.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
RVC = ROOT / "third_party" / "rvc"
MEANVC2 = ROOT / "third_party" / "meanvc2"

RVC_REPO = "lj1995/VoiceConversionWebUI"
MEANVC2_REPO = "ASLP-lab/MeanVC2"

WAVLM_FINETUNE_GDRIVE_ID = "1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP"
WAVLM_LARGE_CFG = {
    "extractor_mode": "layer_norm",
    "encoder_layers": 24,
    "encoder_embed_dim": 1024,
    "encoder_ffn_embed_dim": 4096,
    "encoder_attention_heads": 16,
    "layer_norm_first": True,
    "conv_bias": False,
    "normalize": True,
    "relative_position_embedding": True,
    "gru_rel_pos": True,
}


def hf(repo: str, name: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {repo}:{name}")
    cached = hf_hub_download(repo, name)
    shutil.copyfile(Path(cached).resolve(), dest)


def fetch_rvc() -> None:
    for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        hf(RVC_REPO, f"hubert_base/{name}", RVC / "assets/hubert_base" / name)
    hf(RVC_REPO, "rmvpe.pt", RVC / "assets/rmvpe/rmvpe.pt")


def fetch_rvc_train() -> None:
    for name in ("f0G48k.pth", "f0D48k.pth", "f0G40k.pth", "f0D40k.pth"):
        hf(RVC_REPO, f"pretrained_v2/{name}", RVC / "assets/pretrained_v2" / name)
    if not (RVC / "logs/mute").exists():
        print(f"[get ] {RVC_REPO}:mute.zip")
        shutil.unpack_archive(hf_hub_download(RVC_REPO, "mute.zip"), RVC / "logs")


def write_wavlm_config(dest: Path) -> None:
    """Write the WavLM-Large hyperparameters MeanVC2's speaker model builds from.

    Upstream extracts these from WavLM-Large.pt, whose download URL no longer
    resolves. Only the architecture is needed: every weight comes from
    wavlm_large_finetune.pth, and a wrong value here fails its state-dict load.
    """
    import torch

    torch.save(WAVLM_LARGE_CFG, dest)


def fetch_meanvc2() -> None:
    hf(MEANVC2_REPO, "meanvc2_120ms_40ms.safetensors",
       MEANVC2 / "ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors")
    hf(MEANVC2_REPO, "meanvc2_40ms_40ms.safetensors",
       MEANVC2 / "ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors")
    hf(MEANVC2_REPO, "vocos.pt", MEANVC2 / "ckpts/vocos/vocos.pt")
    hf(MEANVC2_REPO, "fastu2pp_80ms.pt", MEANVC2 / "preprocess/ckpts/fastu2pp_80ms.pt")
    hf(MEANVC2_REPO, "fastu2pp_160ms.pt", MEANVC2 / "preprocess/ckpts/fastu2pp_160ms.pt")

    write_wavlm_config(MEANVC2 / "preprocess/ckpts/wavlm_large_cfg.pt")

    finetune = MEANVC2 / "preprocess/ckpts/wavlm_large_finetune.pth"
    if not finetune.exists():
        import gdown

        print("[get ] wavlm_large_finetune.pth (Google Drive)")
        gdown.download(id=WAVLM_FINETUNE_GDRIVE_ID, output=str(finetune), quiet=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rvc", action="store_true", help="RVC inference weights")
    p.add_argument("--rvc-train", action="store_true", help="RVC pretrained bases for training")
    p.add_argument("--meanvc2", action="store_true", help="MeanVC2 weights")
    a = p.parse_args()
    everything = not (a.rvc or a.rvc_train or a.meanvc2)
    if a.rvc or everything:
        fetch_rvc()
    if a.rvc_train or everything:
        fetch_rvc_train()
    if a.meanvc2 or everything:
        fetch_meanvc2()


if __name__ == "__main__":
    main()
