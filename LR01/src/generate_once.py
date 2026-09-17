"""ЛР №1, вариант 11. Один запуск text-to-image с полной фиксацией условий.

Скрипт выполняет ровно один запуск: загружает закреплённую ревизию модели,
генерирует изображение и сохраняет PNG вместе с манифестом условий.
Повторный запуск выполняется отдельным процессом (src/run_reproducibility.py),
поэтому совпадение SHA-256 проверяет полный цикл «загрузка -> генерация»,
а не повторный вызов уже прогретого pipeline.

Использование:
    python src/generate_once.py run_001
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import os
import platform
import sys
import time

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "run_config.json"

PACKAGES = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "huggingface-hub",
    "Pillow",
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(run_name: str) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    out_dir = ROOT / "artifacts" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Детерминизм фиксируется явно: число потоков и запрет недетерминированных ядер.
    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.use_deterministic_algorithms(bool(config["deterministic_algorithms"]))

    # Устройство выбирается только после проверки доступности.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    load_args = {
        "revision": config["revision"],
        "use_safetensors": True,
        "torch_dtype": dtype,
    }
    if device == "cuda":
        load_args["variant"] = "fp16"

    load_started = time.perf_counter()
    pipe = AutoPipelineForText2Image.from_pretrained(config["model_id"], **load_args)
    pipe = pipe.to(device)
    load_seconds = time.perf_counter() - load_started

    # CPU Generator сопоставим между устройствами, см. документацию Diffusers.
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))

    gen_started = time.perf_counter()
    image = pipe(
        prompt=config["prompt"],
        num_inference_steps=int(config["num_inference_steps"]),
        guidance_scale=float(config["guidance_scale"]),
        height=int(config["height"]),
        width=int(config["width"]),
        generator=generator,
    ).images[0]
    gen_seconds = time.perf_counter() - gen_started

    image_path = out_dir / "result.png"
    image.save(image_path)

    with Image.open(image_path) as saved:
        image_size_observed = list(saved.size)
        image_mode_observed = saved.mode

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant": config["variant"],
        "variant_context": config["variant_context"],
        "variant_factor": config["variant_factor"],
        "run_name": run_name,
        "model_id": config["model_id"],
        "revision": config["revision"],
        "prompt": config["prompt"],
        "seed": config["seed"],
        "num_inference_steps": config["num_inference_steps"],
        "guidance_scale": config["guidance_scale"],
        "height": config["height"],
        "width": config["width"],
        "device": device,
        "dtype": str(dtype),
        "deterministic_algorithms": bool(config["deterministic_algorithms"]),
        "torch_num_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {name: md.version(name) for name in PACKAGES},
        "hf_home": os.environ.get("HF_HOME", ""),
        "model_load_seconds_measured": load_seconds,
        "generation_seconds_measured": gen_seconds,
        "artifact": f"artifacts/{run_name}/result.png",
        "artifact_size_bytes": image_path.stat().st_size,
        "image_size_observed": image_size_observed,
        "image_mode_observed": image_mode_observed,
        "sha256": sha256_of(image_path),
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run_001")
