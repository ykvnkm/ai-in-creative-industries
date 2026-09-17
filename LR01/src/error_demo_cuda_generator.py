"""Намеренно введённая типовая ошибка (методические указания, стр. 4-5).

Безопасная копия рабочего кода. Единственное отличие от src/generate_once.py:

    - было:  generator = torch.Generator(device="cpu").manual_seed(SEED)
    - стало: generator = torch.Generator(device="cuda").manual_seed(SEED)

Скрипт запускается на CPU-машине, воспроизводит симптом, печатает диагностику и
удаляет неполный каталог запуска, как требуют методические указания.
Эталонный src/generate_once.py этим скриптом не изменяется.

Использование:
    python src/error_demo_cuda_generator.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import traceback

import torch
from diffusers import AutoPipelineForText2Image

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "run_config.json"
BROKEN_RUN_DIR = ROOT / "artifacts" / "error_demo_incomplete"


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    torch.set_num_threads(int(config["torch_num_threads"]))

    print(f"# Воспроизведение намеренной ошибки, UTC: {datetime.now(timezone.utc).isoformat()}")
    print("# Изменение: torch.Generator(device='cuda') вместо device='cpu'")
    print("---")

    # Каталог запуска создаётся до генерации — он и останется неполным.
    BROKEN_RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Создан каталог запуска: artifacts/{BROKEN_RUN_DIR.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        use_safetensors=True,
        torch_dtype=dtype,
    ).to(device)
    print(f"Pipeline загружен на устройство: {device}")

    try:
        # НАМЕРЕННАЯ ОШИБКА: устройство генератора задано жёстко, без проверки.
        generator = torch.Generator(device="cuda").manual_seed(int(config["seed"]))
        image = pipe(
            prompt=config["prompt"],
            num_inference_steps=int(config["num_inference_steps"]),
            guidance_scale=float(config["guidance_scale"]),
            height=int(config["height"]),
            width=int(config["width"]),
            generator=generator,
        ).images[0]
        image.save(BROKEN_RUN_DIR / "result.png")
    except Exception as exc:
        print("\n=== СИМПТОМ ===")
        print(f"{type(exc).__name__}: {exc}")
        print("\n=== ПОЛНАЯ ТРАССИРОВКА ===")
        traceback.print_exc()
        print("\n=== ДИАГНОСТИКА ===")
        print(f"torch.__version__              = {torch.__version__}")
        print(f"torch.cuda.is_available()      = {torch.cuda.is_available()}")
        print(f"torch.backends.mps.is_available() = {torch.backends.mps.is_available()}")
        print(f"устройство pipeline            = {device}")
        print(f"устройство генератора в коде    = 'cuda' (жёстко задано)")
        print("Вывод: несовместимость устройства генератора и фактической среды.")
        print("\n=== ИСПРАВЛЕНИЕ ===")
        print("Создать CPU Generator, как в эталонном src/generate_once.py:")
        print("    generator = torch.Generator(device='cpu').manual_seed(SEED)")
        print("либо выбирать устройство только после torch.cuda.is_available().")
        print("\n=== ОЧИСТКА ===")
        shutil.rmtree(BROKEN_RUN_DIR, ignore_errors=True)
        print(f"Неполный каталог artifacts/{BROKEN_RUN_DIR.name} удалён.")
        return 1

    print("НЕОЖИДАННО: ошибка не воспроизвелась, среда содержит CUDA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
