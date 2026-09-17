"""Формальная валидация артефактов: PNG, JSON, длина и формат SHA-256.

Соответствует разделу «Проверка артефактов» методических указаний и обязательной
проверке варианта 11 («запустить и провалидировать артефакты»).
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "run_config.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def check(label: str, condition: bool, detail: str) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}: {detail}")
    return condition


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    ok = True

    for run_name in config["runs"]:
        run_dir = ROOT / "artifacts" / run_name
        png_path = run_dir / "result.png"
        manifest_path = run_dir / "manifest.json"
        print(f"\n=== {run_name} ===")

        ok &= check("PNG существует", png_path.is_file(), str(png_path.relative_to(ROOT)))
        with Image.open(png_path) as image:
            size, mode, fmt = image.size, image.mode, image.format
        ok &= check(
            "Размер изображения",
            size == (config["width"], config["height"]),
            f"{size[0]} x {size[1]}, ожидалось {config['width']} x {config['height']}",
        )
        ok &= check("Формат и цветовой режим", fmt == "PNG", f"format={fmt}, mode={mode}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ok &= check("manifest.json разбирается", True, "json.loads без исключения")

        digest = hashlib.sha256(png_path.read_bytes()).hexdigest()
        ok &= check("SHA-256 — 64 hex-символа", bool(HEX64.match(digest)), digest)
        ok &= check(
            "SHA-256 совпадает с манифестом",
            digest == manifest["sha256"],
            f"файл={digest[:16]}..., манифест={manifest['sha256'][:16]}...",
        )

        for field in (
            "model_id",
            "revision",
            "prompt",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "height",
            "width",
            "device",
            "dtype",
            "packages",
        ):
            ok &= check(
                f"Поле манифеста '{field}'",
                field in manifest and manifest[field] not in (None, "", {}),
                repr(manifest[field])[:80] if field in manifest else "ОТСУТСТВУЕТ",
            )

        ok &= check(
            "Параметры манифеста совпадают с конфигом",
            all(
                manifest[key] == config[key]
                for key in ("model_id", "revision", "prompt", "seed",
                            "num_inference_steps", "guidance_scale", "height", "width")
            ),
            "model_id/revision/prompt/seed/steps/guidance/size",
        )

    print(f"\nИТОГ: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
