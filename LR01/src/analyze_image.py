"""Объективный разбор полученного изображения.

Критерий варианта 11 — «метафора объяснена в отчёте». Чтобы разбор опирался на
измеримые свойства артефакта, а не только на впечатление, скрипт считает палитру,
контраст, симметрию и заполненность краёв кадра.

Все метрики детерминированы: квантование палитры выполняется методом медианного
сечения без случайной инициализации.

Использование:
    python src/analyze_image.py run_001
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# Коэффициенты яркости Rec. 709.
LUMA = np.array([0.2126, 0.7152, 0.0722])


def dominant_palette(image: Image.Image, colors: int = 6) -> list[dict]:
    quantized = image.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette = quantized.getpalette()[: colors * 3]
    # getcolors() возвращает пары (количество пикселей, индекс палитры).
    counts = {index: count for count, index in (quantized.getcolors(maxcolors=colors) or [])}
    total = sum(counts.values()) or 1
    entries = []
    for index in range(colors):
        r, g, b = palette[index * 3 : index * 3 + 3]
        share = counts.get(index, 0) / total
        entries.append(
            {
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "rgb": [r, g, b],
                "share": round(share, 4),
            }
        )
    return sorted(entries, key=lambda item: item["share"], reverse=True)


def edge_energy(gray: np.ndarray) -> np.ndarray:
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    return dy + dx


def main(run_name: str) -> int:
    png_path = ROOT / "artifacts" / run_name / "result.png"
    with Image.open(png_path) as opened:
        image = opened.convert("RGB")
        size = image.size

    array = np.asarray(image, dtype=np.float64) / 255.0
    gray = array @ LUMA
    height, width = gray.shape

    # Центральная зона — 50% кадра по каждой стороне; всё остальное считается полями.
    y0, y1 = height // 4, height - height // 4
    x0, x1 = width // 4, width - width // 4
    mask_center = np.zeros_like(gray, dtype=bool)
    mask_center[y0:y1, x0:x1] = True

    energy = edge_energy(gray)
    center_energy = float(energy[mask_center].mean())
    margin_energy = float(energy[~mask_center].mean())

    mirrored = gray[:, ::-1]
    symmetry_error = float(np.abs(gray - mirrored).mean())

    low, high = np.percentile(gray, [1, 99])
    michelson = float((high - low) / (high + low)) if (high + low) > 0 else 0.0

    report = {
        "run_name": run_name,
        "artifact": f"artifacts/{run_name}/result.png",
        "image_size": list(size),
        "dominant_palette": dominant_palette(image),
        "mean_rgb": [round(float(v), 4) for v in array.reshape(-1, 3).mean(axis=0)],
        "luminance": {
            "mean": round(float(gray.mean()), 4),
            "std": round(float(gray.std()), 4),
            "p01": round(float(low), 4),
            "p99": round(float(high), 4),
            "michelson_contrast": round(michelson, 4),
        },
        "composition": {
            "center_edge_energy": round(center_energy, 5),
            "margin_edge_energy": round(margin_energy, 5),
            "center_to_margin_ratio": round(center_energy / margin_energy, 3)
            if margin_energy > 0
            else None,
            "comment": "Отношение > 1 означает, что детализация сосредоточена в центре, а поля относительно пусты.",
        },
        "symmetry": {
            "horizontal_mirror_mean_abs_error": round(symmetry_error, 5),
            "comment": "0 — идеальная зеркальная симметрия по вертикальной оси; чем меньше, тем симметричнее композиция.",
        },
    }

    out_path = ROOT / "reports" / f"image_analysis_{run_name}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "run_001"))
