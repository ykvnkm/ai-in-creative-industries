"""Повтор запуска в тех же условиях и сравнение контрольных сумм.

Каждый прогон выполняется отдельным процессом, чтобы проверялся полный цикл
«загрузка pipeline -> генерация», а не повторный вызов уже прогретого объекта.
Результат сравнения сохраняется в reports/sha256_comparison.json.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "run_config.json"
REPORTS = ROOT / "reports"


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    REPORTS.mkdir(parents=True, exist_ok=True)

    for run_name in config["runs"]:
        log_path = REPORTS / f"{run_name}.log"
        header = (
            f"# Команда: python src/generate_once.py {run_name}\n"
            f"# Начало (UTC): {datetime.now(timezone.utc).isoformat()}\n---\n"
        )
        completed = subprocess.run(
            [sys.executable, str(ROOT / "src" / "generate_once.py"), run_name],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        log_path.write_text(
            header + completed.stdout + completed.stderr + f"\nEXIT: {completed.returncode}\n",
            encoding="utf-8",
        )
        print(f"{run_name}: exit={completed.returncode}, log={log_path.name}")
        if completed.returncode != 0:
            return completed.returncode

    manifests = {
        run_name: json.loads(
            (ROOT / "artifacts" / run_name / "manifest.json").read_text(encoding="utf-8")
        )
        for run_name in config["runs"]
    }
    digests = {name: m["sha256"] for name, m in manifests.items()}
    sizes = {name: tuple(m["image_size_observed"]) for name, m in manifests.items()}
    modes = {name: m["image_mode_observed"] for name, m in manifests.items()}
    exact_match = len(set(digests.values())) == 1

    comparison = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "variant": config["variant"],
        "runs": config["runs"],
        "sha256": digests,
        "image_size_observed": {k: list(v) for k, v in sizes.items()},
        "image_mode_observed": modes,
        "generation_seconds_measured": {
            name: m["generation_seconds_measured"] for name, m in manifests.items()
        },
        "exact_sha256_match": exact_match,
        "same_image_size": len(set(sizes.values())) == 1,
        "same_image_mode": len(set(modes.values())) == 1,
        "interpretation": (
            "PASS: запуски в одной зафиксированной среде совпали побитово."
            if exact_match
            else "FAIL: контрольные суммы разошлись, требуется диагностика среды."
        ),
    }
    (REPORTS / "sha256_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0 if exact_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
