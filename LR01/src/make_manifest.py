"""Сборка MANIFEST.sha256 со всеми сдаваемыми файлами работы.

Исключаются виртуальное окружение, кэш весов, служебные файлы ОС и сам манифест.
Проверка: shasum -a 256 -c MANIFEST.sha256 (macOS) / sha256sum -c (Linux).
"""

from __future__ import annotations

from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".venv", "cache", "__pycache__", ".git"}
EXCLUDED_NAMES = {"MANIFEST.sha256", ".DS_Store", ".gitkeep"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if EXCLUDED_DIRS & set(relative.parts) or relative.name in EXCLUDED_NAMES:
            continue
        entries.append(f"{sha256_of(path)}  {relative.as_posix()}")

    (ROOT / "MANIFEST.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(f"Записано записей: {len(entries)}")


if __name__ == "__main__":
    main()
