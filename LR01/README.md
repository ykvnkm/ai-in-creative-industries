# ЛР №1 — Запуск открытой модели text-to-image

**Вариант 11:** иллюстрация для материала о кибербезопасности.
Изменяемый фактор — метафора защиты без интерфейсных клише.

**Отчёт:** [REPORT.md](REPORT.md) · **Разбор prompt:** [data/prompt.md](data/prompt.md)

| Параметр | Значение |
| --- | --- |
| Модель | `stabilityai/sd-turbo` |
| Ревизия | `b261bac6fd2cf515557d5d0707481eafa0485ec2` |
| Площадка | локальный инференс, Apple M4, CPU, `float32` |
| Параметры | seed `20260911`, steps `4`, guidance `0.0`, 512 × 512 |
| Критерий приёмки | `reports/sha256_comparison.json` содержит `"exact_sha256_match": true` |

## Как повторить

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision
.venv/bin/python -m pip install "diffusers==0.40.0" "transformers>=5,<6" \
    "accelerate>=1.2,<2" "safetensors>=0.5,<1" "Pillow>=11,<13"
.venv/bin/python -m pip freeze > reports/environment.txt

export HF_HOME="$PWD/cache/huggingface"
.venv/bin/python src/run_reproducibility.py    # два запуска + сравнение SHA-256
.venv/bin/python src/validate_artifacts.py     # формальная валидация PNG и JSON
```

Зависимости отличаются от методических указаний одним пунктом: `transformers>=5`
вместо `transformers>=4.51,<5`. Указанная в методичке комбинация неразрешима,
подтверждение — `reports/dependency_conflict.log`, разбор — в отчёте, раздел 6.

## Что где лежит

| Путь | Содержимое |
| --- | --- |
| [REPORT.md](REPORT.md) | отчёт по обязательной структуре из методических указаний |
| [assignment/](assignment) | методические указания (PDF) |
| [data/prompt.md](data/prompt.md) | происхождение prompt, четыре элемента, проверка безопасности |
| [configs/run_config.json](configs/run_config.json) | все параметры запуска одним файлом |
| [src/](src) | код генерации, повтора, валидации и воспроизведения ошибки |
| [artifacts/](artifacts) | `run_001`, `run_002`: `result.png` + `manifest.json` |
| [reports/](reports) | среда, журналы запусков, сравнение SHA-256, журнал ошибки |
| [appendix_a_demo_linux/](appendix_a_demo_linux) | Приложение А: прогон демо-примера методички в Linux-среде |
| `MANIFEST.sha256` | контрольные суммы всех сдаваемых файлов |

## Проверка целостности

```bash
shasum -a 256 -c MANIFEST.sha256
```

## Что не входит в репозиторий

Каталог `.venv/` и кэш весов `cache/huggingface/` (~5 ГБ) исключены через
`.gitignore`, как требуют методические указания. Веса восстанавливаются по
`model_id` и `revision` из конфига при первом запуске.
