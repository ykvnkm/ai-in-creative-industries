# ЛР №1. Запуск открытой модели text-to-image

**Вариант 11** — иллюстрация для материала о кибербезопасности.
Изменяемый фактор: метафора защиты без интерфейсных клише.

## 📓 [LR01.ipynb](LR01.ipynb) — работа целиком

Ноутбук содержит код, фактические выходы его выполнения и отчёт. GitHub
рендерит его прямо в браузере: видны команда, её результат и изображение подряд.

| | |
| --- | --- |
| Модель | `stabilityai/sd-turbo`, ревизия `b261bac6fd2cf515557d5d0707481eafa0485ec2` |
| Параметры | seed `20260911`, steps `4`, guidance `0.0`, 512 × 512 |
| Среда сохранённого прогона | Apple M4, CPU, `float32`, 1 поток |
| Результат | два запуска, SHA-256 совпали побитово |

## Как повторить

**Вариант А — Google Colab** (ничего не надо ставить):
открыть [colab.research.google.com](https://colab.research.google.com) →
File → Open notebook → GitHub → вставить ссылку на этот репозиторий → выбрать
`LR01/LR01.ipynb` → Runtime → Run all.

**Вариант Б — локально:**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision jupyter
.venv/bin/jupyter notebook LR01.ipynb
```

Первый запуск скачивает веса модели (~2,5 ГБ в fp16 на GPU, ~5 ГБ в fp32 на CPU).

## Что где лежит

| Путь | Содержимое |
| --- | --- |
| [LR01.ipynb](LR01.ipynb) | код, выходы, отчёт |
| [artifacts/](artifacts) | `run_001`, `run_002`: `result.png` + `manifest.json` |
| [reports/](reports) | `environment.txt`, `sha256_comparison.json` |
| [configs/](configs) | `run_config.json` — параметры одним файлом |
| [assignment/](assignment) | методические указания |

Всё в `artifacts/`, `reports/` и `configs/` создаётся самим ноутбуком при
выполнении. Кэш весов и виртуальное окружение в репозиторий не попадают.
