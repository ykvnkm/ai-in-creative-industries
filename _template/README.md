# ЛР №NN — <тема>

Вариант: <номер> — <контекст варианта>.

## Как повторить запуск

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install <зависимости>
.venv/bin/python -m pip freeze > reports/environment.txt
.venv/bin/python src/<скрипт>.py
```

## Что где лежит

| Путь | Содержимое |
| --- | --- |
| `REPORT.md` | отчёт по работе |
| `assignment/` | методические указания |
| `data/` | описание входных данных |
| `configs/` | параметры запуска |
| `src/` | исходный код |
| `artifacts/` | результаты запусков и манифесты |
| `reports/` | среда, журналы, сравнения |
| `MANIFEST.sha256` | контрольные суммы сдаваемых файлов |

## Критерий приёмки

<что должно совпасть/выполниться, чтобы работа считалась воспроизводимой>
