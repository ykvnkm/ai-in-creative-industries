# ЛР №2. Карточка модели, лицензии, происхождение данных и анализ рисков

**Вариант 2** · статус: **выполнена**

**Отчёт:** [Яковенко_Максим_Михайлович_Вариант_2_ЛР_2.docx](Яковенко_Максим_Михайлович_Вариант_2_ЛР_2.docx) ·
**журнал:** [reports/journal.md](reports/journal.md)

| | |
| --- | --- |
| Сценарий | афиша фестиваля (печать, продажа билетов) |
| Профиль дефектов | материалы под лицензией с условием ShareAlike |
| Политика | коммерческое — да, публикация — да, производные — да, ShareAlike — **нет** |
| Обязательные группы рисков | Intellectual Property, Information Integrity |

## Результаты

| | |
| --- | --- |
| Аудит | **0 ошибок, 5 предупреждений** (4 × R5, 1 × R8) |
| Ожидания до запуска | сбылись полностью, зафиксированы коммитом `182f557` |
| Решения | приняты по всем пяти находкам: 4 исключения с заменой, 1 карантин |
| Паспорт набора | 7 разделов, пустых нет |
| Карточка модели | 9 разделов; выявила 2 пробела — лицензия кода и ответственный |
| Реестр рисков | 7 рисков, 6 категорий, `RESULT: PASS` |

Главное содержательное обстоятельство варианта: **ошибок нет, но набор непригоден
без решений**. Методические указания прямо предупреждают, что отсутствие ошибок при
наличии предупреждений — не зелёный свет, а «не рассмотрены находки-предупреждения»
входит в список критических ошибок.

## Как повторить

```bash
python3 src/lab02_make_variant.py --variant 2 --out artifacts/v02
python3 src/lab02_audit.py --manifest artifacts/v02/manifest.csv \
    --policy artifacts/v02/policy.json --out artifacts/v02
python3 src/lab02_model_card.py --passport ../LR01/artifacts/v02/passport.json \
    --out artifacts/model_card.md --code-license MIT --owner "Яковенко Максим Михайлович"
python3 src/lab02_risks.py --csv artifacts/risk_register.csv --out artifacts/risk_ranking.md
```

Сторонние пакеты не нужны — только стандартная библиотека Python 3.10+.

## Что где лежит

| Путь | Содержимое |
| --- | --- |
| [artifacts/data_card.md](artifacts/data_card.md) | паспорт набора данных, 7 разделов |
| [artifacts/model_card.md](artifacts/model_card.md) | карточка модели, 9 разделов |
| [artifacts/model_card_before_decisions.md](artifacts/model_card_before_decisions.md) | карточка до решений — свидетельство выявленных пробелов |
| [artifacts/risk_register.csv](artifacts/risk_register.csv) · [risk_ranking.md](artifacts/risk_ranking.md) | реестр рисков и ранжирование |
| [artifacts/risk_register_broken.csv](artifacts/risk_register_broken.csv) | реестр с намеренными дефектами |
| [artifacts/v02/](artifacts/v02) | данные варианта и отчёт аудита |
| [reports/journal.md](reports/journal.md) | журнал: ожидания до запуска, ход работы, решения |
| [reports/error_demo.log](reports/error_demo.log) | журнал намеренной ошибки реестра |
| [assignment/](assignment) | методические указания, варианты, демонстрационный журнал |
| [src/](src) | пять скриптов, выданы преподавателем |

**Оговорка:** работа учебная, отчёт не является юридической консультацией.
