# Структура репозитория okx-hft-research

## Обзор

Research-репозиторий организован по принципу: данные → фичи → сигналы → валидация → кандидат в production.

## Основные каталоги

| Каталог | Назначение |
|---------|------------|
| `configs/` | YAML-конфигурации (БД, инструменты, настройки research) |
| `research/` | Python-пакет: db, features, labels, signals, backtests, visualization, validation |
| `apps/` | Streamlit-приложение и минимальный API |
| `notebooks/` | Jupyter-ноутбуки для разведки и экспериментов |
| `experiments/` | Структурированные эксперименты с гипотезами |
| `scripts/` | CLI-скрипты для export, feature build, signal eval |

## Импорты

Код импортирует модули из `research`:

```python
from research.db.loaders import load_orderbook_snapshot
from research.features.spread import compute_spread
```

Запуск из корня проекта (где лежит pyproject.toml) с установленным пакетом (`pip install -e .`).
