# OKX HFT Research

Исследовательский репозиторий для pet-проекта по HFT на данных биржи OKX. Здесь живут исследования рыночной микроструктуры, визуализации стакана, проверка гипотез и прототипирование фичей и сигналов перед переносом в production.

---

## Место в архитектуре проекта

Research-слой — это **sandbox между данными и production**. Raw/core-данные собираются в `okx-hft-collector` и хранятся в TimescaleDB; research-репозиторий читает их, исследует и готовит кандидатов для переноса в боевые системы.

### Связанные репозитории

| Репозиторий | Роль |
|-------------|------|
| **okx-hft-collector** | Сбор raw/core данных с OKX (WebSocket, REST) |
| **okx-hft-timescaledb** | Хранение данных в TimescaleDB |
| **okx-hft-executor** | Execution layer (ордера, стратегии) |
| **okx-hft-ops** | Инфраструктура, мониторинг |
| **okx-hft-research** | Исследования, визуализация, сигналы (этот репо) |

### Зачем нужен research-слой

- **Безопасность**: эксперименты и визуализации не задевают production.
- **Скорость итераций**: Jupyter, Streamlit, скрипты — всё для быстрой проверки гипотез.
- **Единый стек**: Python, pandas, Plotly — понятная среда для quant-исследований.
- **Чистый путь в production**: только валидированные фичи и сигналы попадают в executor/airflow-dags.

---

## Цели репозитория

1. **Исследование микроструктуры** — стакан, спред, дисбаланс, микроцена.
2. **Визуализация** — heatmap стакана, replay по времени, кривые глубины.
3. **Проверка гипотез** — формализация в docs/hypotheses, эксперименты в experiments/.
4. **Прототипирование фичей** — spread, imbalance, microprice и др.
5. **Сигналы** — простые сигналы на imbalance, microprice и т.д.
6. **Диагностика** — проверка качества фичей, корреляции, стабильности.

---

## Что должно жить внутри

| Тип | Где |
|-----|-----|
| Исследования | `notebooks/`, `experiments/` |
| Визуализации | `research/visualization/`, Streamlit-страницы |
| Ноутбуки | `notebooks/exploratory/`, `notebooks/signals/`, `notebooks/models/` |
| Гипотезы | `docs/hypotheses/` |
| Фичи | `research/features/` |
| Сигналы | `research/signals/` |
| Диагностика | `research/validation/` |

---

## Структура каталогов

```
okx-hft-research/
├── configs/           # YAML-конфигурации (app, db, instruments, research)
├── docs/              # Документация (архитектура, гипотезы, визуализации)
├── notebooks/         # Jupyter-ноутбуки (exploratory, signals, models)
├── research/          # Основной Python-пакет
│   ├── db/            # Подключение к БД, загрузка данных, SQL-запросы
│   ├── features/      # Фичи (spread, imbalance, microprice)
│   ├── labels/        # Лейблы (future mid move)
│   ├── signals/       # Сигналы (imbalance, microprice)
│   ├── backtests/     # Метрики и бэктесты
│   ├── visualization/ # Heatmap, replay, spread charts, depth curves
│   ├── validation/    # Диагностика фичей
│   └── utils/         # Логирование, время
├── apps/
│   ├── streamlit/     # Streamlit-приложение (heatmap, replay, imbalance)
│   └── api/           # Минимальный FastAPI (health)
├── experiments/       # Эксперименты с гипотезами
├── scripts/           # CLI-скрипты (export, feature build, signal eval)
├── tests/             # Pytest-тесты
└── data_samples/      # Примеры выгруженных данных
```

---

## Workflow: от данных до production-кандидата

```
raw/core data → research → hypothesis → feature → signal → validation → production candidate
```

1. Данные подтягиваются из TimescaleDB через `research/db/`.
2. Гипотеза оформляется в `docs/hypotheses/` и эксперименте в `experiments/`.
3. Фича реализуется в `research/features/`, проверяется в ноутбуке.
4. Сигнал строится в `research/signals/`.
5. Валидация — `research/validation/`, метрики в `research/backtests/`.
6. Кандидат готов — документируется и выносится в executor/airflow-dags.

---

## MVP первого этапа

- Подключение к TimescaleDB и загрузка orderbook/trades.
- Визуализация: heatmap стакана, replay, spread.
- Базовые фичи: spread, imbalance, microprice.
- Простой imbalance-сигнал.
- Streamlit-приложение с тремя страницами: heatmap, replay, imbalance.
- Документация гипотез и шаблон эксперимента.

---

## Первые исследовательские направления

1. **Order book heatmap** — визуализация стакана в виде тепловой карты.
2. **Order book replay** — анимация стакана во времени.
3. **Spread dynamics** — динамика bid-ask spread.
4. **Imbalance** — дисбаланс bid/ask по уровням (L1, L10).
5. **Microprice** — взвешенная цена между bid и ask.

---

## Как запускать проект

```bash
cd okx-hft-research

# Вариант 1: make setup (создаёт .venv и ставит пакет)
make setup
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS

# Вариант 2: вручную
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# Запуск Jupyter (из корня проекта)
make notebook
# или: jupyter notebook notebooks/
```

**Важно:** Запускайте Jupyter из корня проекта (`cd okx-hft-research`), чтобы ноутбуки видели `research`. Либо в каждом ноутбуке есть bootstrap-ячейка, добавляющая корень в `sys.path`.

---

## Как запустить Streamlit

```bash
make streamlit
# или
streamlit run apps/streamlit/Home.py --server.port 8501
```

Откроется браузер с главной страницей; в боковой панели доступны страницы heatmap, replay, imbalance.

---

## Как оформлять гипотезы и эксперименты

1. Создайте запись в `docs/hypotheses/backlog.md` или новый файл по `hypothesis_template.md`.
2. Создайте каталог `experiments/EXP-NNN-<название>/` с `README.md`, `config.yaml`, `results.md`.
3. Используйте ноутбуки в `notebooks/` для разведки.
4. Фиксируйте результаты в `results.md`.

---

## Что НЕ должно попадать в production без проверки

- Сырые исследовательские скрипты без тестов.
- Фичи без валидации (стабильность, корреляции).
- Сигналы без бэктеста и метрик.
- Код без docstring и типизации.
- Конфиги с хардкодом вместо env-переменных.

Перед переносом в production: тесты, документация, code review.

---

## Планы развития

- [ ] Интеграция с okx-hft-timescaledb (схемы, таблицы).
- [ ] Расширение каталога визуализаций.
- [ ] Бэктест-фреймворк для сигналов.
- [ ] Пайплайн: feature build → signal eval → report.
- [ ] CI (lint, test, typecheck).
- [ ] Dashboard для мониторинга качества фичей.
