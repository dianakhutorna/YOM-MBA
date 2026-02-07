**Overview**

- **Project:** Recommender training & inference pipeline (YOM-Recommender-System `training/`).
- **Purpose:** Build anchor→candidate ranking model (LightGBM lambdarank) from order baskets, produce scored candidate lists and a cold-start popularity fallback.

**Data Flow (high level)**

- **Raw → Interim:** raw CSVs (training/data/raw) → `preprocess_orders` → cleaned interim parquet (training/data/interim/orders_sample.parquet).

- **Split:** `split_orders_by_time` splits interim orders into train/val/test by configured ratios.

- **Baskets:** `build_baskets` converts orders → baskets with `products` lists per `kiosk_id`.

- **Candidates:** `generate_candidates` computes co-occurrence (cooc_count, support, lift, cosine_sim) from baskets; `select_top_k_candidates` keeps global top-K per anchor.

- **Feature table:** `build_feature_table` joins (kiosk, anchor) queries with global top-K candidates → rows (kiosk, anchor, candidate) and then `add_all_features` augments with product, kiosk, behavioral, personalization, popularity, and encoded categorical features depending on `FeatureConfig`.

- **Labeling:** `build_labels` uses future test/holdout orders to mark positives for (kiosk, anchor, candidate).

- **Train:** prepare group-wise ranking arrays, create LightGBM `lambdarank` datasets, train with early stopping, save booster and feature list (`<model>.features.json`).

- **Inference:** rebuild candidates and feature table (typically using train orders), ensure features match model feature list, predict scores, and save `predictions.parquet` plus a popularity fallback.

**Key Files (where to look)**
- **`training/src/pipelines/training.py`**: end-to-end training pipeline orchestration; reads config, runs preprocess→split→candidates→features→labels→train→eval and saves model + feature list.
- **`training/src/scripts/run_training_pipeline.py`**: CLI wrapper to call the pipeline (uses `training/configs/training_pipeline.yaml`).
- **`training/src/scripts/generate_predictions.py`**: inference script: rebuilds candidates/features, aligns columns to saved feature list, predicts with LightGBM, saves `predictions.parquet` and a popularity fallback.
- **`training/src/steps/`**: modular step implementations:
  - `preprocessing.py` — cleaning + normalization of raw orders
  - `split_orders.py` — time-based train/val/test splitting
  - `build_baskets.py` — baskets construction
  - `generate_candidates.py` — co-occurrence and MBA metrics
  - `select_top_k_candidates.py` — global top-K selection
  - `build_feature_table.py` — create (kiosk, anchor, candidate) rows
  - `add_*_features.py` — per-domain feature augmentation
  - `build_labels.py` — create labels from holdout orders
- **`training/src/features.py`**: orchestrator that conditionally applies feature groups via `FeatureConfig`.
- **`training/src/io/`**: I/O helpers (`load_orders_csv_sample`, `load_orders_parquet`, `load_products_csv`, `load_commerces_csv`, `save_parquet`).
- **`training/models/`**: saved model files; model name and feature list are saved together (e.g. `lgbm_ranker.txt` + `lgbm_ranker.features.json`).

**How training and inference relate (important details & pitfalls)**

- **Feature list persistence:** After training, the pipeline writes a JSON list of `feature_cols` next to the model (`<model>.features.json`). Inference loads this list (or LightGBM's `feature_name()`) and requires these columns.

- **Missing features at inference:** If a column from the saved feature list is absent, `generate_predictions.py` currently inserts a zero column and logs a warning. This can severely affect predictions if many features are missing or zero-only.

- **Zero-only features:** `generate_predictions.py` also logs features that are all-zero in inference data; many zero-only features usually indicate a mismatch in feature flags or data used for feature computation—consider retraining or reconciling configs.

- **Candidate top-K must match intent:** Training computes `top_k` (config `top_k` in training config) and trains on those candidates. Inference uses `top_k_candidates` in its config — if these differ you'll evaluate/predict on a different candidate set which will change recommendation outputs. Keep `top_k` aligned between training and inference.

- **Split ratios affect candidates:** Different `train/val/test` ratios change which orders are in `train_orders`, which changes baskets and generated candidates. Ensure `generate_predictions` uses the same `train_ratio` (or uses an explicit `train` dataset) if you want identical candidate generation.

- **FeatureConfig consistency:** Make sure the `features_config_path` used at training (often in `training/configs/features.yaml`) is the same as used at inference; otherwise some feature groups may be missing.

**Common commands**

- Run full training pipeline (uses `training/configs/training_pipeline.yaml`):
```bash
python -m training.src.scripts.run_training_pipeline --config training/configs/training_pipeline.yaml
```
- Run inference to produce `predictions.parquet` (uses `training/configs/generate_predictions.yaml`):
```bash
python -m training.src.scripts.generate_predictions --config training/configs/generate_predictions.yaml
```
- Run older playground training script (legacy):
```bash
python training/playground/scripts/train_ranker_lgbm.py
```

**Troubleshooting / Recommendations**

- If you see many **missing features** warnings in inference: compare `training/models/<model>.features.json` against the columns produced by `build_feature_table` + `add_all_features`. If mismatch is large — retrain or align feature flags.
- If you see many **zero-only** features: check `FeatureConfig` (`training/configs/features.yaml`) and ensure categorical encodings (channel/region) are applied identically in train and inference.
- Keep `top_k` (train) and `top_k_candidates` (inference) equal to avoid different candidate pools. Also align `min_cooc`/`min_lift` if you expect identical candidate graphs.
- Use the pipeline logs in `training/logs/` for training metrics and `generate_predictions.py` logs for inference warnings (missing features, zero-only features, score stats).

---

## На русском / In Russian

**Обзор**

- **Проект:** Пайплайн обучения и инференса рекомендателя (YOM-Recommender-System `training/`).
- **Цель:** Построить модель ранжирования якорь→кандидат (LightGBM lambdarank) из корзин заказов, создать списки кандидатов с оценками и fallback по популярности для холодного старта.

**Поток данных (высокий уровень)**

- **Raw → Interim:** raw CSV файлы (training/data/raw) → `preprocess_orders` → очищенный interim parquet (training/data/interim/orders_sample.parquet).

- **Split:** `split_orders_by_time` разбивает interim заказы на train/val/test в соответствии с заданными пропорциями.

- **Корзины:** `build_baskets` преобразует заказы → корзины со списками `products` для каждого `kiosk_id`.

- **Кандидаты:** `generate_candidates` вычисляет совместное появление (cooc_count, support, lift, cosine_sim) из корзин; `select_top_k_candidates` сохраняет глобальный top-K для каждого якоря.

- **Таблица признаков:** `build_feature_table` объединяет (kiosk, anchor) запросы с глобальными top-K кандидатами → строки (kiosk, anchor, candidate), затем `add_all_features` дополняет признаками продукта, киоска, поведением, персонализацией, популярностью и закодированными категориальными признаками в зависимости от `FeatureConfig`.

- **Разметка:** `build_labels` использует будущие тестовые/holdout заказы для разметки позитивов для (kiosk, anchor, candidate).

- **Обучение:** подготовка group-wise ранжирующих массивов, создание LightGBM `lambdarank` датасетов, обучение с ранней остановкой, сохранение модели и списка признаков (`<model>.features.json`).

- **Инференс:** пересчёт кандидатов и таблицы признаков (обычно используя train заказы), выравнивание признаков к списку признаков модели, предсказание оценок и сохранение `predictions.parquet` плюс fallback по популярности.

**Ключевые файлы (куда смотреть)**

- **`training/src/pipelines/training.py`**: сквозная оркестрация пайплайна обучения; читает конфиг, запускает preprocess→split→candidates→features→labels→train→eval и сохраняет модель + список признаков.
- **`training/src/scripts/run_training_pipeline.py`**: CLI-обёртка для вызова пайплайна (использует `training/configs/training_pipeline.yaml`).
- **`training/src/scripts/generate_predictions.py`**: скрипт инференса: пересчитывает кандидаты/признаки, выравнивает колонки с сохранённым списком признаков, предсказывает с помощью LightGBM, сохраняет `predictions.parquet` и fallback по популярности.
- **`training/src/steps/`**: модульная реализация этапов:
  - `preprocessing.py` — очистка + нормализация raw заказов
  - `split_orders.py` — временное разбиение на train/val/test
  - `build_baskets.py` — построение корзин
  - `generate_candidates.py` — совместное появление и метрики MBA
  - `select_top_k_candidates.py` — глобальный выбор top-K
  - `build_feature_table.py` — создание строк (kiosk, anchor, candidate)
  - `add_*_features.py` — добавление признаков по доменам
  - `build_labels.py` — создание меток из holdout заказов
- **`training/src/features.py`**: оркестратор условного применения групп признаков через `FeatureConfig`.
- **`training/src/io/`**: вспомогательные функции I/O (`load_orders_csv_sample`, `load_orders_parquet`, `load_products_csv`, `load_commerces_csv`, `save_parquet`).
- **`training/models/`**: сохранённые файлы моделей; название модели и список признаков сохраняются вместе (например `lgbm_ranker.txt` + `lgbm_ranker.features.json`).

**Связь обучения и инференса (важные детали и подводные камни)**

- **Сохранение списка признаков:** После обучения пайплайн записывает JSON-список `feature_cols` рядом с моделью (`<model>.features.json`). Инференс загружает этот список (или `feature_name()` от LightGBM) и требует наличие этих колонок.

- **Отсутствующие признаки при инференсе:** Если колонка из сохранённого списка признаков отсутствует, `generate_predictions.py` в настоящий момент вставляет нулевую колонку и логирует предупреждение. Это может серьёзно повлиять на предсказания, если отсутствуют или состоят только из нулей много признаков.

- **Признаки только из нулей:** `generate_predictions.py` также логирует признаки, которые состоят только из нулей в данных инференса; много нулевых признаков обычно указывают на несоответствие флагов признаков или данных, используемых для вычисления признаков — рассмотрите переобучение или согласование конфигов.

- **Кандидаты top-K должны совпадать по интенту:** Обучение вычисляет `top_k` (конфиг `top_k` в конфиге обучения) и обучается на этих кандидатах. Инференс использует `top_k_candidates` в своём конфиге — если они отличаются, вы будете оценивать/предсказывать на другом наборе кандидатов, что изменит выходные рекомендации. Держите `top_k` выровненным между обучением и инференсом.

- **Пропорции split влияют на кандидатов:** Разные пропорции `train/val/test` изменяют, какие заказы находятся в `train_orders`, что изменяет корзины и сгенерированные кандидаты. Убедитесь, что `generate_predictions` использует ту же `train_ratio` (или использует явный датасет `train`), если вы хотите идентичное генерирование кандидатов.

- **Консистентность FeatureConfig:** Убедитесь, что `features_config_path`, используемый при обучении (часто в `training/configs/features.yaml`), совпадает с используемым при инференсе; иначе некоторые группы признаков могут отсутствовать.

**Частые команды**

- Запустить полный пайплайн обучения (использует `training/configs/training_pipeline.yaml`):
```bash
python -m training.src.scripts.run_training_pipeline --config training/configs/training_pipeline.yaml
```

- Запустить инференс для создания `predictions.parquet` (использует `training/configs/generate_predictions.yaml`):
```bash
python -m training.src.scripts.generate_predictions --config training/configs/generate_predictions.yaml
```

- Запустить старый playground скрипт обучения (legacy):
```bash
python training/playground/scripts/train_ranker_lgbm.py
```

**Отладка / Рекомендации**

- Если вы видите много предупреждений о **missing features** при инференсе: сравните `training/models/<model>.features.json` со строками, созданными `build_feature_table` + `add_all_features`. Если расхождение велико — переобучите или выровняйте флаги признаков.

- Если вы видите много **zero-only признаков:** проверьте `FeatureConfig` (`training/configs/features.yaml`) и убедитесь, что категориальные кодирования (channel/region) применяются идентично при обучении и инференсе.

- Держите `top_k` (обучение) и `top_k_candidates` (инференс) равными, чтобы избежать разных пулов кандидатов. Также выровняйте `min_cooc`/`min_lift`, если вы хотите идентичные графы кандидатов.

- Используйте логи пайплайна в `training/logs/` для метрик обучения и логи `generate_predictions.py` для предупреждений инференса (missing features, zero-only features, score stats).

---




