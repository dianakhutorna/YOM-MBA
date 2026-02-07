Три этапа: Обучение → Инференс → Сервинг
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                           │
│                    training.py (ОДИН РАЗ)                       │
├─────────────────────────────────────────────────────────────────┤
│ 1. Загрузить raw CSV                                            │
│ 2. Preprocess → interim orders.parquet                          │
│ 3. Split на train/val/test (by time)                            │
│ 4. Build baskets из train_orders                                │
│ 5. Generate candidates (MBA: cooc, lift, cosine_sim)            │
│ 6. Build feature table (join kiosk×anchor + candidates)         │
│ 7. Add features (product, behavioral, categorical, etc.)        │
│ 8. Build labels (из test_orders)                                │
│ 9. Train LightGBM lambdarank с eval на val/test                 │
│ 10. Сохранить модель + feature list                             │
│                                                                  │
│ OUTPUT:                                                          │
│  - lgbm_ranker.txt (модель)                                      │
│  - lgbm_ranker.features.json (список колонок для инференса)      │
│  - orders_sample.parquet (очищенные заказы)                      │
│  - logs/ (метрики обучения)                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               INFERENCE / BATCH SCORING                          │
│            generate_predictions.py (ЧАСТО)                       │
├─────────────────────────────────────────────────────────────────┤
│ 1. Загрузить сохранённую модель (lgbm_ranker.txt)               │
│ 2. Загрузить сохранённый feature list (features.json)           │
│ 3. Загрузить актуальные заказы (из БД или interim)              │
│ 4. Build baskets (из этих заказов)                              │
│ 5. Generate candidates (те же MBA метрики)                      │
│ 6. Build feature table                                          │
│ 7. Add features (с ТЕМИ ЖЕ флагами что и при обучении!)         │
│ 8. Выравнять колонки:                                           │
│    - Убрать лишние признаки                                     │
│    - Добавить missing признаки (нули) ⚠️ ОСТОРОЖНО!             │
│ 9. Predict scores (LightGBM)                                    │
│ 10. Сохранить predictions.parquet                               │
│ 11. Build popularity fallback (для холодного старта)            │
│                                                                  │
│ OUTPUT:                                                          │
│  - predictions.parquet (kiosk, anchor, candidate, score)         │
│  - popularity_fallback.parquet (топ товары по частоте)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SERVING (в production)                        │
│              serve_bundle.py (ПОСТОЯННО онлайн)                 │
├─────────────────────────────────────────────────────────────────┤
│ 1. Загрузить predictions.parquet в память (индекс по kiosk_id)  │
│ 2. На каждый запрос:                                            │
│    - Get kiosk_id + anchor_product_id (из контекста)            │
│    - Lookup в predictions → top-K candidates + scores           │
│    - Если нет в predictions (холодный старт):                   │
│      * Fallback к popularity_fallback.parquet                   │
│    - Вернуть JSON с рекомендациями                              │
│                                                                  │
│ OUTPUT:                                                          │
│  - JSON API ответ (для каждого запроса <1ms)                    │
└─────────────────────────────────────────────────────────────────┘

Что тренировать один раз? Что часто?

training.py	🔴 Редко (еженедельно/ежемесячно)	
Затратно: preprocess + candidates + features + train; нужны новые данные
generate_predictions.py	🟡 Часто (ежедневно/еженедельно)	
Быстро: используем готовую модель, только переоценяем кандидаты с актуальными данными
serve_bundle.py	🟢 Постоянно (в prod 24/7)	Просто lookup в памяти, <1ms на запрос; обновляется когда generate_predictions создаст новый parquet

Ключевые различия
training.py (Один раз)
🔧 Строит кандидатов с нуля из train_orders
📚 Обучает модель на исторических данных
💾 Сохраняет модель + feature list
⏱️ Занимает часы (в зависимости от объёма данных)

# Редко, когда нужны новые фичи или переобучение
python -m training.src.scripts.run_training_pipeline \
  --config training/configs/training_pipeline.yaml

generate_predictions.py (Часто)
🔄 Переиспользует модель из training.py
🎯 Переоценивает кандидатов на новых/актуальных данных
📊 Генерирует predictions.parquet для сервинга
⏱️ Занимает минуты (только predict, без обучения)

# Часто, при обновлении данных (ежедневно?)
python -m training.src.scripts.generate_predictions \
  --config training/configs/generate_predictions.yaml

serve_bundle.py (Постоянно)
🚀 Загружает predictions.parquet в память один раз
⚡ Отвечает на запросы (<1ms)
🔄 Обновляется когда generate_predictions создаёт новый parquet
⏱️ Нет вычислений, только lookups

# Постоянно работает
python training/src/scripts/serve_bundle.py --model lgbm_ranker.txt

⚠️ Критические моменты

Feature consistency: generate_predictions ДОЛЖЕН использовать ТЕ ЖЕ флаги признаков как при обучении (features_config_path), иначе:

Missing features → вставляются нули ❌
Zero-only features → сломана модель ❌
Candidate alignment: top_k в training vs top_k_candidates в generate_predictions должны быть равны, иначе разные кандидаты.

Split ratios: Если меняешь train_ratio между обучением и инференсом, кандидаты будут другие.
