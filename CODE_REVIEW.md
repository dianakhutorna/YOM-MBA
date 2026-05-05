# Code Review Summary

## 📋 Deployment Status
✅ **Используется в Lambda**:
- `training/src/scripts/lambda_handler.py` — AWS Lambda entry point
- `training/src/scripts/serve_recommendations_api.py` — FastAPI приложение для recommendations
- `training/src/services/recommendation_service.py` — сервис рекомендаций

❌ **НЕ используется в deployment**:
- `training/src/scripts/serve_bundle_api.py` — локальное API (не в Docker)
- `training/src/scripts/serve_bundle.py` — helper для bundle building (локально)
- `training/src/scripts/test_serve_bundle.py` — тесты для serve_bundle

---

## 🧹 Ненужные зависимости в requirements.txt

Эти пакеты установлены но **не импортируются** нигде в коде:
- `zenml` — не используется
- `xgboost` — не используется (используется только LightGBM)
- `scikit-learn` — не используется
- `pymongo[srv]==3.11` — не используется
- `matplotlib` — не используется
- `seaborn` — не используется
- `ipykernel` — для Jupyter только

**Рекомендация**: Убрать из requirements.txt, оставить в requirements-backend.txt только необходимое.

---

## 📁 Структура скриптов

### Core (нужны для обучения и inference)
✅ `run_training_pipeline.py` — основной training pipeline
✅ `generate_predictions.py` — генерация предсказаний после обучения
✅ `lambda_handler.py` — entry point для AWS Lambda
✅ `serve_recommendations_api.py` — FastAPI для recommendations

### Analysis/Debug (опционально)
⚠️  `check_personalization.py` — анализ персонализации моделиДЭ
⚠️  `check_new_vs_repeat.py` — анализ новых vs повторных товаров
⚠️  `model_test.py` — локальное тестирование моделей

### Deprecated (можно удалить)
❌ `serve_bundle_api.py` — локальное приложение, не нужно для deployment
❌ `serve_bundle.py` — helper для bundle building, не используется в Lambda
❌ `test_serve_bundle.py` — тесты для serve_bundle
❌ `run_experiment.py` — экспериментальный пайплайн (возможно)

---

## 💬 Комментарии и читаемость
✅ **Хорошо**:
- Нет странных комментариев
- Нет TODO/FIXME/HACK
- Код хорошо структурирован
- Логирование на месте

---

## 🎯 Рекомендации для сдачи профессору

### 1️⃣ Очистить requirements.txt
```bash
# Удалить неиспользуемые зависимости
pandas              # используется? проверить
numpy
scikit-learn         # ❌ УДАЛИТЬ
pymongo[srv]==3.11   # ❌ УДАЛИТЬ  
matplotlib           # ❌ УДАЛИТЬ
zenml                # ❌ УДАЛИТЬ
ipykernel            # ❌ УДАЛИТЬ
seaborn              # ❌ УДАЛИТЬ
xgboost              # ❌ УДАЛИТЬ
```

### 2️⃣ Удалить ненужные файлы (опционально)
```bash
rm training/src/scripts/serve_bundle_api.py     # не нужен, только локально
rm training/src/scripts/serve_bundle.py         # не нужен  
rm training/src/scripts/test_serve_bundle.py    # тесты к удаленному коду
# Возможно удалить:
# rm training/src/scripts/run_experiment.py    # экспериментальный pipeline
```

### 3️⃣ Проверить использование pandas
```bash
grep -r "import pandas\|from pandas" training/src --include="*.py"
```

---

## ✅ Что хорошо
- ✅ Deployment правильно настроен (Lambda + ECR/Docker)
- ✅ Чистая архитектура (отдельно training, inference, services)
- ✅ Хорошее логирование
- ✅ Конфиги отделены (training_pipeline.yaml, generate_predictions.yaml, etc.)
- ✅ Полный ML pipeline: preprocessing → features → training → evaluation → serving
