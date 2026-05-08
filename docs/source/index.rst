.. YOM Recommender documentation master file

YOM Bundle Recommender System
==============================

.. image:: https://img.shields.io/badge/Python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :alt: License

**YOM Bundle Recommender** — A production-grade ML system for product bundle recommendations using learning-to-rank models.

The system identifies the best products to recommend as bundles paired with specific anchor products at each point of sale (POS terminal).

**Key Principle:** No online ML inference. The model pre-computes scores for all possible (store, anchor, candidate) combinations in batch mode. Serving simply performs dictionary lookups (~2ms per request).

---

**3-Stage Architecture:**

.. code-block:: text

   ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
   │  TRAINING STAGE     │      │  BATCH SCORING      │      │  SERVING STAGE      │
   │  (Monthly)          │  →   │  (Daily/Weekly)     │  →   │  (24/7 Lambda)      │
   │  training.py        │      │ generate_predictions│      │ lambda_handler.py   │
   └─────────────────────┘      └─────────────────────┘      └─────────────────────┘
          ↓                           ↓                            ↓
    Trained Model           Parquet Files                Pre-computed Dict
    + Features           (Predictions + Fallbacks)       + Business Rules


**Getting Started:**

- :doc:`quickstart` — First run in 5 minutes
- :doc:`architecture` — Full system design and decisions
- :doc:`training` — Detailed model training guide

**Main Documentation:**

.. toctree::
   :maxdepth: 3
   :caption: Documentation:

   quickstart
   architecture
   data_flow
   training
   model
   model_code
   inference
   serving
   fallback
   features
   file_reference
   configuration
   testing
   deployment


**Index and Search:**

* :ref:`genindex`
* :ref:`search`

