Serving Layer
=============

Overview
--------

Serving returns recommendations from precomputed artifacts.
The online API does not train or score the model at request time.

Code entry points
-----------------

- ``training/src/scripts/serve_recommendations_api.py``
- ``training/src/services/recommendation_service.py``
- ``training/src/scripts/lambda_handler.py``

Production path (Lambda)
------------------------

- ``lambda_handler.py`` exposes ``handler = Mangum(app)``.
- FastAPI app is created with lifespan startup.
- On startup, service requires environment variables:

  - ``PREDICTIONS_S3_BUCKET``
  - ``PREDICTIONS_S3_KEY``
  - optional: ``MODEL_ID``
  - optional: ``LOCAL_PREDICTIONS_PATH``

- Service downloads predictions parquet from S3 and builds in-memory lookup.

Lookup logic
------------

``RecommendationService`` builds a dict keyed by ``(anchor_id, kiosk_id)``.
For each key, product IDs are sorted by score descending and returned with metadata fields.

API endpoints
-------------

- ``GET /health``
- ``GET /recommendations``
- ``POST /recommendations/multi``
- ``GET /docs``

Request contract
----------------

.. code-block:: text

   GET /recommendations?kioskId=<kiosk_id>&anchorId=<anchor_id>&limit=<N>

Endpoint constraints from code:

- ``anchorId`` and ``kioskId`` are required query parameters.
- ``limit`` default is ``20``.
- ``limit`` is validated in range ``1..100``.

Fallback context
----------------

The batch scoring pipeline generates fallback parquet artifacts (see :doc:`fallback`).
Serving code remains lookup-based and uses predictions parquet as primary online source.

Why this design
---------------

- Precomputed lookup keeps online latency low.
- FastAPI + Mangum provides API interface compatible with AWS Lambda.
- Offline scoring and online serving are decoupled for operational stability.

Local development
-----------------

.. code-block:: bash

   pip install -r requirements.txt
   ./venv/bin/python -m training.src.scripts.serve_recommendations_api
