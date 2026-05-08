Testing and Analysis
====================

Unit tests
----------

Run all tests:

.. code-block:: bash

   ./venv/bin/python -m pytest training/tests/ -q

The test suite in ``tests/`` covers preprocessing, basket building, candidate generation, feature table creation, labels, top-k selection, and pipeline smoke checks.

Analysis scripts
----------------

Personalization analysis:

.. code-block:: bash

   ./venv/bin/python -m training.src.scripts.check_personalization \
     --config training/configs/training_pipeline.yaml --top-k 5 --sample-kiosks 300

New-vs-repeat analysis:

.. code-block:: bash

   ./venv/bin/python -m training.src.scripts.check_new_vs_repeat --top-k 5 --sample-kiosks 200

Local API verification
----------------------

.. code-block:: bash

   ./venv/bin/python -m training.src.scripts.serve_recommendations_api
   curl http://localhost:8000/health
   curl "http://localhost:8000/recommendations?kioskId=<kiosk_id>&anchorId=<anchor_id>&limit=10"

Why these checks matter
-----------------------

- Unit tests validate core pipeline transformations.
- Analysis scripts validate recommendation behavior beyond pure offline ranking metrics.
- API checks verify serving readiness and request/response path.
