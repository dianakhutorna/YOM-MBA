Deployment
==========

Overview
--------

Production serving is deployed on AWS Lambda via Docker image workflow.

Deployment path
---------------

1. Train model and generate batch artifacts.
2. Build Docker image for Lambda runtime.
3. Push image to registry.
4. Update Lambda function image through CI/CD.

Documented build command
------------------------

.. code-block:: bash

   docker buildx build \
     --platform linux/amd64 \
     --provenance=false \
     -t diana-backend:latest \
     --push .

Application entry point
-----------------------

- Lambda handler target: ``training.src.scripts.lambda_handler.handler``
- Handler wraps FastAPI app with Mangum.

CI/CD notes
-----------

Project documentation describes deployment through GitHub Actions workflow ``.github/workflows/deploy.yml``.

Runtime characteristics
-----------------------

Documented characteristics for production serving:

- Startup: about 90 seconds
- Warm request latency: about 2 ms
- Memory: about 2 GB per Lambda instance

Dependencies
------------

The project uses two requirement sets:

- ``requirements.txt``: full development/training environment
- ``requirements-backend.txt``: minimal backend runtime dependencies

Local development serving
-------------------------

.. code-block:: bash

   pip install -r requirements.txt
   ./venv/bin/python -m training.src.scripts.serve_recommendations_api

Environment variables for API startup
-------------------------------------

Serving startup requires:

- ``PREDICTIONS_S3_BUCKET``
- ``PREDICTIONS_S3_KEY``

Optional:

- ``MODEL_ID``
- ``LOCAL_PREDICTIONS_PATH``

Operational sequence
--------------------

To refresh production recommendations:

1. Run training pipeline.
2. Run batch scoring pipeline.
3. Deploy updated serving image/workflow.
