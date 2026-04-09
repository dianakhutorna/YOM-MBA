FROM public.ecr.aws/lambda/python:3.12

# Install deps
COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy only application code
COPY training/src /var/task/training/src

# Lambda handler: module.function
CMD ["training.src.scripts.lambda_handler.handler"]