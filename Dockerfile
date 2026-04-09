FROM public.ecr.aws/lambda/python:3.12

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY training /var/task/training

CMD ["training.src.scripts.lambda_handler.handler"]