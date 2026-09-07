FROM python:3.12-slim

WORKDIR /opt/ml/processing

COPY docker/processing/requirements.txt /opt/ml/processing/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /opt/ml/processing/requirements.txt

COPY src /opt/ml/processing/src
COPY scripts/sagemaker /opt/ml/processing/scripts/sagemaker
COPY pyproject.toml /opt/ml/processing/pyproject.toml

ENV PYTHONPATH=/opt/ml/processing

ENTRYPOINT ["python"]
