# One image for both services: the API (default) and the Streamlit dashboard (SERVICE=ui).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY models ./models
COPY data ./data
COPY reports ./reports
COPY deploy/start.sh deploy/healthcheck.py ./deploy/

EXPOSE 8000 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
  CMD python deploy/healthcheck.py

CMD ["sh", "deploy/start.sh"]
