FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app

# dependencies first: this slow layer is cached until requirements change
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# source code last: changing code only rebuilds from here
COPY app/ ./app/

# run as a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:5000/health')"

CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app.app:app"]