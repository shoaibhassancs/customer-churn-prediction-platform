# Image for the Streamlit app. Build and run it with `docker compose up --build`.
FROM python:3.12-slim

# uv, copied from its official image, installs the exact versions in uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so this slow layer is reused until pyproject.toml or uv.lock
# changes.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# A demo model trained on made-up customers, so a host that wipes its disk on
# restart still starts with a working model. Copied into place at start-up.
ENV CHURN_DEMO_MODEL_PATH=/app/demo-model/baseline.joblib
RUN /app/.venv/bin/python -m churn_platform.demo build

# Run as a normal user, not root. /data/models is where the saved model lives.
RUN useradd --create-home app && mkdir -p /data/models && chown app /data/models
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    CHURN_MODEL_PATH=/data/models/baseline.joblib

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Put the demo model in place if there is no model yet, then start the app.
CMD ["sh", "-c", "python -m churn_platform.demo restore && exec streamlit run src/churn_platform/app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false"]
