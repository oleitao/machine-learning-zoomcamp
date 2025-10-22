FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

# Install uv (the package manager)
RUN python -m pip install -U uv

# Copy project files
COPY pyproject.toml ./
COPY app.py ./
COPY pipeline_v1.bin ./

# Install deps into a local .venv
RUN uv sync --no-dev --frozen || uv sync --no-dev

# Ensure the venv binaries (uvicorn, python, etc.) are on PATH
ENV VIRTUAL_ENV=/code/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# If you're using FastAPI, this should work out of the box
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
