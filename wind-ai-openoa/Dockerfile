FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

COPY . .

CMD ["python", "scripts/check_project.py"]
