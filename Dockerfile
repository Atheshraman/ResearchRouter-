FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --create-home app

WORKDIR /app

# Install dependencies first (cache layer)
COPY pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy source code
COPY src/ src/

# Switch to non-root user
USER app

# Default: run MCP server on stdio
CMD ["python", "-m", "research_router"]
