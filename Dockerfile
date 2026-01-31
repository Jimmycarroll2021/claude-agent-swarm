# Claude Agent Swarm - Docker Image

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (for MCP servers)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 swarmuser && chown -R swarmuser:swarmuser /app
USER swarmuser

# Create data directory
RUN mkdir -p /app/data

# Expose port (for web UI if enabled)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import claude_agent_swarm; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "claude_agent_swarm"]
