# Hermetic base image for SHL 2026.
# Pin every layer. Reproduce by building from the same git SHA.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pin uv to a known version (update deliberately).
ENV UV_VERSION=0.4.20
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && uv --version

WORKDIR /opt/shl2026

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock* /opt/shl2026/
RUN uv venv /opt/venv --python 3.11 \
 && . /opt/venv/bin/activate \
 && uv pip install --no-cache --strict ".[torch,dev]"

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV="/opt/venv"

# Copy code last so source edits don't bust the deps layer.
COPY . /opt/shl2026

RUN uv pip install --no-deps --no-cache -e .

# Smoke-import to fail the build early if packaging is broken.
RUN python -c "import shl2026; print('shl2026', shl2026.__version__)"

ENTRYPOINT ["python", "-m", "shl2026"]
