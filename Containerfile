FROM rayproject/ray:2.9.0-py311

USER root

# System deps for PyBoy (SDL2 headless) and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-dev \
    libsdl2-2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER ray
WORKDIR /home/ray

# Install Python dependencies
COPY --chown=ray:users pyproject.toml ./
RUN pip install --no-cache-dir \
    "pyboy>=2.6.0" \
    "gymnasium>=0.29.0" \
    "numpy>=1.24.0" \
    "ray[default,rllib]>=2.9.0" \
    "torch>=2.1.0" \
    "httpx>=0.25.0" \
    "boto3>=1.28.0" \
    "Pillow>=9.0.0" \
    "PyYAML>=6.0" \
    "pydantic>=2.0.0"

# Copy application code
COPY --chown=ray:users agent/ ./agent/
COPY --chown=ray:users scripts/ ./scripts/

# Copy ROM and save state
COPY --chown=ray:users new/ignored/zelda.gbc ./roms/zelda.gbc
COPY --chown=ray:users new/ignored/zelda.gbc.state ./roms/zelda.gbc.state
COPY --chown=ray:users new/ignored/zelda.gbc.ram ./roms/zelda.gbc.ram

ENV ROM_PATH=/home/ray/roms/zelda.gbc
ENV SAVE_STATE_PATH=/home/ray/roms/zelda.gbc.state
ENV PYTHONPATH=/home/ray

EXPOSE 6379 8265 10001 8080
