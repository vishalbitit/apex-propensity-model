# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Apex Securities Customer Propensity Model
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT THIS FILE DOES:
#   Packages the entire ML project into a self-contained Docker image.
#   Anyone with Docker installed can run this — no Python setup needed.
#
# HOW TO BUILD:
#   docker build -t apex-propensity-model .
#
# HOW TO RUN:
#   docker run apex-propensity-model generate    → generate dataset
#   docker run apex-propensity-model train       → train models
#   docker run apex-propensity-model evaluate    → evaluate models
#   docker run apex-propensity-model predict     → score new customers
#   docker run apex-propensity-model all         → run full pipeline
# ─────────────────────────────────────────────────────────────────────────────


# ── Step 1: Choose base image ─────────────────────────────────────────────────
# We start from an official Python 3.11 image (slim = smaller size, no extras)
# Think of this as: "start with a clean laptop that already has Python installed"
FROM python:3.11-slim


# ── Step 2: Set working directory inside the container ───────────────────────
# All our project files will live at /app inside the container
# Think of this as: "create a project folder inside the container"
WORKDIR /app


# ── Step 3: Install system dependencies ──────────────────────────────────────
# Some Python packages need system-level C libraries to compile
# libgomp1 is required by LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# ── Step 4: Copy requirements.txt FIRST (Docker caching trick) ───────────────
# We copy requirements.txt before the rest of the code.
# Why? Docker caches each step. If requirements.txt hasn't changed,
# Docker skips reinstalling packages — making rebuilds much faster.
COPY requirements.txt .


# ── Step 5: Install Python dependencies ──────────────────────────────────────
# Install all packages from requirements.txt
# --no-cache-dir keeps the image size smaller
RUN pip install --no-cache-dir -r requirements.txt


# ── Step 6: Copy all project files into the container ────────────────────────
# The . (dot) means "copy everything from current folder on laptop"
# into /app inside the container
COPY . .


# ── Step 7: Create folders that the scripts will write to ────────────────────
# data/, models/, reports/ are created by the scripts but
# let's ensure they exist upfront
RUN mkdir -p data models reports


# ── Step 8: Set environment variables ────────────────────────────────────────
# Prevents Python from writing .pyc bytecode files (keeps container clean)
# Ensures print() statements appear in Docker logs immediately (no buffering)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# ── Step 9: Define entrypoint script ─────────────────────────────────────────
# ENTRYPOINT = the command that runs when the container starts
# We use a shell script so we can pass different commands (generate, train etc.)
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command if none is provided
CMD ["all"]
