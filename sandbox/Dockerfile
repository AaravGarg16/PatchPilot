# Execution sandbox for repository code. Tests and the linter run here, never on the host.
FROM python:3.13-slim

# Test tooling is baked in so runs never need the network for it. Versions match requirements.txt.
RUN pip install --no-cache-dir pytest==9.1.1 ruff==0.16.10 \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin sandbox

USER 10001:10001
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH=/deps \
    HOME=/tmp
