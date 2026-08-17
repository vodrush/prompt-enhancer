# Hermes prompt watcher — reformulates the last user message via the OmniRoute combo.
FROM python:3.11-slim

RUN pip install --no-cache-dir aiohttp==3.14.3

WORKDIR /app
COPY watcher.py reformulate.py ./

ENV WATCHER_LISTEN=0.0.0.0:20500
EXPOSE 20500
CMD ["python", "watcher.py"]
