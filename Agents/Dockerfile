FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV EPSPEC_AGENTS_DIR=/workspace/Agents
ENV EPSPEC_PROJECT_ROOT=/workspace

WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir "/workspace/Agents"
RUN useradd --create-home --shell /usr/sbin/nologin epspec && chown -R epspec:epspec /workspace/Agents

USER epspec
WORKDIR /workspace/Agents
EXPOSE 8000

CMD ["epspec-agents", "serve", "--host", "0.0.0.0", "--port", "8000"]
