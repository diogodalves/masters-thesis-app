FROM python:3.10-slim AS setup

ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PATH="/root/.poetry/bin:$PATH" \
    POETRY_CACHE_DIR=/tmp/.cache/pypoetry \
    TRANSFORMERS_CACHE=/opt/sam/.cache/huggingface \
    POETRY_VERSION=1.8.3

COPY poetry.lock pyproject.toml ./

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get --no-install-recommends install build-essential libpq-dev -y && \
    apt-get clean && \
    pip install --upgrade pip && \
    pip install setuptools>=70.0.0 && \
    pip install poetry==$POETRY_VERSION && \
    poetry config installer.max-workers 10 && \
    poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-root

FROM python:3.10-slim

COPY --from=setup /.venv /.venv
COPY app/ app/
COPY start.sh /start.sh

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get --no-install-recommends install \
                                    libgl1 \
                                    build-essential \
                                    libpq-dev \
                                    libglib2.0-0 \
                                    libsm6 \
                                    libxrender1 \
                                    libxext6 \
                                    -y && \
    apt-get clean


ENV PATH="/.venv/bin:$PATH"

CMD ["bash", "./start.sh"]
