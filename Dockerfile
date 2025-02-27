FROM python:3.11.11-slim

RUN mkdir /app
WORKDIR /app

# Install system dependencies
RUN apt-get -y update && apt-get -y install curl tar

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3.11 -
ENV PATH="${PATH}:/root/.local/bin"

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install

# Download pre-trained models
RUN curl https://dental-pano-ai.s3.ap-southeast-1.amazonaws.com/models.tar.gz | tar -zx -C .

# Copy source code
COPY main.py ./

ENTRYPOINT ["poetry", "run", "python3.11", "main.py"]
