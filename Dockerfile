FROM lablup/python:3.11-ubuntu22.04

RUN mkdir -p /workspace/
WORKDIR /workspace/

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install poetry

# Clone the repository
RUN git clone https://github.com/stmharry/dental-pano-ai.git 
WORKDIR /workspace/dental-pano-ai

# Install dependencies using Poetry.
RUN poetry install

# Download pre-trained models
RUN curl -o models.tar.gz https://dental-pano-ai.s3.ap-southeast-1.amazonaws.com/models.tar.gz
RUN tar zxvf models.tar.gz


