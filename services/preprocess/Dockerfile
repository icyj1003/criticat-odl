FROM bytewax/bytewax:0.8.0-python3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN /venv/bin/pip install -r requirements.txt

ENTRYPOINT ["/venv/bin/python", "-u", "dataflow.py"]
