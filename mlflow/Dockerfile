# Build from a LINUX lightweight version of Anaconda
FROM python:3.10

# Update packages and install nano unzip and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    nano unzip curl && rm -rf /var/lib/apt/lists/*

# Install AWS cli - Necessary since we are going to interact with S3 
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
 && unzip awscliv2.zip \
 && ./aws/install

# THIS IS SPECIFIC TO HUGGINGFACE
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy project
COPY --chown=user . $HOME/app

# Dependencies
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Launch mlflow server (PORT fourni par HF Spaces)
CMD mlflow server -p $PORT \
  --host 0.0.0.0 \
  --backend-store-uri $BACKEND_STORE_URI \
  --default-artifact-root $ARTIFACT_STORE_URI \
  --allowed-hosts jyvuillequez-mlflow-server.hf.space \
  --cors-allowed-origins "https://jyvuillequez-mlflow-server.hf.space"