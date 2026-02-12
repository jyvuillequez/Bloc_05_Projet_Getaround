FROM python:3.11-slim

WORKDIR /app

# Installation dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie appplication et artifacts
COPY app.py .
COPY model.joblib .

# HF default port
EXPOSE 7860

# Start API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]