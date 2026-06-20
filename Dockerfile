FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir flask flask-cors requests beautifulsoup4

COPY . .

# Run full pipeline then start the web server.
# If results/models/ already has .joblib files (mounted volume), skip training.
CMD ["sh", "-c", "\
  if [ ! -f results/models/rf.joblib ]; then \
    echo '==> Running ML pipeline...' && \
    cd src && \
    python data_prep.py && \
    python features.py && \
    python train.py && \
    python evaluate.py && \
    cd ..; \
  else \
    echo '==> Models found, skipping training.'; \
  fi && \
  echo '==> Starting web server on :5000' && \
  python app.py \
"]

EXPOSE 5000
