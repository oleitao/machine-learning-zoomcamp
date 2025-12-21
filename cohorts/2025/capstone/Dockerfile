FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV XG_MODEL_FILE=models/xg_logreg_sigmoid.joblib

EXPOSE 9696

CMD ["python", "predict.py"]

