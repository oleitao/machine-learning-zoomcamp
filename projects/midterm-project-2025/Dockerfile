FROM python:3.11-slim

RUN pip install --no-cache-dir flask waitress xgboost scikit-learn pandas numpy

WORKDIR /app

COPY predict_model.py model_diabetes.bin ./

EXPOSE 5000

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:5000", "predict_model:app"]