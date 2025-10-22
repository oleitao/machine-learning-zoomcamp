
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

class Lead(BaseModel):
    lead_source: str | None = None
    number_of_courses_viewed: float | int | None = 0
    annual_income: float | int | None = 0

with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

app = FastAPI()

@app.get('/')
def root():
    return {'status': 'ok'}

@app.post('/predict')
def predict(lead: Lead):
    record = lead.model_dump()
    proba = float(pipeline.predict_proba([record])[0, 1])
    return {'probability': proba}
