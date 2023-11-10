from fastapi import FastAPI
from pydantic import BaseModel
from review_classifier import review_classifier_gpt35

app = FastAPI()


class reviewClassifierIn(BaseModel):
    text: str 

class reviewClassifierOut(BaseModel):
    pred: str 


@app.get('/')
async def get_root():
    return "Welcome to the new TheyDo API!!!"


@app.post('/predict', response_model=reviewClassifierOut)
async def detect_face(user_input: reviewClassifierIn):

    text = user_input.text
    pred = review_classifier_gpt35(text=text)

    return {"pred": pred}