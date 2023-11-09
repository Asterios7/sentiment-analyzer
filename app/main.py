from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class predRequest(BaseModel):
    text: str 

class predResponse(BaseModel):
    pred: str 


@app.get('/')
async def get_root():
    return "Welcome to the new TheyDo API!!!"


@app.post('/predict', response_model=predResponse)
async def detect_face(user_input: predRequest):
    pred = user_input.text

    return {"pred": pred}