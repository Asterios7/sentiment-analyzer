from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from gptClassifiers import review_classifier_gpt35, ood_classifier_gpt35

app = FastAPI()


class reviewClassifierIn(BaseModel):
    text: str 

class reviewClassifierOut(BaseModel):
    pred: str 


@app.get('/')
async def get_root():
    return "Welcome to my Movie Sentiment Analysis API!!!"


@app.post('/predict', response_model=reviewClassifierOut)
async def detect_face(user_input: reviewClassifierIn):


    text = user_input.text
    try:
        is_movie_review = ood_classifier_gpt35(text=text)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, 
                            detail=f"{str(e)}")
    
    if is_movie_review == "yes":
        try:
            pred = review_classifier_gpt35(text=text)
            return {"pred": pred}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, 
                                detail=f"{str(e)}",)
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                            detail="Not a movie review, invalid input")




