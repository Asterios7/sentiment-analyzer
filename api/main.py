from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from gptClassifiers import review_classifier_gpt35, ood_classifier_gpt35
import logging

logging.basicConfig(filename='logs/log.log', filemode="w",
                    encoding='utf-8', level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

    logger.info(f"Received request with text: {user_input.text}")

    text = user_input.text
    try:
        is_movie_review = ood_classifier_gpt35(text=text)
    except Exception as e:
        logger.error(f"Error in ood_classifier: {str(e)}")
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, 
                            detail=f"{str(e)}")
    
    if is_movie_review == "yes":
        try:
            pred = review_classifier_gpt35(text=text)
            logger.info(f"Prediction for text '{text}': {pred}")
            return {"pred": pred}
        except Exception as e:
            logger.error(f"Error in review_classifier: {str(e)}")
            raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, 
                                detail=f"{str(e)}",)
    else:
        logger.warning("Not a movie review, invalid input")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                            detail="Not a movie review, invalid input")




