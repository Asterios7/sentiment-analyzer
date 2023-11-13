import openai
import os
import logging
logger = logging.getLogger(__name__)

openai.api_key = os.environ.get("OPENAI_API_KEY")


def review_classifier_gpt35(text: str) -> str:
    """
    Classifies whether a movie review is positive or negative,
    using Openai's gpt-3.5-turbo model.

    Args:
    text: str
        The review string
    response_text: list
        The predicted labels
    """
    messages = [
        {"role": "user", 
         "content": f"""Analyze the following movie review and determine if the sentiment is: positive or negative.
                                        Return answer in single word as either positive or negative: {text}"""}
        ]

    response = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=messages,
                      max_tokens=1,
                      n=1,
                      stop=None,
                      temperature=0.1,
                      request_timeout=4)

    response_text = response.choices[0].message.content.strip().lower()

    return response_text


def ood_classifier_gpt35(text: str) -> str:
    """
    Classifies whether a given text is a movie review or not,
    using Openai's gpt-3.5-turbo model.

    Args:
    text: str
        A text string
    response_text: str
        The prediction
    """
    messages = [
        {"role": "user",
         "content": f""" Assess if the following is a movie review or not and return exactly one word: 
         'yes' if it is a review for movies or any kind of tv show review or film review or cinema review,
         'no' if it is not 
         {text} """}
        ]
    
    response = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=messages,
                      max_tokens=3,
                      n=1,
                      stop=None,
                      temperature=0.1,
                      request_timeout=4)

    response_text = response.choices[0].message.content.strip().lower()

    return response_text