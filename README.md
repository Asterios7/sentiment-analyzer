# Sentiment Analysis App

[**Overview**](#overview)
| [**App**](#app)
| [**Notebooks**](#notebooks)

## Overview<a id="overview"></a>

Movie reviews sentiment analysis with Openai's gpt-3.5-turbo.

This repository contains:

1. A sentiment analysis application, consisting of Streamlit web app which connects to a Fastapi app.
2. Notebooks for measuring the performance of gpt-3.5-turbo on analyzing movie review sentiment and on detecting whether a certain text is a movie review.

## App<a id="app"></a>

#### System Requirements

- Docker

#### How to start the app

After you clone this repository:

1. Go to /path/to/sentiment-analyzer
2. Create a .env that looks like the following (**remember to replace the key**):
    ```
    OPENAI_API_KEY=<replace-with-my-actual-openai-key>
    ```
3. Execute `docker compose up --build`
4. Open browser at http://localhost:8501/

For stopping the app from the same terminal path execute:

`docker compose down`

## Notebooks<a id="notebooks"></a>

#### Install Notebook Requirements

Go to sentiment-analyzer/notebooks and:

`pip install -r requirements.txt`

#### 

1. **`sentiment_analysis.ipynb`** 
    - This notebook conducts sentiment analysis on movie reviews from the imdb dataset, utilizing the GPT-3.5-turbo model. It calculates and stores metrics like precision, recall, F1 score, and accuracy in `metrics_sentiment.json`.

2. **`out_of_distribution.ipynb`** 
    - This notebook focuses on classifying whether a given text is a movie review or not using datasets from IMDB, Amazon Polarity, and Yelp Polarity datasets. The GPT-3.5-turbo model is employed for this classification task. The notebook calculates and stores metrics such as precision, recall, F1 score, and accuracy in `metrics_ood.json`.