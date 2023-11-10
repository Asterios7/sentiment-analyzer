import streamlit as st
from PIL import Image
import requests
import os

IP_ADDRESS = os.environ.get('IP_ADDRESS')


def analyze_sentiment(text: str) -> dict:
    url = f'http://{IP_ADDRESS}/predict'
    data = {"text": text}
    response = requests.post(url, json=data)
    return response.json()


def main():
    # Basic page config
    st.set_page_config(
        page_title="Movie Review Sentiment Analyzer",
        page_icon = Image.open("./movie.png"),
        layout="wide"
    )

    st.title("Movie Review Sentiment Analysis")
    st.write("Enter your movie review below:")

    # Create text area
    user_input = st.text_area("")

    # Button (sends to api)
    if st.button("Analyze Sentiment"):
        if user_input == "":
            st.warning("Please enter a movie review.")
        else:
            sentiment_result = analyze_sentiment(user_input)
            st.write("Sentiment: ", f"**{sentiment_result['pred']}**" )
    

if __name__ == "__main__":
    main()