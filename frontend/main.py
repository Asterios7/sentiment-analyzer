import streamlit as st
from PIL import Image
import requests
import os

IP_ADDRESS = os.environ.get('IP_ADDRESS')


def analyze_sentiment(text: str) -> requests.models.Response:
    url = f'http://{IP_ADDRESS}/predict'
    data = {"text": text}
    return requests.post(url, json=data)


def main():
    # Basic page config
    st.set_page_config(
        page_title="Movie Review Sentiment Analyzer",
        page_icon = Image.open("./movie.png"),
        layout="wide"
    )

    st.title("Movie Review Sentiment Analysis")

    # Create text area for review
    user_input = st.text_area("Enter your movie review here: ")

    # Button (sends to api)
    if st.button("Analyze Sentiment"):

        if user_input.isspace() or user_input == "":
            st.warning("Please enter a movie review.")
        else:
            response = analyze_sentiment(user_input)
            if response.status_code == 200:
                st.write("Review sentiment: ", f"**{response.json()['pred']}**" )
            elif response.status_code == 422:
                st.write(response.json()['detail'])
            else:
                st.write(response.json()['detail'])
                st.write("Openai api error, please retry.")


if __name__ == "__main__":
    main()