import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the trained CNN model
model = keras.models.load_model('cnn_model.h5')

# Load word index from IMDb dataset
word_index = keras.datasets.imdb.get_word_index()

# Reverse word index for decoding reviews
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Function to decode review
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Function to preprocess text data
def preprocess_text(text):
    # Convert text to sequence of integers
    sequence = [word_index[word] if word in word_index and word_index[word] < 5000 else 2 for word in text.split()]
    # Pad sequence to fixed length
    padded_sequence = pad_sequences([sequence], maxlen=400)
    return padded_sequence

# Function to predict sentiment
def predict_sentiment(review):
    # Preprocess input text
    processed_review = preprocess_text(review)
    # Predict sentiment
    prediction = model.predict(processed_review)[0][0]
    return prediction

# Home page

def home():
    st.subheader("Home")
    st.write("Welcome to the Sentiment Analysis App")
    imageha = mpimg.imread('senti.jpg')     
    st.image(imageha)
    st.write('By using CNN Model to predict  Sentiment in movies review more Accurately')
    st.header('About Dataset')
    st.write("The IMDb dataset is a widely-used benchmark dataset for sentiment analysis tasks. It consists of movie reviews along with their associated sentiment labels, typically positive or negative. This dataset is often used in deep learning and natural language processing research to train and evaluate sentiment analysis models.")
    st.header("Key Features")
    col1,col2,col3 = st.columns(3)

    
    col1.header('Large scale 📈')
    col2.header("Balanced sentiment ⚖️")
    col3.header("Human labeled data 🏷")
# Analysis page
def prediction():
    st.title("Analysis Page")
    imagehb = mpimg.imread('images.jpeg')     
    st.image(imagehb)
    st.write("Enter your movie review in the text box below and click on 'Predict Sentiment' to see the prediction.")
    
    # Text input for user to enter review
    review = st.text_input("Enter your movie review:", "")

    # Button to predict sentiment
    if st.button("Predict Sentiment"):
        if review.strip() == "":
            st.warning("Please enter a movie review.")
        else:
            # Predict sentiment
            prediction = predict_sentiment(review)
            # Display prediction
            if prediction >= 0.5:
                st.success(f"Sentiment: Positive ({prediction})")
            else:
                st.error(f"Sentiment: Negative ({prediction})")

def main():
    st.set_page_config(layout="wide")
    st.title("Sentiment Analysis App")
# Create the tab layout
    tabs = ["Home", "Prediction"]

    page = st.sidebar.selectbox("Select a page", tabs)

# Show the appropriate page based on the user selection
    if page == "Home":
        home()
    elif page == "Prediction":
        prediction()
    
   
main()