import streamlit as st
st.title("Mental Health State analysis")


st.write("""
The aim of this project is to use the data available about the user in social media to identify their mental health state.
Social media data can provide insights into a user's mental health state in several ways. The type of content they post and consume,
are linked to mental health. The sentiment analysis of their posts, comments, and interactions can give an idea of their emotional state and how they are feeling.
""")


st.subheader("Data used")
st.write("""
The data for the building the model was scraped from reddit. Top 500 posts from the following 5 subreddits were used:
1. Depression
2. Anxiety
3. BPD (Bipolar Disorder)
4. Eating Disorder
5. PTSD (Post Traumatic Stress Disorder)
""")

st.subheader("Model Building")
st.write("""
Naive bayes was used for building the prediction model. 5 models each for one disorder was built and is used for predicting if the user has the respective disorder and if so how severe it is.
""")