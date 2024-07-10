import joblib
import pickle
import snscrape.modules.twitter as sntwitter
import snscrape.modules.reddit as snreddit
import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st

nltk.download(["stopwords","wordnet","punkt"])

data = []
lemmatizer = WordNetLemmatizer()
stop_list = stopwords.words('english')
stop_list.extend(["reddit","subreddit","rt","tweet","twitter"])

vectorizer = pickle.load(open("./Model Files/vectorizer.pickle","rb"))

anxiety_model = joblib.load("./Model Files/anxiety_model.sav")
bpd_model = joblib.load("./Model Files/bpd_model.sav")
depression_model = joblib.load("./Model Files/depression_model.sav")
eatingdisorder_model = joblib.load("./Model Files/eatingdisorder_model.sav")
ptsd_model = joblib.load("./Model Files/ptsd_model.sav")

models = [("Anxiety",anxiety_model),
          ("Depression",depression_model),
          ("BPD",bpd_model),
          ("Eating Disorder",eatingdisorder_model),
          ("PTSD",ptsd_model)]


def getTwitterData(user:str):
    for i,tweet in enumerate(sntwitter.TwitterProfileScraper(user).get_items()):
        data.append(tweet.rawContent)


def getRedditData(user:str):
    for i,content in enumerate(snreddit.RedditUserScraper(user).get_items()):
        try:
            data.append(content.body)
        except:
            data.append(content.title)


def preProcessText(text:str):
    text = re.sub(r'@\S+',' ',text)
    text = re.sub(r'#\S+',' ',text)
    text = re.sub(r'u\/\S+',' ',text)
    text = re.sub(r'r\/S+',' ',text)
    text = re.sub(r'http\S+',' ',text)
    text = re.sub(r'[^a-zA-Z]+',' ',text).strip().lower()
    
    tokens = word_tokenize(text)
    sw_removed = [word for word in tokens if word not in stop_list]
    ls = []
    for w in sw_removed:
        ls.append(lemmatizer.lemmatize(w))
    
    return ' '.join(ls)


def getUserData(twitter_name:str = None,reddit_name:str = None):
    if twitter_name != "None":
        getTwitterData(twitter_name)

    if reddit_name != "None":
        getRedditData(reddit_name)
    
    d = pd.DataFrame(data)
    d = d[0].apply(preProcessText)
    clean_string = " ".join(d.to_list())
    
    return clean_string


def predictMentalState(twitter_name:str = None,reddit_name:str = None):
    clean_string = getUserData(twitter_name,reddit_name)
    vector = vectorizer.transform([clean_string])

    res = []
    for model in models:
        res.append((model[0],model[1].predict_proba(vector)[0][0]))

    preds_df = pd.DataFrame(res)
    preds_df.columns = ["Disorder","Percentage"]
    return preds_df[preds_df["Percentage"] > 0.5]


if __name__ == "__main__":
    st.title("""
    Mental State analysis
    """)

    st.write("""
    Predicting a user’s mental health by analysing their behaviour and the type of content they consume on social media.
    """)
    
    twitter_name = st.text_input("Twitter Username",None)
    reddit_name = st.text_input("Reddit Username",None)

    predict = st.button("Predict Mental State")

    if predict:
        out = predictMentalState(twitter_name,reddit_name)
        out.reset_index(inplace = True)
        out.drop("index",axis = 1,inplace = True)
        st.dataframe(out)
