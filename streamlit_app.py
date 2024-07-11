import streamlit as st
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import sqlalchemy
import psycopg2
import spacy 
from spacy import displacy
from textblob import TextBlob
import numpy as np
import pyecharts.options as opts
from pyecharts.charts import Calendar
from streamlit_echarts import st_pyecharts
import en_core_web_md
import subprocess


st.set_page_config(page_title="Guardian Stats", 
                   layout="wide", 
                   initial_sidebar_state="expanded")


# connecting the web app to database 

conn = st.connection("postgresql", type="sql")
query_output = conn.query('SELECT * FROM student.dabble')

@st.cache_data
def get_data(query_ouput):
    df = pd.DataFrame(query_output)
    return df
# df['webpublicationdate'] = pd.to_datetime(df['webpublicationdate']).dt.date()


@st.cache_resource
def download_en_core_web_sm():
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])

@st.cache_resource  
def load_model():
    return en_core_web_md.load()

download_en_core_web_sm()

nlp = load_model()


st.title("The Guardian A.I News Data Explorer")
st.markdown('- Features NLP-processed article titles collected using the Guardian OpenPlatform API')
st.markdown('- Sentiment analysis of your group of articles chosen using the sidebar filters')
df = get_data(query_output)
today2 = datetime.date.today()
yesterday = today2 - datetime.timedelta(days=1)
yesterday = str(yesterday)
today_df = df[df['webpublicationdate'] == yesterday]
today_df = today_df.groupby(['pillarname']).size().reset_index(name='count')
st.markdown("### What were yesterdays publications by pillar? ")

fig = plt.figure(figsize=(10, 4))

a = sns.barplot(today_df, x='pillarname', y='count')
a.set_xlabel('Pillar')
a.set_ylabel('Article Count')
a.set_ybound(0, len(today_df))
st.pyplot(fig)

# # Sidebar for filtering articles

with st.sidebar: 
    st.header('Choose your filters')

    # Section names
    section_names = df['sectionname'].unique()
    pillar_list = df['pillarname'].unique()

#Section/topics (make it optional/delete)
    section = st.multiselect('Topics :newspaper:', options=section_names, default=section_names)
    pillar = st.multiselect('Pillar :pillar:', options=pillar_list, default=pillar_list)
    user_choice = st.text_input('Enter a keyword: ', '')

df_selection = df.query(
    "sectionname == @section & pillarname == @pillar & webtitle.str.contains(@user_choice, case=False, na=False)", engine='python')

df_selection = df_selection.reset_index()

def preprocess(text): 
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

# Process review.text to generate a column for sentiment analysis titled 'processed_review'
df_selection['webtitle_processed'] = df_selection['webtitle'].apply(preprocess)

# Return sentiment polarity for each review using Textblob 
def get_sentiment(article):
    blob = TextBlob(article) #Process the input using Textblob to generate a polarity, subjectivity tuple 
    sentiment = blob.sentiment.polarity 
    return sentiment # Output the polarity score 

# Apply the function to return a sentiment score for each review
df_selection['sentiment'] = df_selection['webtitle_processed'].apply(get_sentiment)

# Classify the sentiment polarity into one of three groups: negative, neutral or positive 
def classify_sentiment(sentiment):
# If the sentiment is larger than 0, output should be "positive"
    if sentiment > 0: 
        return "positive"
    elif sentiment < 0: 
        return 'negative'
    else: 
# If the sentiment is zero, output should be "neutral"
        return 'neutral'

df_selection['sentiment_group'] = df_selection['sentiment'].apply(classify_sentiment)
df_selection['webpubyear'] = pd.to_datetime(df_selection['webpublicationdate'], format="mixed").dt.year

#Bar chart article data grouped by publication year and sentiment group
x = df_selection.groupby(['webpubyear', 'sentiment_group']).size().reset_index(name='count')
display_df = df_selection.loc[:, ['webtitle', 'weburl', 'webtitle_processed', 'sentiment', 'sentiment_group']]
display_df.rename(columns={"webtitle": "Article Title", "weburl": "Article Page URL", "webtitle_processed": "Processed Article Title", "sentiment":"Sentiment Score", "sentiment_group":"Sentiment Group"})
st.header('Sentiment Scores for Selected Article Titles')
st.dataframe(display_df)

# Section 3 : Articles grouped into sentiment groups
st.markdown("### Are these articles positive, negative or neutral?")
fig = plt.figure(figsize=(10, 4))
xxx= sns.set_palette("dark:#5A9_r")
fig2 = plt.figure(figsize=(10, 4))
b = sns.barplot(x='webpubyear', y='count', hue='sentiment_group', data=x)
b.set_xlabel('Article Published Year')
b.set_ylabel('Article Count')
st.pyplot(fig2)

st.divider()


#Visualisations 
    # extract entities from the webtitles 
    # Calendar charts 

i = st.slider('Visualise the entities in the article title: ', min_value=0, max_value=len(df_selection))
def get_image(i):
    doc = nlp(df_selection['webtitle'][i]) 
    doc_html = displacy.render(doc, style='ent', jupyter=False)
    st.markdown(doc_html, unsafe_allow_html=True)
    return doc_html

if len(df_selection) > 0: 
    get_image(i)
else: 
    st.write('None')

# calendar chart 
articles_by_day = df_selection.groupby('webpublicationdate').size().reset_index(name='count').sort_values('webpublicationdate', ascending=True)
data = articles_by_day[['webpublicationdate', 'count']].values.tolist()

article_calender_23 = (
    Calendar()
.add(
        "", data, calendar_opts=opts.CalendarOpts(range_='2023')
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Publication tally 2023"),
        visualmap_opts=opts.VisualMapOpts(
            max_=0, min_=10, orient="horizontal", is_piecewise=False
        ) 
    )
    ) 

article_calender_24 = (
    Calendar()
.add(
        "", data, calendar_opts=opts.CalendarOpts(range_='2024')
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Publication tally 2024"),
        visualmap_opts=opts.VisualMapOpts(
            max_=0, min_=10, orient="horizontal", is_piecewise=False
        ) 
    )
    ) 

article_calender_22 = (
    Calendar()
.add(
        "", data, calendar_opts=opts.CalendarOpts(range_='2022')
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Publication tally 2022"),
        visualmap_opts=opts.VisualMapOpts(
            max_=0, min_=10, orient="horizontal", is_piecewise=False
        ) 
    )
    ) 

st.divider()

st.markdown("### Check when these articles were published throughout the year")

with st.expander("Publication Calendar: "):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: 
        label1 = st.button("2022")
    with col2: 
        label2 = st.button("2023")
    with col3: 
        label3 = st.button("2024")

    if label1: 
        st_pyecharts(article_calender_22)
    if label2:
        st_pyecharts(article_calender_23)
    if label3:
        st_pyecharts(article_calender_24)
# top 10 - people, organisations 

# '''

# list_of_entities = []
# for sentence in list: 
#     word_ent = ''
#     ent_type = None

#     for word in sentence: 
#         term = word.text
#         tag = word.ent
#         if tag: 
#             word_ent = ''.join(word_ent, term)
            
# '''
# topic modelling 

# hydralit feedback + sentiment bars 

# use LLM to create categories of AI use??

