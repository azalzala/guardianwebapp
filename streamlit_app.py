import streamlit as st
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import sqlalchemy
import psycopg2
import spacy 
from textblob import TextBlob
from spacy import displacy

st.set_page_config(page_title="Guardian Stats", 
                   layout="wide", 
                   initial_sidebar_state="expanded")


# connecting the web app to database 
conn = st.connection("postgresql", type="sql")
df = conn.query('SELECT * FROM student.dabble')

@st.cache_data
def get_data(df):
    df = pd.DataFrame(df)
    df['webpublicationdate'] = pd.to_datetime(df['webpublicationdate'])
    return df

@st.cache_resource  
def load_model():
    return spacy.load('en_core_web_md')

nlp = load_model()


st.title("Guardian A.I News Data")

df = get_data(df)
today2 = datetime.date.today()
yesterday = today2 - datetime.timedelta(days=1)
today = str(yesterday)
today_df = df[df['webpublicationdate'] == today]
today_df = today_df.groupby(['pillarname']).size().reset_index(name='count')
st.markdown("### Yesterday's Publications: ")
fig = plt.figure(figsize=(10, 4))
sns.barplot(today_df, x='pillarname', y='count')
st.pyplot(fig)

today_df2 = df[df['webpublicationdate'] == today2]
number_of_articles = len(today_df2)
st.write(f"Total number of articles published today: {number_of_articles}")

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
df_selection['webpubyear'] = pd.to_datetime(df_selection['webpublicationdate']).dt.year

x = df_selection.groupby(['webpubyear', 'sentiment_group']).size().reset_index(name='count')

st.header('NLP dataframe')
st.dataframe(df_selection)

st.markdown("### Article sentiment scores: ")
fig = plt.figure(figsize=(10, 4))
xxx= sns.set_palette("dark:#5A9_r")
sns.barplot(x='webpubyear', y='count', hue='sentiment_group', data=x, palette=xxx)

st.pyplot(fig)
st.divider()



# search = st.sidebar.button('Search')

# # # # Print webtitles
# webtitles = list(user_df['webtitle'])
# print(webtitles)

# # Button to trigger search action
# if st.button('Show Articles'):
#     if not user_df.empty:
#         st.success('Articles found.')
#         st.write(webtitles)
#     else:
#         st.error('No articles to show.')

#     # Display the final filtered DataFrame
# st.dataframe(user_df)   

# st.divider()

# # #with col 2: 
# # #visualisations 
# # # top 10 ent/organisations mentioned ni articles in the past month 
# # # most positive month? negative also / neutral 
# #     # extract entities from the webtitles 
# #     # sentiment scores for web titiles 

sns.barplot(x='webpubyear', y='count', hue='sentiment_group', data=x)
# #     # group the articles by month published/? - count sentiment. 

#user_choice = st.text_input('Enter a keyword: ', '')

#user_df = df[df['webtitle'].str.contains(user_choice, case=False, na=False)]
#st.dataframe(user_df)

# i = st.number_input('Investigate the article entities further for a web title of your choice: ', 0, len(df_selection))
# doc = df_selection['webtitle'][i]
# displacy.render(doc, style='ent', manual=True)
