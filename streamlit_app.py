import streamlit as st
import pandas as pd
import datetime
import seaborn as sns
import sqlalchemy
import psycopg2

# connecting the web app to database 
conn = st.connection("postgresql", type="sql")
df = conn.query('SELECT * FROM student.dabble', ttl='10m')
st.dataframe(df)
df['webpublicationdate'] = pd.to_datetime(df['webpublicationdate']).dt.date

col1, col2 = st.columns([1, 2])
with col1: 
    st.title("Guardian News Environmental Articles through the Years")
    st.write(
        "Choose a tag and we will find Guardian articles related to that tag. "
        "Explore the semantics, people or countries primarily involved with articles, available through NLP. "
        "Close in on which environmental topics have the most articles. "
)


# sidebar for filtering articles 
st.sidebar.header('Choose your filters')

# section names 
array = df['sectionname'].unique()
# from date / to date 
begin_date = st.sidebar.date_input('Start date:', df['webpublicationdate'].min())
end_date = st.sidebar.date_input('End Date: ', df['webpublicationdate'].max()) 
#word-based filtering 
user_choice = st.sidebar.text_input('Enter a keyword: ', '')
section = st.sidebar.multiselect('Topics :+1:', array)
pillar = st.sidebar.checkbox('Pillar', ['Arts', 'Opinion', 'Sport', 'News'])

user_df = df[(df['webtitle'] == user_choice)]
final_df = pd.DataFrame()
final_df = df[(user_df[user_df['webpublicationdate']] >= begin_date) & (user_df[user_df['webpublicationdate']] <= end_date)].copy()

#return dataframe 
#with col1: 

st.divider()

#with col 2: 
#visualisations 
# top 10 ent/organisations mentioned ni articles in the past month 
# most positive month? negative also / neutral 
    # extract entities from the webtitles 
    # sentiment scores for web titiles 
    # group the articles by month published/? - count sentiment. 



