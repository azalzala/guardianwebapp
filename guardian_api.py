import requests
import pandas as pd
import psycopg2 as pg
import datetime 
import os 
from dotenv import load_dotenv

load_dotenv() 

api_key = os.environ["API-KEY"]
database = os.environ["DATABASE"]
user = os.environ["DBUSER"]
password = os.environ["PASSWORD"]
host = os.environ["HOST"]
port = os.environ["PORT"]

conn = pg.connect(database=database,
                  user=user,
                  password=password,
                  host=host,
                  port=port)
cur = conn.cursor()

def get_data(query, from_date):

    url = f'https://content.guardianapis.com/search?q={query}&from-date={from_date}&page-size=200'
    params = {'api-key': api_key,
              'q': query,
              'from-date': from_date,
              'page-size': 200,

              }
              # Page size parameter 
    response = requests.get(url, params = params).json()

    return response

df = pd.DataFrame(columns=['id'])

#From date parameters for collecting data/updating the database
today = datetime.date.today()
data = get_data('ai', today)
f = data['response']
ai_data = f['results']
for i in ai_data:
  id = i['id']

  for key, value in i.items():
      df.loc[id, key] = value


#Add new numbered index column
df.set_index('id', inplace=True)
df.reset_index(inplace=True)
# Format the datetime column 
df['webPublicationDate'] = pd.to_datetime(df['webPublicationDate']).dt.date
df.drop(columns=['sectionId', 'apiUrl', 'isHosted', 'pillarId'], inplace=True)
df.sort_values('webPublicationDate', ascending=True)

for i in range(len(df)): 
    row_values = list(df.loc[i][['id', 'type','sectionName','webPublicationDate', 'webTitle', 'webUrl', 'pillarName']])
    sql = f''' INSERT INTO student.dabble (id, type, sectionName, webPublicationDate, webTitle, webUrl, pillarName) 
          VALUES (%s, %s, %s, %s, %s, %s, %s) 
          ON CONFLICT (id) DO nothing;
        '''
    try: 
        cur.execute(sql, row_values)
        conn.commit()
    except pg.Error as e: 
        pass

conn.commit()
conn.close()
