
# coding: utf-8

# In[3]:


import re
import pandas as pd

# function to scrape player and team data
def data_load(year_start, year_end, url, read_list):

    df = pd.DataFrame()
    years = reversed(range(year_start, year_end+1))

    for year in years:
        year = str(year)
        url_year = url.format(year)
        f = pd.read_html(url_year, header=0)
        f[read_list]['year'] = year
        df = df.append(f)
    df = df.reset_index(drop=True)
    return df


# function to format player statistical data
def player_format(data):
    data = data.rename(columns={'Unnamed: 1': 'Player'})
    col_order = ['Player', 'Tm', 'Pos', 'year', 'Age', 'Ctch%', 'Fmb', 'G', 'GS', 'Lng',
             'R/G', 'Rec','TD', 'Tgt', 'Y/G', 'Y/R', 'Yds']
    df = data[col_order]
    df = df[pd.notnull(df['Player'])]
    return df


# removing un-needed characters
def name_clean(col):
    char_remove = re.sub('[\*\+\%\,]', '', str(col))
    return char_remove


# selecting first and last names from adp data
def name_select(col):
    first = col.split(' ')[1]
    last = col.split(' ')[0]
    new_col = ' '.join([first,last])
    return new_col


# select team name from adp data
def team_select(col):
    team = col.split(' ')[2]
    return team

# function to append data to sqlite
def append_to_db(df, db_name='Season_Stats.sqlite3', table_name='NA', if_exist='append'):

    import sqlite3
    import os
    
    #--------
    # Append pandas df to database in Github
    #--------

    os.chdir('/Users/Mark/Documents/Github/Fantasy_Football/Data/')

    conn = sqlite3.connect(db_name)

    df.to_sql(
    name=table_name,
    con=conn,
    if_exists=if_exist,
    index=False
    )

    #--------
    # Append pandas df to database in Dropbox
    #--------

    os.chdir('/Users/Mark/Dropbox/FF/')

    conn = sqlite3.connect(db_name)

    df.to_sql(
    name=table_name,
    con=conn,
    if_exists=if_exist,
    index=False
    )
