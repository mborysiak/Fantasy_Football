
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

def convert_to_float(df):
    for c in df.columns:
        try:
            df[c] = df[c].astype('float')
        except:
            pass
    
    return df

# function to append data to sqlite
def append_to_db(df, db_name='Season_Stats.sqlite3', table_name='NA', if_exist='append'):

    import sqlite3
    import os
    import datetime as dt
    
    #--------
    # Append pandas df to database in Github
    #--------

    os.chdir('/Users/Mark/Documents/Github/Fantasy_Football/Data/Databases/')
    conn = sqlite3.connect(db_name)

    #--------
    # Save Backup
    #--------
    
    backup = pd.read_sql_query(f'''SELECT * FROM {table_name}''', conn)

    df.to_sql(
    name=table_name,
    con=conn,
    if_exists=if_exist,
    index=False
    )

    #--------
    # Append pandas df to database in Dropbox
    #--------

    os.chdir('/Users/Mark/OneDrive/FF/DataBase/')

    conn = sqlite3.connect(db_name)

    df.to_sql(
    name=table_name,
    con=conn,
    if_exists=if_exist,
    index=False
    )
    
    backup_date = dt.datetime.now().strftime('%Y%m%d%H%M')
    backup_path_ext = f'Backups/{db_name.replace('.sqlite3', '')}_{table_name}_{backup_date}.csv'
    df.to_csv(f'/Users/Mark/OneDrive/FF/DataBase/{backup_path_ext}', index=False)
    
#==========
# Clean the ADP data
#==========

'''
Cleaning the ADP data by selecting relevant features, and extracting the name and team
from the combined string column. Note that the year is not shifted back because the 
stats will be used to calculate FP/G for the rookie in that season, but will be removed
prior to training. Thus, the ADP should match the year from the stats.
'''

def clean_adp(data_adp, year):

    #--------
    # Select relevant columns and clean special figures
    #--------

    data_adp['year'] = year

    # set column names to what they are after pulling
    df_adp = data_adp.iloc[:, 1:].rename(columns={
        1: 'Player', 
        2: 'Avg. Pick',
        3: 'Min. Pick',
        4: 'Max. Pick',
        5: '# Drafts Selected In'
    })

    # selecting relevant columns and dropping na
    df_adp = df_adp[['Player', 'year', 'Avg. Pick']].dropna()

    # convert year to float and move back one year to match with stats
    df_adp['year'] = df_adp.year.astype('float')

    # selecting team and player name information from combined string
    df_adp['Tm'] = df_adp.Player.apply(team_select)
    df_adp['Player'] = df_adp.Player.apply(name_select)
    df_adp['Player'] = df_adp.Player.apply(name_clean)

    # format and rename columns
    df_adp = df_adp[['Player', 'Tm', 'year', 'Avg. Pick']]

    colnames_adp = {
        'Player': 'player',
        'Tm': 'team',
        'year': 'year',
        'Avg. Pick': 'avg_pick'
    }

    df_adp = df_adp.rename(columns=colnames_adp)
    
    return df_adp