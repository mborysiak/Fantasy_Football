#%%
import yaml
import pytz
import requests
import pandas as pd 
import numpy as np
import datetime as dt

from ff.db_operations import DataManage
from ff import general
import ff.data_clean as dc

# set to this year
year = 2025

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

import warnings
from scipy.stats import poisson, truncnorm
from typing import Dict

with open(f'{root_path}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
api_key = config['odds_api_key']
sport = 'americanfootball_nfl'
region = 'us2' 
odds_format = 'decimal' 
date_format = 'iso'

#%%
class OddsAPIPull:

    def __init__(self, year, api_key, base_url, sport, region, odds_format, date_format, historical=False):
        
        self.year = year
        self.api_key = api_key
        self.sport = sport
        self.region = region
        self.odds_format = odds_format
        self.date_format = date_format
        self.historical = historical

        if self.historical: 
            self.base_url = f'{base_url}/historical/sports/'
        else: 
            self.base_url = f'{base_url}/sports/'

    def get_response(self, r_pull):
        if r_pull.status_code != 200:
            print(f'Failed to get odds: status_code {r_pull.status_code}, response body {r_pull.text}')
        else:
            r_json = r_pull.json()
            print('Number of events:', len(r_json))
            print('Remaining requests', r_pull.headers['x-requests-remaining'])
            print('Used requests', r_pull.headers['x-requests-used'])

        return r_json

    @staticmethod
    def convert_utc_to_est(est_dt):
        
        # Define the EST timezone
        est = pytz.timezone('US/Eastern')

        # Localize the datetime object to EST
        local_time_est = est.localize(est_dt)

        # Convert the localized datetime to UTC
        utc_time = local_time_est.astimezone(pytz.utc)
        
        return utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def get_weekday_name(date_string):
        dt_utc = dt.datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
        dt_utc = dt_utc.replace(tzinfo=pytz.UTC)

        # Convert to EST
        est_tz = pytz.timezone('America/New_York')
        dt_est = dt_utc.astimezone(est_tz)

        # Get the day name in EST
        weekday_name = dt_est.strftime("%A")
        return weekday_name

    def pull_events(self, start_time, end_time):
        
        if start_time is not None: start_time = self.convert_utc_to_est(start_time)
        if end_time is not None: end_time = self.convert_utc_to_est(end_time)

        get_params = {
                'api_key': self.api_key,
                'regions': self.region,
                'oddsFormat': self.odds_format,
                'dateFormat': self.date_format,
                'commenceTimeFrom': start_time,
                'commenceTimeTo': end_time
            }
        
        if self.historical:
            get_params['date'] = start_time
            self.start_time = start_time

        events = requests.get(
            f'{self.base_url}/{self.sport}/events',
            params=get_params
            )
        
        events_json = self.get_response(events)
        if self.historical: events_json = events_json['data']

        events_df = pd.DataFrame()
        for e in events_json:
            events_df = pd.concat([events_df, pd.DataFrame(e, index=[0])], axis=0)
        
        events_df['year'] = self.year
        events_df['day_of_week'] = events_df.commence_time.apply(self.get_weekday_name)
        events_df = events_df.rename(columns={'id': 'event_id'})

        return events_df.reset_index(drop=True)
    

    def pull_lines(self, markets, event_id):

        get_params={
                    'api_key': self.api_key,
                    'regions': self.region,
                    'markets': markets,
                    'oddsFormat': self.odds_format,
                    'dateFormat': self.date_format,
                }
       
        if self.historical:
            get_params['date'] = self.start_time

        odds = requests.get(
                f'{self.base_url}/{self.sport}/events/{event_id}/odds',
                params = get_params
            )
        
        odds_json = self.get_response(odds)
        if self.historical: odds_json = odds_json['data']

        props = pd.DataFrame()
        for odds in odds_json['bookmakers']:
            bookmaker = odds['key']
            market_props = odds['markets']
            for cur_prop in market_props:
                p = pd.DataFrame(cur_prop['outcomes'])
                p['bookmaker'] = bookmaker
                p['prop_type'] = cur_prop['key']
                p['event_id'] = event_id

                if cur_prop['key'] in ('spreads', 'h2h'):
                    p = p.rename(columns={'name': 'description'})

                props = pd.concat([props, p], axis=0)
                

        props = props.reset_index(drop=True)
        props['year'] = self.year

        return props

    def all_market_odds(self, markets, events_df):

        props = pd.DataFrame()
        for event_id in events_df.event_id.values:
            try:
                print(event_id)
                cur_props = self.pull_lines(markets, event_id)
                props = pd.concat([props, cur_props], axis=0)
            except:
                print(f'Failed to get data for event_id {event_id}')

        return props

# %%

set_year = 2025
pull_historical = False

base_url = 'https://api.the-odds-api.com/v4/'
odds_api = OddsAPIPull(set_year, api_key, base_url, sport, region, odds_format, date_format, historical=pull_historical)

start_time = dt.datetime.now()
end_time = (start_time + dt.timedelta(hours=70*24))

events_df = odds_api.pull_events(start_time=start_time, end_time=end_time)
event_ids = tuple(events_df.event_id.unique()) + (0,)
events_df

# %%
r_pull = requests.get(
    f'{odds_api.base_url}/{sport}/odds',
    params={
        'api_key': odds_api.api_key,
        'regions': odds_api.region,
    }
)

odds_api.get_response(r_pull)
# %%