import sys
import tweepy
from tweepy import OAuthHandler
import json
import csv
import configparser
import pandas as pd
import numpy as np

def main(argv):
    output_file_name = pull_tweets()
    generate_features(output_file_name, output_file_name)
    analyze_twitter_data(output_file_name)

def pull_tweets():
    config = configparser.ConfigParser()
    config.read('config.ini')

    num_tweets = int(config['twitter']['num_tweets'])
    output_file = config['twitter']['output_file']
    coordinates = config['twitter']['coordinates']

    consumer_key = config['twitter_auth']['consumer_key']
    consumer_secret = config['twitter_auth']['consumer_secret']
    access_token = config['twitter_auth']['access_token']
    access_secret = config['twitter_auth']['access_secret']

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def handle_status(status, w):
        w.writerow(status)
        #print(json.dumps(status))

    with open(output_file, 'w') as f:
        for status in tweepy.Cursor(api.search,  geocode=coordinates).items(1):
            w = csv.DictWriter(f, status._json.keys(), extrasaction='ignore')
            w.writeheader()
        for status in tweepy.Cursor(api.search,  geocode=coordinates).items(num_tweets):
            handle_status(status._json, w)

    
    return output_file

def generate_features(input_file, output_file):
    df = pd.read_csv(input_file , sep=',', encoding='utf-8')
    lat = df["geo"].apply(lambda x:  eval(x)['coordinates'][0] if pd.notnull(x) else np.nan)
    lon = df["geo"].apply(lambda x:  eval(x)['coordinates'][1] if pd.notnull(x) else np.nan)
    user_id = df["user"].apply(lambda x:  eval(x)['id_str'] if pd.notnull(x) else np.nan)
    df['lat']=lat
    df['lon']=lon
    df['user_id'] = user_id
    df.to_csv(output_file , sep=',', encoding='utf-8')

def analyze_twitter_data(file_name):
    df = pd.read_csv(file_name , sep=',', encoding='utf-8')
    null_percentages = df.isnull().sum()/len(df.index)
    print("Percentage of null values:\n{}".format(null_percentages))

if __name__ == "__main__":
    main(sys.argv)