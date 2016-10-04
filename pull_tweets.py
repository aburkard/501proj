import sys
import tweepy
from tweepy import OAuthHandler
import json
import csv
import configparser
import pandas as pd
import numpy as np

def main(argv):
    
def pull_tweets(auth):
    config = configparser.ConfigParser()
    config.read('config.ini')

    num_tweets = config['twitter']['num_tweets']
    output_file = config['twitter']['output_file']
    coordinates = config['twitter']['coordinates']

    consumer_key = config['twitter_auth']['consumer_key']
    consumer_secret = config['twitter_auth']['consumer_secret']
    access_token = config['twitter_auth']['access_token']
    access_secret = config['twitter_auth']['access_secret']

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    with open(output_file, 'w') as f:
        for status in tweepy.Cursor(api.search,  geocode=coordinates).items(1):
            w = csv.DictWriter(f, status._json.keys(), extrasaction='ignore')
            w.writeheader()
        for status in tweepy.Cursor(api.search,  geocode=coordinates).items(num_tweets):
            handle_status(status._json, w)

    def handle_status(status, w):
        w.writerow(status)
        #print(json.dumps(status))


def analyze_twitter_data(file_name):
    df = pd.read_csv(file_name , sep=',', encoding='utf-8')

if __name__ == "__main__":
    main(sys.argv)