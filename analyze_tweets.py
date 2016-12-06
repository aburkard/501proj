import sys
from datetime import datetime
from elasticsearch import Elasticsearch
import certifi
import csv
from textblob import TextBlob
import re

def main(argv):
    get_es_tweets()


def get_es_tweets():
    es = Elasticsearch(
        'https://search-twitter-kinesis-6vxdfwsur57tuf2ddb2mfdcday.us-east-1.es.amazonaws.com',
        port=443,
        use_ssl=True
    )
    # Query syntax to retrieve all tweets within Chicago boundaries
    res = es.search(index="twitter",
                    scroll='2m',
                    search_type='scan',
                    size=1000,
                    body={
                        "query": {
                            "bool": {
                                "must": {
                                    "match_all": {}
                                },
                                "filter": {
                                    "geo_bounding_box": {
                                        "coordinates.coordinates": {
                                            "top_left": {
                                                "lat": 42.023135,
                                                "lon": -87.940101
                                            },
                                            "bottom_right": {
                                                "lat": 41.644286,
                                                "lon": -87.523661
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    })

    # used for scrolling results sets
    sid = res['_scroll_id']
    scroll_size = res['hits']['total']

    output_file = "all_tweets.csv"
    first = True

    with open(output_file, 'w') as f:
        # Start scrolling
        while (scroll_size > 0):
            print("Scrolling...")
            res = es.scroll(scroll_id=sid, scroll='2m')
            # Update the scroll ID
            sid = res['_scroll_id']
            # Get the number of results that we returned in the last scroll
            scroll_size = len(res['hits']['hits'])
            # Output each status in the obtained page
            for hit in res['hits']['hits']:
                # Unnest nested fields
                status = hit['_source']
                status['user_id'] = status['user']['id']
                status['user_name'] = status['user']['name']
                del status['user']
                status['lon'] = status['coordinates']['coordinates'][0]
                status['lat'] = status['coordinates']['coordinates'][1]
                del status['coordinates']
                status['sentiments'] = get_sentiment(status['text'])
                # Write column headers once
                if first:
                    first = False
                    w = csv.DictWriter(f, status.keys(), extrasaction='ignore')
                    w.writeheader()
                w.writerow(status)


class Sentiments:
    POSITIVE = 'Positive'
    NEGATIVE = 'Negative'
    NEUTRAL = 'Neutral'
    CONFUSED = 'Confused'

emoji = {Sentiments.POSITIVE: '😀|😁|😂|😃|😄|😅|😆|😇|😈|😉|😊|😋|😌|😍|😎|😏|😗|😘|😙|😚|😛|😜|😝|😸|😹|😺|😻|😼|😽',
         Sentiments.NEGATIVE: '😒|😓|😔|😖|😞|😟|😠|😡|😢|😣|😤|😥|😦|😧|😨|😩|😪|😫|😬|😭|😾|😿|😰|😱|🙀',
         Sentiments.NEUTRAL: '😐|😑|😳|😮|😯|😶|😴|😵|😲',
         Sentiments.CONFUSED: '😕'
         }


# Determine sentiment from emoji first, then text
def get_sentiment(status):
    sentiment = get_emoji_sentiment(status)
    if sentiment == None:
        sentiment = get_text_sentiment(status)
    return sentiment

def get_text_sentiment(status):
    blob = TextBlob(status)
    sentiment_polarity = blob.sentiment.polarity
    if sentiment_polarity < 0:
        sentiment = Sentiments.NEGATIVE
    elif sentiment_polarity <= 0.2:
        sentiment = Sentiments.NEUTRAL
    else:
        sentiment = Sentiments.POSITIVE
    return sentiment


def get_emoji_sentiment(status):
    sentiments = []
    for sentiment, icons in emoji.items():
        matched_emoji = re.findall(icons, status)
        if len(matched_emoji) > 0:
            sentiments.append(sentiment)

    # Tweets with positive and negative emoji are "confused"
    if Sentiments.POSITIVE in sentiments and Sentiments.NEGATIVE in sentiments:
        return Sentiments.CONFUSED
    elif Sentiments.POSITIVE in sentiments:
        return Sentiments.POSITIVE
    elif Sentiments.NEGATIVE in sentiments:
        return Sentiments.NEGATIVE
    return None


if __name__ == "__main__":
    main(sys.argv)
