from datetime import datetime
from elasticsearch import Elasticsearch
import certifi
import csv

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
            status = hit['_source']
            status['user_id'] = status['user']['id']
            status['user_name'] = status['user']['name']
            del status['user']
            status['lon'] = status['coordinates']['coordinates'][0]
            status['lat'] = status['coordinates']['coordinates'][1]
            del status['coordinates']
            if first:
                first = False
                w = csv.DictWriter(f, status.keys(), extrasaction='ignore')
                w.writeheader()
            w.writerow(status)
