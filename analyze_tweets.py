from datetime import datetime
from elasticsearch import Elasticsearch
import certifi

es = Elasticsearch(
    'https://search-twitter-kinesis-6vxdfwsur57tuf2ddb2mfdcday.us-east-1.es.amazonaws.com',
    port=443,
    use_ssl=True
)

doc = {
    # 'author': 'kimchy',
    'text': 'test',
    # 'timestamp': datetime.now(),
}
res = es.get(index="twitter", doc_type='tweet', id=796662644161683456)
print(res['_source'])

res = es.search(index="twitter", body={"query": {"match_all": {}}})
print("Got %d Hits:" % res['hits']['total'])
#for hit in res['hits']['hits']:
#    print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])
