# 501proj


Note the Twitter script requires the tweepy package:
	
	pip install tweepy

Twitter API settings are defined in config.ini. We've included an API key for a dummy Twitter account. To pull tweets, generate new attributes, and output cleanliness data just run: 
    
    py pull_tweets.py 

For the Chicago crime data:
    
    py chicago_crime.py


ChicagoCrime2016W3newfeatures.tsv contains the Chicago crime data with additional features. Note that this is a tab seperated file, since many of the fields contain commas and possibly quotes.

tweets_10k_augmented.csv contains the Twitter data with additional features.