# 501proj

The project report is available at https://sites.google.com/a/georgetown.edu/chi-crime-twitter/

The project is organized into folders:
- `scripts` contains all of them python scripts and supporting files
- `data` is initially empty and is used by the scripts for file I/O
- `viz` contains visualization screenshots. Live interactive versions are available on the project report site.
- `output` contains some script console output
- `data_archive` contains the resulting output from running the scripts.

Note the scripts requires the following packages installed:
```
	pip install tweepy
	pip install elasticsearch
	pip install certifi
	pip install TextBlob
	pip install gensim
	pip install fim
```
To run the entire process yourself:
```
python pull_tweets.py
python analyze_tweets.py
python tweets_lda.py
python ChicagoCrimeDataCombined_2016_10_04.py
python chicago_crime.py
python CaryLou_projectpart2_chidata.py
python donut_chart.py
```
It's possible that you may have to install additional nltk packages using nltk.download()

Twitter API settings are defined in config.ini. We've included an API key for a dummy Twitter account. To pull tweets, generate new attributes, and output cleanliness data just run:

    py pull_tweets.py

For the Chicago crime data:

    py chicago_crime.py


ChicagoCrime2016W3newfeatures.tsv contains the Chicago crime data with additional features. Note that this is a tab seperated file, since many of the fields contain commas and possibly quotes.

tweets_10k_augmented.csv contains the Twitter data with additional features.

All interative visuals except the donut chart, which was created with Bokeh, were created using Tableau.
