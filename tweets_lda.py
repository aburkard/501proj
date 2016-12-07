import sys
import numpy as np
import lda
import lda.datasets
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import pandas as pd
import numpy as np
from operator import itemgetter

# Global vars
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def main(argv):
    df = pd.read_csv("all_tweets.csv", sep=',', encoding='utf-8')
    # Save cleaned tweet text to eventual output
    df["cleaned_text"] = df["text"].apply(lambda x: clean(x).split())
    run_lda(df["cleaned_text"].tolist(), df)


def run_lda(docs, df):
    # Convert documents to term-document matrix
    dictionary = corpora.Dictionary(docs)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]

    # Create a model with 20 topics
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=20,
                   id2word=dictionary, iterations=500, passes=5)
    print(ldamodel.print_topics(num_topics=20, num_words=10))

    # Get top topic for each tweet
    df["topics"] = df["cleaned_text"].apply(
        lambda x: ldamodel[dictionary.doc2bow(x)])
    df["top_topic"] = df["topics"].apply(
        lambda x: max(x, key=itemgetter(1))[0])
    df["top_score"] = df["topics"].apply(
        lambda x: max(x, key=itemgetter(1))[1])
    df.to_csv("tweets_with_topics.csv", sep=',', encoding='utf-8')


# Remove stopwords, punctuation, hyperlinks, and do some basic stemming
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    http_free = " ".join(word for word in normalized.split()
                         if not word.startswith('http'))
    return http_free


if __name__ == "__main__":
    main(sys.argv)
