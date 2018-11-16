import emoji
from imblearn.over_sampling import SMOTE
import itertools
import matplotlib.pyplot as plt
import missingno as msno
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn import manifold
from sklearn.datasets import make_blobs
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation, printable
from subprocess import call
#from fancyimpute import SimpleFill, KNN,  IterativeSVD, IterativeImputer
#from wordcloud import WordCloud, STOPWORDS

nlp = spacy.load('en')

all_tweets = pd.read_csv('/Users/haven/galvanize/capstone_2/all_tweets.csv',
                                    dtype={'Unnamed': int, 'text': object,
                                    'source': object,'user_id': float,
                                    'truncated': float, 'in_reply_to_status_id': float,
                                    'in_reply_to_user_id': float,
                                    'in_reply_to_screen_name': object,
                                    'retweeted_status_id': float, 'geo': float,
                                    'place': object,'contributors': float,
                                    'retweet_count': float, 'reply_count': float,
                                    'favorite_count': float, 'favorited': float,
                                    'retweeted': float, 'possibly_sensitive': float,
                                    'num_hashtags': float, 'num_urls': float,
                                    'num_mentions': float,'label': int})

all_tweets.drop(['truncated', 'in_reply_to_screen_name', 'geo', 'place', 'contributors', 'favorited', 'retweeted', 'possibly_sensitive'], inplace=True, axis=1)

tweets_only = all_tweets[['text','label']]
tweets_only.to_csv('tweets_only.csv')

y = all_tweets['label']
X = all_tweets[['Unnamed: 0', 'user_id', 'in_reply_to_status_id',
       'in_reply_to_user_id', 'retweeted_status_id', 'retweet_count',
       'reply_count', 'favorite_count', 'num_hashtags', 'num_urls',
       'num_mentions', 'label']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)




def clean_data(dataframe,col):
    '''
    Removes punctuation and import errors from dataframe

    INPUT:  DataFrame (df)
            Column name to clean (string)

    OUTPUT: Cleaned DataFrame Column (series)

    '''
    punctuation = '!"$%&()*+,-./:;<=>?[\]^_`{|}~'
    import_errors = ['_„Ž','_„ñ','_ã_','_„','Ã±','ñ','ð']
    df2 = dataframe.copy()
    for e in import_errors:
        df2[col] = df2[col].str.replace(e,'')
    for p in punctuation:
        df2[col] = df2[col].str.replace(p,'')
    return df2[col]

tweets_only = pd.read_csv('tweets_only.csv')

def extract_emojis(str):
  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

emojis_list = []
for line in tweets_only:
    for char in line:
        if char in emoji.UNICODE_EMOJI:
            emojis_list.append(char)
        else:
            pass


def lemmatize_string(doc, stop_words=STOP_WORDS):
    """
    takes a list of strings where each string is a document
    returns a list of strings
    """

    if not stop_words:
        stop_words = []

    # remove unicode
    clean_doc = "".join([char for char in doc if char in printable])

    # Run the doc through spaCy
    doc = nlp(clean_doc)

    # Lemmatize and lower text
    tokens = [re.sub("\W+","",token.lemma_.lower()) for token in doc ]
    tokens = [t for t in tokens if len(t) > 1]

    return ' '.join(w for w in tokens if w not in stop_words)

corpus = [lemmatize_string(line) for line in tweets_only['text']]
#
print("lemmatized words loaded")
df['corpus'] = corpus
#
# print("...extracting tf-idf features")
# stop_words = ['the','a','and','an','so']
# n_features = 1000
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
#                                    max_features=n_features,
#                                    stop_words=stop_words,ngram_range=(1,2))
# tfidf = tfidf_vectorizer.fit_transform(corpus).todense()
# tfidf_arr = np.asarray(tfidf)
# tfidf_feature_names = tfidf_vectorizer.get_feature_names()
# print(tfidf_arr.shape)

def extract_emojis(str):
  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

corpus = [extract_emojis(line) for line in all_tweets['text']]
all_tweets['corpus'] = corpus

twitter_dict = {'#' : '<hashtag>', '@' : '<attention>', 'http://' : '<hypertext>'}

def translate_message(list):
    for str in list:
        for word in twitter_dict:
            if word in str:
                str = str.replace(word, twitter_dict[word])
    return list

def extract_emojis(a_list):
    emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
    r = re.compile('|'.join(re.escape(p) for p in emojis_list))
    aux=[' '.join(r.findall(s)) for s in a_list]
    return(aux)




#language = gen_use.groupby('lang')

# plot tweets by language
# plt.figure(figsize=(15,10))
# language.size().sort_values(ascending=False).plot.bar()
# plt.xticks(rotation=50)
# plt.xlabel("Lanuage")
# plt.ylabel("Number of Tweets")
#
#
def make_wordcloud(data):
    stopwords = set(STOPWORDS)
    # iterate through the csv file
    for val in all_tweets.text:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

    comment_words = ' '
    for i in range(5000):
        comment_words+=(np.random.choice(all_tweets.text.values, replace=False))


    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(comment_words)

    # plot the WordCloud image
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    plt.show()


# genuine: (2839362, 25); retweeted, reply_count, favorite_count, num_hahstags, num_urls, num_mentioned, created_at, text

# spam: (1610034, 25); retweeted,favorite_count, num_hahstags, num_urls, num_mentioned, created_at


# trad_spam_tweets = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/traditional_spambots_1.csv/tweets.csv')
# # shape 145094, 25)
# trad_spam_users = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/traditional_spambots_1.csv/users.csv')
# # shape (1000, 40)
