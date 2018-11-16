import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.metrics import mean_squared_error
#from fancyimpute import SimpleFill, KNN,  IterativeSVD, IterativeImputer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

genuine_tweets = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/tweets.csv')
# (2839362, 25); retweeted, reply_count, favorite_count, num_hahstags, num_urls, num_mentioned, created_at
genuine_users = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/users.csv')
# (3474, 42); statuses_count, followers_count, friends_count, favorites_count, listed_count, default_profile, geo_enables, profile_users_background_image, verified, protected

soc_spam_tweets = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/social_spambots_1.csv/tweets.csv')
# shape (1610034, 25); retweeted,favorite_count, num_hahstags, num_urls, num_mentioned, created_at
soc_spam_users = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/social_spambots_1.csv/users.csv')
# shape (991, 41)

trad_spam_tweets = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/traditional_spambots_1.csv/tweets.csv')
# shape 145094, 25)
trad_spam_users = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/traditional_spambots_1.csv/users.csv')
# shape (1000, 40)

#spam_words = ' '.join(list(soc_spam_tweets[]))
