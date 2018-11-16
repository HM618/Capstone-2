import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics.pairwise import paired_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.decomposition import LatentDirichletAllocation

# new_stops = ['starring', 'written', 'writer', 'directed', 'series', 'produced', 'released', 'based',
# 			'developed', 'versions', 'version', 'new']
# stop_words = text.ENGLISH_STOP_WORDS.union(new_stops)
stopwords = text.ENGLISH_STOP_WORDS

class Data():
	def __init__(self, file_path, sample_size = 1000):
		self.df = pd.read_csv(file_path)
		self.X = None
		self.sample_size = sample_size
		self.sample = None
		self.tokens = None
		self._clean_data()
		self.get_tokens()

	def _clean_data(self):
		self.X = self.df.dropna(axis=0)
		self.X = self.X[~self.X.text.str.contains('list', case=False)]

	def get_tokens(self):
		# cv = CountVectorizer(max_df=0.85, min_df=2, max_features=1000, stop_words=stop_words)
		self.sample = self.X.iloc[np.random.randint(0,21000, self.sample_size)]
		# cv = cv.fit(self.sample)
		# joblib.dump(cv, 'cv.joblib')
		cv = joblib.load('cv.joblib')
		self.tokens = cv.transform(self.sample.text)
		self.feature_names = cv.get_feature_names()


def get_important_features(lda, features):
	top_feature_idx = lda.components_.argsort(axis=1)[:,-11:-1]
	top_features = np.empty_like(top_feature_idx, dtype='object')
	for i, row in enumerate(top_feature_idx):
		for j, idx in enumerate(row):
			top_features[i,j] = features[idx]
	return top_features

def get_important_articles(lda, X, articles):
	probs = lda.transform(X).T.argsort(axis=1)[:,-11:-1]
	top_articles = np.empty_like(probs, dtype='object')
	for i, row in enumerate(probs):
		for j, idx in enumerate(row):
			top_articles[i,j] = articles[idx]
	return top_articles

def get_distances(lda, X, article_idx):
	prep = lda.transform(X)
	distances = paired_distances(prep - prep[article_idx])
	return distances

def recommend_articles(lda, X, article_idx):
	pass


if __name__ == '__main__':
    data = Data('/Users/haven/galvanize/capstone_2/all_tweets.csv')

    X = data.tokens
    features = data.feature_names

    # lda = LatentDirichletAllocation(n_components=10, learning_method='online', n_jobs=-2)
    # lda.fit(X)
    # joblib.dump(lda, 'lda_model.joblib')
    lda = joblib.load('lda_model.joblib')

    print(get_important_features(lda, features))

    probs = get_important_articles(lda, X, data.sample.text.values)
