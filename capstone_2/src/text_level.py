from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

text = pd.read_csv('/Users/haven/galvanize/capstone_2/tweets_only.csv')

text['totalwords'] = text['text'].str.split().str.len()

text['hashtags'] = text['text'].str.count('#')
text['attention'] = text['text'].str.count('@')

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

text['text'] = clean_data(text, 'text')
text.text = text.text.astype(str).str.lower()

text.to_csv('text_info.csv')

corpus = text['text']
#
# words = []
# tokenized_words = []
#
# for doc in corpus:
#     words += doc.split()
#     tokenized_words += word_tokenize(str(words))

def make_wordcloud(data):
    stopwords = set(STOPWORDS)
    # iterate through the csv file
    for val in text.text:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

    comment_words = ' '
    for i in range(5000):
        comment_words+=(np.random.choice(text.text.values, replace=False))


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
