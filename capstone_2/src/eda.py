from imblearn.over_sampling import SMOTE
import itertools
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, classification_report, confusion_matrix, mean_squared_error, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz, DecisionTreeClassifier, DecisionTreeRegressor
from subprocess import call
from wordcloud import WordCloud, STOPWORDS

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

genuine_users = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/users.csv')
# (3474, 42); statuses_count, followers_count, friends_count, favorites_count, listed_count, default_profile, geo_enables, profile_users_background_image, verified, protected
genuine_users['label'] = int(0)
#gen_use = genuine_users[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'default_profile', 'lang', 'profile_use_background_image']]
# shape (2839362, 8)
#gen_use['label'] = int(0)
soc_spam_users = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/social_spambots_1.csv/users.csv')
# shape (991, 41)
soc_spam_users['label'] = int(1)
#soc_use = soc_spam_users[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'default_profile', 'lang', 'profile_use_background_image']]
#soc_use['label'] = int(1)
full_data = genuine_users.merge(soc_spam_users, how='outer')
'''
'id', 'name', 'screen_name', 'statuses_count', 'followers_count',
'friends_count', 'favourites_count', 'listed_count', 'url', 'lang',
'time_zone', 'location', 'default_profile',
'geo_enabled', 'profile_image_url', 'profile_banner_url',
'profile_use_background_image', 'profile_background_image_url_https',
'profile_text_color', 'profile_image_url_https',
'profile_sidebar_border_color', 'profile_background_tile',
'profile_sidebar_fill_color', 'profile_background_image_url',
'profile_background_color', 'profile_link_color', 'utc_offset',
'is_translator', 'follow_request_sent', 'protected', 'verified',
'notifications', 'description', 'contributors_enabled', 'following',
'created_at', 'timestamp', 'crawled_at', 'updated'
'''
#full_data = gen_use.merge(soc_use,how='outer')
full_data.drop('default_profile', inplace=True, axis=1)
full_data = full_data.fillna('0')
genuine_tweets = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/tweets.csv')
#gen_tw = genuine_tweets[['reply_count', 'text', 'favorite_count', 'num_hashtags', 'num_urls', 'num_mentions','created_at']]
#
# # create new csv file for content of tweets
# genuine_tweets = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/tweets.csv')
# genuine_tweets.drop(['created_at','timestamp', 'crawled_at', 'updated', 'id'], inplace=True, axis=1)
# genuine_tweets['label'] = int(0)
spam_tweets = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/social_spambots_1.csv/tweets.csv')
trad_spam_tweets = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/traditional_spambots_1.csv/tweets.csv')
bot_tweets = pd.concat([spam_tweets, trad_spam_tweets])
# spam_tweets.drop(['created_at','timestamp', 'crawled_at', 'updated', 'id'], inplace=True, axis=1)
# spam_tweets['label'] = int(1)
# all_tweets= genuine_tweets.merge(spam_tweets, how='outer')
# all_tweets.to_csv('all_tweets.csv')
describe = genuine_tweets['text']
describe2 = bot_tweets['text']

def make_wordcloud(data):
    comment_words = ' '
    # iterate through the csv file
    for val in data:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

    for words in tokens:
        comment_words = comment_words + words + ' '


    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
#
                    min_font_size = 10).generate(comment_words)

    # plot the WordCloud image
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    plt.show()


#split data into features and target
y = full_data['label']
X = full_data[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'profile_use_background_image']] #'lang',
# X = full_data2[['statuses_count', 'followers_count',
# 'friends_count', 'favourites_count', 'listed_count',
# 'geo_enabled',
# 'profile_use_background_image',
# 'profile_text_color',
# 'profile_sidebar_border_color', 'profile_background_tile',
# 'profile_sidebar_fill_color',
# 'profile_background_color', 'profile_link_color', 'utc_offset',
# 'is_translator', 'follow_request_sent', 'protected', 'verified',
# 'notifications', 'contributors_enabled', 'following']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#after evidence of severe imbalance, implemented SMOTE* to inject a dose of homeostasis
sm = SMOTE(random_state=2)
# *Synthetic Minority Over-sampling Technique*
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`. Play around!
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1
        #print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def logistic_model(X_train_res, y_train_res):
    '''
    Takes partitioned data and returns a logistic regression model.
    Note, in this function we also perform a GridSearch to gain additional insight.
    '''
    parameters = {'C': np.linspace(1, 10, 10)}
    logistic = LogisticRegression()
    #performs a gridsearch to obtain top features, parameters, etc
    clf = GridSearchCV(logistic, parameters, cv=5, verbose=5, n_jobs=5)
    clf.fit(X_train_res, y_train_res.ravel())
    logistic2 = LogisticRegression(C=4,penalty='l1', verbose=5)
    logistic2.fit(X_train_res, y_train_res.ravel())
    # clf.best_params_ = {'C': 10.0}
    clf.get_params
    y_train_pre = logistic2.predict(X_train_res)
    cnf_matrix_tra = confusion_matrix(y_train_res, y_train_pre)
    print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))
#    Recall metric in the train dataset: 97.37
    return logistic2


def running_random(X_train_res, y_train_res,
                             X_test, y_test):
    '''
    Takes partitioned data and returns a random forest classifier model.
    '''
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train_res, y_train_res)
    y_predict = random_forest.predict(X_train_res)
    cnf_rfmatrix_tra = confusion_matrix(y_train_res, y_predict)
    # Recall metric in the train dataset: 98.36
    return random_forest

# # make confusion matrix with each model
# class_names = [0,1]
# plt.figure()
# plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='{}% Confusion matrix'.format(random_forest))
# plt.savefig('Account Confusion Matrix')
# plt.show()

def plot_roc():
    tmp = random_forest.fit(X_train_res, y_train_res.ravel())
    y_pred_sample_score = tmp.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score)
    roc_auc = auc(fpr,tpr)
    # Plot ROC
    plt.title('ROC of Random Forest')
    plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('ROC of Random Forest')
    plt.show()

def initial_metrics(X_train, X_test,
                    y_train, y_test):
    '''
    This function takes partitions of a dataset and
    prints its shapes.
    '''
    print("Shape of X_train dataset: ", X_train.shape)
    print("Shape of y_train dataset: ", y_train.shape)
    print("Shape of X_test dataset: ", X_test.shape)
    print("Shape of y_test dataset: ", y_test.shape)


def smote_metrics(X_train, y_train,
                        X_train_res, y_train_res):
    '''
    This function takes partitioned data before and after implenting smote
    and prints its metrics using 'before' and 'after' statements.
    '''
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
    print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
    print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


def print_scores(model, X, y_test, predict):
    cv_score = cross_val_score(model, X, y_test)
    print("=== Confusion Matrix {}% ===".format(model))
    print(confusion_matrix(y_test, predict))
    print('\n')
    print("=== Classification Report {}% ===".format(model))
    print(classification_report(y_test, predict))
    print('\n')
    print("=== All AUC Scores {}% ===".format(model))
    print(rf_cv_score)
    print('\n')
    print("=== Mean AUC Score {}% ===".format(model))
    print("Mean AUC Score", cv_score.mean())


def roc_curve(model, X, y_test, predict):
    roc_auc = roc_auc_score(y_test, predict)
    cv_score = cross_val_score(model, X, y)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Roc Auc After Reshape' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('{}% ROC After Reshape'.format(model))
    plt.show()

#sns.countplot(x=y, data=full_data, palette='hls')

#plt.savefig('Counting Followers')
#plt.show()
# gen_txt = list(gen_tw.text)
# vectorizer = TfidfVectorizer()
# matrix = vectorizer.fit_transform(gen_tw['text'].values.astype('U'))
# #shape (2839362, 1713580)
# text = vectorizer.gen_tw['text']
# features = text.get_feature_names()
# print(matrix)
