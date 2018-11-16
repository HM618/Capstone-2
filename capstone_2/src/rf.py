import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.metrics import auc, classification_report, confusion_matrix, mean_squared_error, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pdb

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

genuine_users = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/users.csv')
# (3474, 42); statuses_count, followers_count, friends_count, favorites_count, listed_count, default_profile, geo_enables, profile_users_background_image, verified, protected
genuine_users['label'] = int(0)

soc_spam_users = pd.read_csv('/Users/haven/galvanize/capstone_2/cresci-2017.csv/datasets_full.csv/social_spambots_1.csv/users.csv')
# shape (991, 41)
soc_spam_users['label'] = int(1)

full_data = genuine_users.merge(soc_spam_users, how='outer')

full_data.drop('default_profile', inplace=True, axis=1)
full_data = full_data.fillna('0')
dummy_lang = pd.get_dummies(full_data['lang'])
dummy_geo =pd.get_dummies(full_data['time_zone'])
full_data = pd.concat([full_data, dummy_geo, dummy_lang], axis=1)

y = full_data['label']
X = full_data[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'profile_use_background_image', 'Abu Dhabi',  'Alaska',  'Almaty',  'America/Chicago','America/Los_Angeles',  'America/Port_of_Spain',  'Amsterdam',  'Arizona','Asia/Calcutta',  'Asia/Karachi',  'Asia/Manila',  'Athens',  'Atlantic Time (Canada)','Auckland',  'Azores',  'Baghdad',  'Baku',  'Bangkok',  'Beijing',  'Belgrade',  'Berlin','Bogota',  'Brasilia',  'Brisbane',  'Brussels',  'Bucharest',  'Buenos Aires',  'CST',  'Cairo','Caracas',  'Casablanca',  'Central America',  'Central Time (US & Canada)',  'Chennai','Chongqing',  'Copenhagen',  'Dhaka',  'Dublin',  'Eastern Time (US & Canada)', 'Edinburgh',  'Europe/Belgrade',  'Fiji',  'Georgetown',  'Greenland',  'Guadalajara',  'Guam','Hawaii',  'Helsinki',  'Hong Kong',  'International Date Line West',  'Irkutsk','Islamabad',  'Istanbul',  'Jakarta',  'Jerusalem',  'Kabul',  'Karachi',  'Kathmandu', 'Kolkata',  'Krasnoyarsk',  'Kuala Lumpur',  'Kuwait',  'Kyiv',  'La Paz',  'Lima',  'Lisbon',
'Ljubljana',  'London',  'Madrid',  'Magadan',  'Mazatlan',  'Melbourne',  'Mexico City',
'Mid-Atlantic',  'Midway Island',  'Moscow',  'Mountain Time (US & Canada)',  'Mumbai',
'New Delhi',  "Nuku'alofa",  'Osaka',  'Pacific Time (US & Canada)',  'Paris',  'Perth',
'Pretoria',  'Quito',  'Rome',  'Samoa',  'Santiago',  'Sarajevo',  'Seoul',  'Singapore',  'Sofia',
'Stockholm',  'Sydney',  'Taipei','Tallinn',  'Tehran',  'Tijuana',  'Tokyo', 'Urumqi', 'Vienna',
'Wellington','West Central Africa','Yakutsk','ar','da','de',
'el','en', 'en-AU','en-GB','en-gb','es','fil','fr','id','it','ja','ko','nl','pl','pt','ru',
'sv', 'tr',  'xx-lc',  'zh-TW',  'zh-tw']] #'lang',


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#after evidence of severe imbalance, implemented SMOTE* to inject a dose of homeostasis
sm = SMOTE(random_state=2)
# *Synthetic Minority Over-sampling Technique*
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())


def to_booleans(x):
    true_list =['yes','True','True.']
    false_list = ['no', 'False', 'False.']
    if x in true_list:
        return True
    elif x in false_list:
        return False
    else:
        return x

data = full_data.applymap(to_booleans)


forest = RandomForestClassifier()
forest.fit(X,y)
score = forest.score(X_test,y_test)
#print(score)
#0.9952038369304557

y_predict = forest.predict(X_test)

forest_matrix = confusion_matrix(y_test, y_predict)
#[[706   0]
# [  4 124]]

precision = precision_score(y_test, y_predict)
#1.0
recall = recall_score(y_test, y_predict)
#0.96

forest2 = RandomForestClassifier(n_estimators=20, oob_score=True)
forest2.fit(X,y)
score2 = forest2.score(X_train_res,y_train_res)
# score2 0.9975990396158463
# forest.oob_score_ produced  0.9414941494149415

reg_forest = RandomForestRegressor(n_estimators=20)
reg_forest.fit(X_train_res, y_train_res)

#feature_names = data.drop('Churn?', axis=1).columns

def making_plots(y):
    plt.figure(figsize= (15, 10))
    plt.bar(feature_names, forest2.feature_importances_)
    plt.title('Features Importances')
    plt.ylabel(':)')
    plt.xlabel('Features')


#returned array([0.03194009, 0.00673192, 0.07516007, 0.02612099, 0.03539164])

feature_importances = zip(data.columns, forest2.feature_importances_)

def modify_trees(X_train_res, y_train_res, X_test, y_test, numbers):
    tree_count = []
    score_list = []
#    pdb.set_trace()
    for number in numbers:
        tempforest = RandomForestClassifier(n_estimators=number)
        tempforest.fit(X_train_res, y_train_res)
        tempscore = tempforest.score(X_test, y_test)
        tree_count.append(number)
        score_list.append(tempscore)
    return [tree_count,score_list]

accuracy_points = modify_trees(X_train_res, y_train_res, X_test, y_test, [10, 50, 100, 500, 1000])

def plot_accurate(variable):
    plt.figure(figsize=(15,10))
    plt.plot(accuracy_points[0], accuracy_points[1], 'o-')
    plt.ylabel('accuracy')
    plt.xlabel('tree_count')
    #plt.show()

def feature_play(X_train_res, y_train_res, X_test, y_test, feature_num=18):
    feat_count = []
    score_list = []
#    pdb.set_trace()
    for number in range(1,feature_num+1):
        tempforest = RandomForestClassifier(n_estimators=20, max_features=number)
        tempforest.fit(X_train_res, y_train_res)
        tempscore = tempforest.score(X_test, y_test)
        feat_count.append(number)
        score_list.append(tempscore)
    return [feat_count,score_list]


roc_auc = roc_auc_score(y_test, y_predict)
cv_score = cross_val_score(reg_forest, X_train_res, y_train_res)
fpr, tpr, thresholds = roc_curve(y_test, predict(y_predict))
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


feat_stuff = feature_play(X_train_res, y_train_res, X_test, y_test, feature_num=18)

plt.figure(figsize=(15,10))
plt.plot(feat_stuff[0], feat_stuff[1], 'o-')
plt.ylabel('accuracy')
plt.xlabel('feat_count')
plt.show()
