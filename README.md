# How-to-Spot-a-Bot

<br>
Early in 2018 Facebook conceded to investors that somewhere between 60 and 80 million fake profiles currently exist of the social network’s 2 billion users. This is a significant increase from 2016, when it last reported a mere 18 million counterfeit profiles.<br>
In July, Twitter announced that it would be suspending 70 million accounts, an aggressive action prompted by continuous allegations that the social media platform exercises undue influence on news outlets, such as the U.S.’s most recent presidential elections or the caravan migration of workers from South America.
With populations becoming increasingly dependent on social media platforms as a news source, and with a growing number of false endorsements (in the form os retweets, facebook likes, etc), it is critical that we get familiar with what spambots are made of and auto-clean this mess!<br>

## Project Goals
<br>
In this study I seek to build a model which will take a user’s information, such as their name, gender, age and sample tweets, and identify whether or not a profile is genuine or produced via an automated account creator.

## The Data
<br>
This data was compiled by Crowdflower and is comprised of genuine, traditional, and social spambot Twitter accounts. The initial gathering of data was done on behalf of Twitter Italia and their pursuit of research in the same field. There are a total of 4,449,396 rows and 42 columns (of text_level data) and 4465 rows and 42 columns (of metadata) that include unique id numbers, screen_name, count of friends, count of favorites, personal urls, language, origin location, entire tweets, as well as image descriptions.

With such a massive data set, I wondered if the undertaking best be approached using high-level account information (the metadata of a user) and comparing it to text analysis of actual tweets.

## Methods
<br>
I implemented Random Forest and Logistic models to become more familiar with my data on the account level. As someone who doesn't actively engage with social media platforms day to day, I first needed to understand some of me features better.
<br>

<img src='https://github.com/HM618/Capstone-2/blob/master/Screen%20Shot%202018-11-15%20at%2011.33.10%20PM.png'>
<br>

Statuses ~ the number of tweets, or status updates, made by an account*
*each account is unique and considered as an independent entity*

Followers ~ unique accounts who receive status updates from a particular profile

Friends ~ unique accounts that profiles follow

Favorites ~ number of tweets a profile ‘likes’ or ‘saves’

Listed ~ a curated group of twitter followers

RT ~ retweets. A declarative acronym stating that you are quoting someone using their tweet

My initial Random Forests models were a bit wonkey, and I discovered my metadata was terribly imbalanced. Using SMOTE (<a href='https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html'>*Synthetic Minority Over-sampling Technique*</a>) I was able to balance my data 50/50, and proceed with a train/test split more representative.
<img src='https://github.com/HM618/Capstone-2/blob/master/capstone_2/Images/Random%20Forest%20ROC.png'>
<br>
<small> With Imbalanced Data</small>
<img src='https://github.com/HM618/Capstone-2/blob/master/capstone_2/Images/ROC%20better.png'>
<br>
<small> With Balanced Data</small>

After using Random Forest to highlight where features were predictive or not, I was able to widdle my features down to 'statuses_count', 'geo_timezone' 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'lang', and 'profile_use_background_image'

<img src='https://github.com/HM618/Capstone-2/blob/master/capstone_2/Images/Figure_1.png'>
<small> Confusion Matrix With Specified Features</small>

<img src='https://github.com/HM618/Capstone-2/blob/master/capstone_2/Images/Confusion%20Matrix%20and%20Classification%20Report%2C%20nonshaped.png'>
<br> While the scores look good, I will revisit the notion of data leakage. Even though these features seem unrelated, further investigation into their meaning and how they are quantified may offer valuable insight.
<br>
<br>
As to the text level analysis, I found a few things interesting. While there was a slight imbalance of bots to nots, rectifying these numbers did not change some of the information I gathered. While genuine Tweeters use an average of 1.5 hashtags a tweet, twitterbots use even less. However, Twitterbots know how to get people enrolled by giving an average of 3 shoutouts in the form of an '@' symbol per tweet, while real people only occasionally give one.
<br>
<br>
One last thing that should be mentioned is the noticeably different language used between Bots and NotBots...
<img src='https://github.com/HM618/Capstone-2/blob/master/capstone_2/Images/Bot_words_with_Stop_Words.png'>
<small> This is a sampling of the most frequently used words by Italian twitterbots<br><b>che</b> *what?* <b>grandi dubbi</b> *big doubts* <b>amo</b> *love*
<b>non</b> *no* <b>Dubbi Ma</b> *No Doubt* <b>gli uomini</b> *the men*</small>
<img src='https://github.com/HM618/Capstone-2/blob/master/capstone_2/Images/Genuine_Words.png'>


## Further Work

While I certainly became familiar with different aspects that may define a bot, I'd like to continue my efforts in a more pronounced direction with success in MVP's:
 - Determine common words or phrases used in bot language and that of human vernacular for comparison
 - Discover trends in topics, products or markets where bot language is most prevalent
 - Provide interpretable visuals that accurately capture these relationships 
