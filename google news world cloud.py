# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:46:59 2020

@author: Ashvin
"""
from GoogleNews import GoogleNews
from newspaper import Article
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
#nltk.download('punkt')

googlenews=GoogleNews(start='05/01/2020',end='05/31/2020')
googlenews.search('Narendra Modi')
result=googlenews.result()
df=pd.DataFrame(result)
print(df.head())

for i in range(6):
    googlenews.getpage(i)
    result=googlenews.result()
    df=pd.DataFrame(result)
list=[]
for ind in df.index:
    dict={}
    article = Article(df['link'][ind])
    article.download()
    article.parse()
    article.nlp()
    dict['Date']=df['date'][ind]
    dict['Media']=df['media'][ind]
    dict['Title']=article.title
    dict['Article']=article.text
    dict['Summary']=article.summary
    list.append(dict)
news_df=pd.DataFrame(list)
news_text = news_df['Summary']
#print(news_text)

#news_text.append(news_text)

# Appending all rows data into single row
news_text = news_text.str.cat(sep=',', na_rep=None, join='left')

# Removing unecessary charecters
news_text.rstrip('\n , .')

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()
results = []

for line in news_text:
    pol_score = sia.polarity_scores(line)
    pol_score['Summary'] = line
    results.append(pol_score)

import pprint
pprint.pprint(results[:3], width=100)
print(results)
df = pd.DataFrame.from_records(results)
df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
df.head()

print(df.label.value_counts())

print(df.label.value_counts(normalize=True) * 100)


sns.set(style='darkgrid', context='talk', palette='Dark2')

fig, ax = plt.subplots(figsize=(8, 8))

counts = df.label.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

plt.show()

