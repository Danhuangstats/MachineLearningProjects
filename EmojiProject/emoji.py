## load data
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing

emoji_data = pd.read_csv('./EmojiProject/emoji_df.csv')
emoji_data.head(5)
emoji_data.info
emoji_data.describe()
group = emoji_data['group'].unique()

group = [str.lower(w) for w in group]

## word cloud

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(('&'))


def remove_stopwords(data):
    output_array = []
    for sentence in data:
        temp_list = []
        for word in word_tokenize(sentence):
            if word not in stop_words:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array


group_clean = remove_stopwords(group)

from wordcloud import WordCloud

wc = WordCloud(stopwords=stop_words, background_color='white',
               colormap='Dark2', max_font_size=150, random_state=42)

wc.generate(str(group_clean))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')

####
# import re
# from collections import namedtuple, Counter
#
# with open('./EmojiProject/emoji-test.txt', 'rt') as file:
#     emoji_text = file.read()
#
# emoji_text[:2800]
import demoji
import numpy as np
from sklearn.preprocessing import LabelEncoder

X = emoji_data['emoji']

X_text = []

for i in range(len(X)):
    X_text.append(demoji.findall(X[i]))

Y = emoji_data['group'].astype('category').cat.codes

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

X_train_scaled = sklearn.preprocessing.StandardScaler().fit(X_train).transform(X_train)

import re

re.compile(r' ?E\d+\.\d+').split(str(X))
