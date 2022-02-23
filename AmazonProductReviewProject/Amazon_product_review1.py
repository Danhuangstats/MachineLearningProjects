### collect data
### scrape data from website
### now I obtain the data from Kaggle, so this step can be skipped

## Input: question, how can we review the products in score
## Data gathering: scope the project and scrape the website
## Data Cleaning: put data into standard format for future analysis
## Output: a corpus and a document-term matrix


## read data
import os
import pickle

os.getcwd()

import pandas as pd
import pickle
import nltk

productReview = pd.read_csv('AmazonProductReviewProject/7817_1.csv')

productReview.head()
productReview.shape

productReview.columns

productReview.describe()
productReview.info()

productReview['asins'].unique()  ### asins are the products
productReview['reviews.text']
productReview['reviews.title']
productReview['reviews.rating'].isna().count()

df = pd.DataFrame(productReview[['asins', 'reviews.text']])

# apply the first around of text cleaning techniques

import re
import string


def cleanText(text):
    '''make text lowercase, remove punctuations, remove stop words  '''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


r1 = lambda x: cleanText(x)

df['reviews.text'] = pd.DataFrame(df['reviews.text'].apply(r1))


# apply a second round of cleaning

def cleanText2(text):
    ''' get rid of some additional punctuation and non-sensical
    text that was missed
    '''
    text = re.sub('[''""...]', '', text)
    text = re.sub('\n', '', text)
    return text


r2 = lambda x: cleanText2(x)

df['reviews.text'] = pd.DataFrame(df['reviews.text'].apply(r2))

# let's pickle for later use

df.to_pickle('corpus.pkl')

# Now do document-term matrix using CountVectorize
# and exclude common English terms

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')

dfCV = cv.fit_transform(df['reviews.text'])
dfDTM = pd.DataFrame(dfCV.toarray(), columns=cv.get_feature_names_out())

dfDTM.index = df.index

# create a pickle for later use

dfDTM.to_pickle('dtm.pkl')

pickle.dump(cv, open('cv.pkl', 'wb'))

# Exploratory data analysis: EDA steps: Data, Aggregate, Visualize, Insights
# what are some ways you can think of to explore our data set?
# Top words, vocabulary, amount of profanity, positive word, negative word

# Most common words


data = pd.read_pickle('dtm.pkl')
data['asins'] = df['asins']

data = data.groupby('asins').sum()

data = data.transpose()
data.head()

# find top 50 words for reviews of products

topDict = {}
for i in data.columns:
    top = data[i].sort_values(ascending=False).head(500)
    topDict[i] = list(zip(top.index, top.values))

# print the top 100 words said by each product

for i, j in topDict.items():
    print(i)
    print(', '.join([word for word, count in j[0:99]]))
    print('---')

from collections import Counter

words = []
for products in data.columns:
    top = [word for (word, count) in topDict[products]]
    for t in top:
        words.append(t)

Counter(words).most_common()

addStopwords = [word for word, count in Counter(words).most_common() if count
                > 26]

# create a new document-term matrix using a new list of stop words

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

dataClean = pd.read_pickle('corpus.pkl')

stopWords = text.ENGLISH_STOP_WORDS.union(addStopwords)

cv = CountVectorizer(stop_words=stopWords)
stopCV = cv.fit_transform(dataClean['reviews.text'])
dfStop = pd.DataFrame(stopCV.toarray(), columns=cv.get_feature_names_out())

dfStop.index = dataClean.index

pickle.dump(cv, open('cvStop.pkl', 'wb'))
dfStop.to_pickle('dtmStop.pkl')

from wordcloud import WordCloud

wc = WordCloud(stopwords=stopWords, background_color='white',
               colormap='Dark2', max_font_size=150, random_state=42)

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 6]

dataClean = pd.DataFrame(data=dataClean)
dataClean.columns = ['asins', 'Reviews']

fullProducts = data.columns

for index, products in enumerate(data.columns):
    wc.generate(dataClean.Reviews[index])
    plt.subplot(6, 9, index + 1)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(fullProducts[index])

plt.show()

### sentiment analysis
# input: a corpus. great=positive or not great=negative;
# TextBlob
# Output

from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

dataClean['polarity'] = dataClean.Reviews.apply(pol)
dataClean['subjectivity'] = dataClean.Reviews.apply(sub)

dataClean.groupby('asins ').agg({
    'polarity': 'mean', 'subjectivity': 'mean'})

for index, products in enumerate(dataClean.index):
    x = dataClean.polarity.loc[products]
    y = dataClean.subjectivity.loc[products]
    plt.scatter(x, y, color='blue')

plt.title('Sentiment Analysis')
plt.xlabel('Negative-Positive')
plt.ylabel('Subjectivity')

plt.show()

# Topic modeling:
# Input: a document-term matrix. Each topic will consist of words
# gensim, latent dirichlet allocation is a statistical model

## every document consists of a mix of topics
## every topic consists of a mix of words


dataDTM = pd.read_pickle('dtmStop.pkl')

tdm = dataDTM
tdm['asins'] = dataClean['asins']
tdm = tdm.groupby('asins').sum()

tdm = tdm.transpose()

sparseCounts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparseCounts)

cv = pickle.load(open('cvStop.pkl', 'rb'))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())

# consider all text
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=10)
lda.print_topics()

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=3, passes=10)
lda.print_topics()

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, passes=10)
lda.print_topics()

# consider nouns only, or only adjective
import nltk
from nltk import word_tokenize, pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def adj(text):
    isAdj = lambda pos: pos[:2] == 'JJ'
    Token = word_tokenize(text)
    allAdjs = [word for (word, pos) in pos_tag(Token) if isAdj(pos)]
    return ' '.join(allAdjs)


datacLean = pd.read_pickle('corpus.pkl')

dataAdjs = pd.DataFrame(datacLean['reviews.text'].apply(adj))

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

topDict = {}
for i in tdm.columns:
    top = data[i].sort_values(ascending=False).head(200)
    topDict[i] = list(zip(top.index, top.values))

# print the top 100 words said by each product

for i, j in topDict.items():
    print(i)
    print(', '.join([word for word, count in j[0:49]]))
    print('---')

words = []
for products in tdm.columns:
    top = [word for (word, count) in topDict[products]]
    for t in top:
        words.append(t)

Counter(words).most_common()

addstopwords = [word for word, count in Counter(words).most_common() if count
                > 24]

stopwords = text.ENGLISH_STOP_WORDS.union(addstopwords)

cvadj = CountVectorizer(stop_words=stopwords)

dataCvadj = cvadj.fit_transform(dataAdjs['reviews.text'])
dataDTMadj = pd.DataFrame(dataCvadj.toarray(), columns=cvadj.get_feature_names_out())

corpus = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(dataDTMadj.transpose()))

id2word = dict((v, k) for k, v in cvadj.vocabulary_.items())

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=10)
lda.print_topics()

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=3, passes=10)
lda.print_topics()

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, passes=10)
lda.print_topics()


### consider nouns and adjectives


def na(text):
    isna = lambda pos: pos[:2] == 'JJ' or pos[:2] == 'NN' or pos[:2] == 'BB' or pos[:2] == 'VV'
    Token = word_tokenize(text)
    allna = [word for (word, pos) in pos_tag(Token) if isna(pos)]
    return ' '.join(allna)


datacLean = pd.read_pickle('corpus.pkl')

datana = pd.DataFrame(datacLean['reviews.text'].apply(na))

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

cvna = CountVectorizer(stop_words=stopWords)

datana = cvna.fit_transform(datana['reviews.text'])
datana = pd.DataFrame(datana.toarray(), columns=cvna.get_feature_names_out())

corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(datana.transpose()))

id2word = dict((v, k) for k, v in cvna.vocabulary_.items())

#
# lda = models.LdaModel(corpus=corpusna, id2word=id2word, num_topics=2, passes=10)
# lda.print_topics()
#
# lda = models.LdaModel(corpus=corpusna, id2word=id2word, num_topics=3, passes=10)
# lda.print_topics()
#
# lda = models.LdaModel(corpus=corpusna, id2word=id2word, num_topics=4, passes=10)
# lda.print_topics()

### identify topics in each document

lda = models.LdaModel(corpus=corpusna, id2word=id2word, num_topics=10, passes=10,
                      random_state=100, update_every=1, chunksize=100,
                      alpha='auto', per_word_topics=True)

print(lda.print_topics())
doc_lda = lda[corpusna]

## which topics each review contains

# corpusTransformed = lda[corpusna]
#
# list(zip([a for [(a, b)] in corpusTransformed], datana.index))

# evaluate model using coherence score
import gensim
from gensim.models import CoherenceModel

from gensim import corpora

word2id = dict((v, k) for k, v in cvna.vocabulary_.items())
d = corpora.Dictionary()
d.id2token = id2word
d.token2id = word2id

corpora.Dictionary(datacLean['reviews.text'])

coherence_model_lda = CoherenceModel(model=lda, texts=datacLean['reviews.text'], dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

### visualize the topics

import pyLDAvis

vis = pyLDAvis.gensim.prepare(lda, corpusna, id2word)
