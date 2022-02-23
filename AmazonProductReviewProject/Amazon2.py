## read data
import os

os.getcwd()
import pickle

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

# remove stopwords

import nltk
from nltk import word_tokenize, pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import gensim
from gensim.models import CoherenceModel

from gensim import corpora
from gensim import matutils, models
import scipy.sparse

from gensim.utils import simple_preprocess


def sent_to_words(texts):
    for text in texts:
        yield (gensim.utils.simple_preprocess(str(text), deacc=True))


df_reviews = df['reviews.text'].values.tolist()

df_words = list(sent_to_words(df_reviews))

import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')
import spacy

stop_words = stopwords.words('english') + list(string.punctuation)


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


df_nostop_words = remove_stopwords(df_words)


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
df_lemmatized = lemmatization(df_nostop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary
id2word = corpora.Dictionary(df_lemmatized)

# Create Corpus
corpus = [id2word.doc2bow(text) for text in df_lemmatized]

# Build LDA model

from gensim import matutils, models
import scipy.sparse

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=15,
                                            iterations=200,
                                            per_word_topics=True, alpha=0.1)
# print the 10 topics

import pprint

pprint.pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

from gensim.models import CoherenceModel

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=df_lemmatized, corpus=corpus, dictionary=id2word,
                                     coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()

import math

math.exp(coherence_lda)

##############
## Number of Topics (K)
# Dirichlet hyperparameter alpha: Document-Topic Density
# Dirichlet hyperparameter beta: Word-Topic Density

# supporting function

(0.51 - 0.47) / 0.47
