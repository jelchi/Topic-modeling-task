#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
np.random.seed(4)
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',50)
pd.set_option('display.width',5000)
pd.set_option('max_colwidth', 500)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)
from gensim import corpora, models
from matplotlib import pyplot as plt
from gensim.corpora.dictionary import Dictionary


# In[11]:


def predict_topic(lemmatized_text):
    dictionary = Dictionary.load('dictionary')
    tfidf = models.TfidfModel.load('tfidf')
    lda_model = gensim.models.LdaMulticore.load('lda.model')
    corpus = dictionary.doc2bow(lemmatized_text)
    corpus_tfidf = tfidf[corpus]
    topics = sorted(lda_model.get_document_topics(corpus_tfidf, minimum_probability= 0.2),key=lambda x:x[1],reverse=True)
    if topics:
        return topics[0][0]
    else:
        return -1


# In[12]:


def topic_name(topic_num):
    dictionary_topics = {4: 'Phone problems (mostly iPhone): battery, glitch, crash, freeze', 
                         5: 'Train problems (late etc.)',
                         3: 'Problems with account/password/logging', 
                         9: 'Problem with websites',
                         8: 'Problems with package delivery', 
                         11:'Bad customer service', 
                         7: 'Problems with taxi rides (Uber, Lyft)', 
                         6: 'Food category ',
                         10: 'Internet problems (outage etc.)', 
                         0: 'General questions, customer support',
                         2: 'Problems with apps (download, speed) and games',
                         1: 'Problems with travel (mostly airlines delays etc.)',
                         -1: 'Unknown'}
    return dictionary_topics.get(topic_num)


# In[ ]:




