#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import numpy as np
import gensim
import spacy
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
np.random.seed(4)
import string
#from spellchecker import SpellChecker
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',50)
pd.set_option('display.width',5000)
pd.set_option('max_colwidth', 500)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)
from gensim import corpora, models
from matplotlib import pyplot as plt
import spacy


# In[ ]:


nlp = spacy.load("en_core_web_sm")


# In[ ]:


def preprocess(text):
    if type(text) is not str:
        return
    result = []
    text = text.lower() #lowercase
    text = re.sub('http://\S+|https://\S+', '', text)  # remove urls
    text = re.sub('@\S+', '', text)  #remove mentions
    text = re.sub('_{2}\w+_{2}', '', text) #remove masked data
    text = re.sub('\d', '', text) #remove digits
    text = re.sub('’|‘', '', text) #remove additional punctuation
    
    #remove emojis
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    text = regrex_pattern.sub(r'',text)
    
    text = text.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
            
    return " ".join(text.split())


# In[ ]:


def lemmatize(text):
    lemmatized_text = []
    text_nlp = nlp(text)
    lemmatized_text = [str(token.lemma_) for token in text_nlp # only nouns, verbs, adjectives and adverbs 
                  if (token.pos_ in ['NOUN', 'ADV', 'VERB', 'ADJ']) and token.text not in nlp.Defaults.stop_words] 
   
    return lemmatized_text


# In[ ]:


def lemma(text):
    words = []
    text = ' '.join(text)
    text = re.sub('\]|\[|\,|\'', '', text)
    text = re.sub('\\\\ud200', '', text)
    text = re.sub('。', '', text)
    for i in text.split():
        if i:
            words.append(i)
    return words


# In[1]:


def remove_more_stop(text):
    my_stop_words = STOPWORDS.union(set(['do', 'so', 'be', 'go', 'i', 'm', 're', 'have','why', 'when', 'how', 're', 'then', 's', 'get', 'try',
                                    'e', 'aa', 'fuck', 'r', 'fucking', ]))
    new_text = []
    for word in text:
        if (word not in my_stop_words):
            new_text.append(word)
    return new_text


# In[ ]:




