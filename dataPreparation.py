#%%
import pandas as ps
import re
from collections import Counter 
from nltk.corpus import stopwords
from nltk.stem.porter import *
import nltk
from operator import itemgetter
import matplotlib.pyplot as plt
import os
import string
import time 
from tqdm import tqdm
#%%
os.chdir('C:\\Users\\daan_\\GitHub\\MachineLearning\\')
#%%
#%%
testData = ps.read_csv('drugsCom_raw\\drugsComTest_raw.tsv', sep="\t")
trainData = ps.read_csv('drugsCom_raw\\drugsComTrain_raw.tsv', sep="\t")
#%%
# Function 'negate' copied from Thomascode/negation.py
def negate(words):
    prev = []
    result = []
    neg = ["not", "n't", "no"]
    for word in words:
        if word not in neg:
            if prev in neg:
                result.append('not_' + word)
            else:
                result.append(word)
        prev = word
    return(result)
#%%
def processData(data):
    import tqdm
    filteredData = data
    stemmer = PorterStemmer()
    stopWords = set(stopwords.words('english'))
    #process each review individually
    for i in tqdm.trange(len(data)):
        #Replace ' hex code with \'
        rev = re.sub('&#039;',"'" ,filteredData.review[i])
        #Remove tokens not needed
        rev = re.sub(r'[^a-zA-Z \']+', '', rev)
        #Translate to lower case
        rev = rev.lower()
        #Tokenize using NLTK's tokenizer
        rev = nltk.word_tokenize(rev)
        #Stem the words
        rev = [stemmer.stem(word) for word in rev]
        #negate
        rev = negate(rev)
        #Remove stopwords
        rev = [word for word in rev if word not in stopWords]
        filteredData.review[i] = rev
    return filteredData
#%%
    
a = processData(testData)