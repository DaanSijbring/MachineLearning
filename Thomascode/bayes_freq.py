import numpy as np
import pandas as pd
import pickle
from importlib import reload
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support as score

#returns most likely rating basedon the naive bayes method
def bayes(prior,freqs,sentence):
    probs = prior
    for word in sentence:
        if word in freqs.keys():
            probs = probs*freqs[word]
            probs = probs / sum(probs)
    return(probs.tolist().index(max(probs))+1)

#read processed data
df = pickle.load( open( "trainData.p", "rb" ) )
ratings  = df.rating.values
revs = df.review
c = Counter(ratings); prior = np.zeros(10)
for i in range(10):
    prior[i] =  c[i+1]/len(ratings)

#construct frequency table
freq = {}; relfreq = {}
for i in range(len(revs)):
    for word in revs[i]:
        if word not in freq.keys():
            freq[word] = np.zeros(10)
        freq[word][int(ratings[i])-1] += 1 
#filter out low frequency words 
for word in freq.keys():
    if sum(freq[word]) > 20:
        relfreq[word] = (freq[word] / (prior*len(ratings))) 

df_test = pickle.load( open( "testData.p", "rb" ) )
ratings_test  = df_test.rating.values
revs_test = df_test.review

est_r = np.zeros(len(revs_test))
for i,rev in enumerate(revs_test):
    est_r[i] = bayes(prior,relfreq,rev)



wrongfreqs = Counter(abs(est_r-ratings_test))
precision, recall, fscore, support = score(ratings_test, est_r) 
for k in wrongfreqs.keys():
    wrongfreqs[k] = wrongfreqs[k] / len(revs_test)




        

