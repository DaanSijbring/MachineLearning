import pandas as pd
import numpy as np
import pickle
from importlib import reload
from collections import Counter
#import nltk
#from nltk.corpus import stopwords

def bayes(prior,freqs,sentence):
    probs = prior
    for word in sentence:
        if word in freqs.keys():
            probs = probs*freqs[word]
    return(probs.tolist().index(max(probs))+1)


revs = pickle.load( open( "stemstrain.pkl", "rb" ) )
df = pd.read_csv('drugsComTrain_raw.tsv',delimiter="\t")
ratings  = df.rating.values
c = Counter(ratings); prior = np.zeros(10)
for i in range(10):
    prior[i] =  c[i+1]/len(ratings)

freq = {}; relfreq = {}
print('aids')
for i in range(len(revs)):
    for word in revs[i]:
        if word not in freq.keys():
            freq[word] = np.zeros(10)
        freq[word][int(ratings[i])-1] += 1 


for word in freq.keys():
    if sum(freq[word]) > 20:
        relfreq[word] = (freq[word] / prior) / sum((freq[word] / prior))

revs_test = pickle.load( open( "stems.pkl", "rb" ) )
df = pd.read_csv('drugsComTest_raw.tsv',delimiter="\t")
ratings_test  = df.rating.values

est_r = np.zeros(len(revs_test))
prior = np.zeros(10) + 0.1
for i,rev in enumerate(revs_test):
    est_r[i] = bayes(prior,relfreq,rev)

est_var = np.mean((est_r - ratings_test)**2)
wrongfreqs = Counter(np.abs(est_r - ratings_test))
for k in wrongfreqs.keys():
    wrongfreqs[k] = wrongfreqs[k] / len(revs_test)
R_score = 1 - est_var / np.var(ratings_test)



        

