from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import pickle
from neurnet import *

def compute_mean(sentence):
    mean = np.zeros(150)
    for word in sentence:
        if word in model.wv.vocab:
            mean += model.wv[word]
    return(mean/len(sentence))

model = Word2Vec.load('word2vec.model')
df = pd.read_csv('drugsComTest_raw.tsv',delimiter="\t")
ratings  = df.rating.values
revs = pickle.load( open( "stems.pkl", "rb" ) )
means = [0] * len(revs)
for i in range(len(revs)):
    means[i] = compute_mean(revs[i])

nn = neuralnet(means,ratings,3)

for i in range(20):
    print(revs[i])
    print(f'estimated rating: {nn.y[i]}, actual rating: {ratings[i]}')

nn.performance()








        


