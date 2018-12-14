#%%
import pandas as ps
import re
from collections import Counter 
from nltk.corpus import stopwords
from operator import itemgetter
import matplotlib.pyplot as plt
#%%
testData = ps.read_csv('C:/Users/daan_/OneDrive/Documents/HMC/Machine Learning/drugsCom_raw/drugsComTest_raw.tsv', sep="\t")
trainData = ps.read_csv('C:/Users/daan_/OneDrive/Documents/HMC/Machine Learning/drugsCom_raw/drugsComTrain_raw.tsv', sep="\t")
#%%
#transform text to lower case + remove special characters
def tokenize(text):
    text = text.lower()
    tokens = re.sub('[^a-z]', ' ', text).split()
    return tokens
#%%    

def tokenizeDataSet(data):
    tokenized = []
    for i in range(len(data['review'])):
        tokenized.append(tokenize(data['review'][i]))
    
    return tokenized
#%%
tokTrain = tokenizeDataSet(trainData)
conList = [item for sublist in tokTrain for item in sublist]
counts = Counter(conList)
#%%
stopWords = set(stopwords.words('english'))
len(stopWords)

wordsFiltered = []
for w in conList:
    if w not in stopWords:
        wordsFiltered.append(w)

res = len(wordsFiltered)/len(conList)
#%%
counts = Counter(wordsFiltered)
counts = list(counts.items())
counts.sort(key=itemgetter(1),reverse=True)
freq = [x[1] for x in counts]
plt.plot(freq[1:10])