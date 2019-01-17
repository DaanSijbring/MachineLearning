# Importing libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as ps
import string
import re
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support as score
import pickle
from gensim.models import Word2Vec


#%%
testData = ps.read_pickle("testData.p")
trainData = ps.read_pickle('trainData.p')
#%%
testData['review'] = testData['review'].apply(' '.join)
trainData['review'] = trainData['review'].apply(' '.join)

#%% 
test_clean_sentences = testData['review']  
train_clean_sentences = trainData['review']

#%%
#Word2Vec
model = Word2Vec.load("word2vec_model_150.dms")
word2vecSize = 150


#%%
total = np.zeros((len(train_clean_sentences), word2vecSize))
i = 0
for sentence in train_clean_sentences:
	parsedWords = 0;
	for word in train_clean_sentences[i]:
		if (word in model.wv):
			total[i] += model.wv[word]
			parsedWords += 1

	if (parsedWords):
		total[i] /= parsedWords;

	i += 1
X=total
#%%

Test = np.zeros((len(test_clean_sentences), word2vecSize))
i=0
for sentence in test_clean_sentences:
    parsedWords = 0;
    for word in test_clean_sentences[i]:
        if (word in model.wv):
            Test[i] += model.wv[word]
            parsedWords += 1

    if (parsedWords):
        Test[i] /= parsedWords;
        
    i += 1
#%%
#get list of trainData ratings
ratings = []
for i in range(len(trainData['rating'])):
    ratings.append(trainData['rating'].iloc[i])

#%%
#actual test ratings
ratings_true = []
for i in range(len(testData['rating'])):
    ratings_true.append(testData['rating'].iloc[i])

#%%

#Once for the whole dataset
    
# Clustering the trainData with KNN classifier
modelknn = KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='ball_tree')
modelknn.fit(X,ratings)

#%%
predicted_labels_knn = modelknn.predict(Test)

#%%
#Calculate difference between predicted and actual ratings
diff = []
for i in range(0,len(predicted_labels_knn)):
    diff.append(ratings_true[i] - predicted_labels_knn[i])
    
#%%
#precision and recall measures
precision, recall, fscore, support = score(ratings_true, predicted_labels_knn) 
p, r, f, s = score(ratings_true, predicted_labels_knn, average='weighted') 

#%%
#print precision, recall and f score
print("Rating\t Prec.\t Recall\t fscore\t Support")
for i in range(0,10):
    print(i+1,':\t',round(precision[i],3),'\t',round(recall[i],3),'\t', round(fscore[i],3),'\t', round(support[i],3))
print('Total:\t',round(p,3),'\t',round(r,3),'\t',round(f,3),'\t',len(predicted_labels_knn))
"""

#%%
#Multiple times for range of n
p_values = []   #Precision
r_values = []   #Recall
f_values = []   #F-Score
n = 1000        #amount of included reviews
for i in range(1,3):
    modelknn = KNeighborsClassifier(n_neighbors=i, weights='distance')
    modelknn.fit(X,ratings)
    predicted_labels_knn = modelknn.predict(Test[:n])
    p, r, f, s = score(ratings_true[:n], predicted_labels_knn, average='weighted')
    p_values.append(p)
    r_values.append(r)
    f_values.append(f)
    print('k=',i,'\t p=',round(p,3),'\t r=',round(r,3),'\t f=', round(f,3))
#%%    
# The next part was also included in the tutorial I used for this method, but I have not checked how accurate it is (it took too long)
"""
"""

#------------------------OPTIONAL-----------------------------------
# Clustering with K-means technique (takes ages)
modelkmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, n_init=100)
modelkmeans.fit(X)    
#%%

predicted_labels_kmeans = modelkmeans.predict(Test)
"""
#%%
#---------------------------Source---------------------------------------
#https://appliedmachinelearning.blog/2018/01/18/conventional-approach-to-text-classification-clustering-using-k-nearest-neighbor-k-means-python-implementation/
