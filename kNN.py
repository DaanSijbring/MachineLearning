#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 18:01:16 2018

@author: leo
"""

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

 #%%
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
 
# Cleaning the text sentences so that punctuation marks, stop words & digits are removed
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y

#%%
testData = ps.read_csv("drugsComTest_raw.tsv", sep="\t")
trainData = ps.read_csv('drugsComTrain_raw.tsv', sep="\t")
#%%
#prepare and vecorize trainData

train_clean_sentences = []
fp = trainData['review']
for line in fp:
    line = line.strip()
    cleaned = clean(line)
    cleaned = ' '.join(cleaned)
    train_clean_sentences.append(cleaned)
 
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_clean_sentences)
 
#%%
#get list of trainData ratings
ratings = []
for i in range(len(trainData['rating'])):
    ratings.append(trainData['rating'].iloc[i])
#%%
#prepare and vectorize testData
test_clean_sentence = []
test_sentences = testData['review']
for test in test_sentences:
    cleaned_test = clean(test)
    cleaned = ' '.join(cleaned_test)
    #cleaned = re.sub(r"\d+","",cleaned)
    test_clean_sentence.append(cleaned)

Test = vectorizer.transform(test_clean_sentence)

#%%
#actual test ratings
ratings_true = []
for i in range(len(testData['rating'])):
    ratings_true.append(testData['rating'].iloc[i])

#%%
# Clustering the trainData with KNN classifier
modelknn = KNeighborsClassifier(n_neighbors=1)
modelknn.fit(X,ratings)

#%%
#Use subset of destData (Kernel crashes with entire data)
Test2 = Test[:10000]

#%%
#predict ratings using kNN - Replace "Test2" with "Test" in order to use whole dataset. 
predicted_labels_knn = modelknn.predict(Test2)

#%%
#Calculate difference between predicted and actual ratings
diff = []
for i in range(0,len(predicted_labels_knn)):
    diff.append(ratings_true[i] - predicted_labels_knn[i])
    
#%%
#plot difference
plt.hist(diff)

#%%
# print % of correctly classified reviews
count = 0
for i in range(0,len(predicted_labels_knn)):
    if predicted_labels_knn[i] == ratings_true[i]:
        count += 1
print(count/len(predicted_labels_knn)*100)

#%%
# The next part was also included in the tutorial I used for this method, but I have not checked how accurate it is (it took too long)
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