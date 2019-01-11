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
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support as score

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
#%%
#run algorithm for multiple n
p_values = []   #Precision
r_values = []   #Recall
f_values = []   #F-Score
n = 1000        #amount of included reviews
for i in range(1,3):
    modelknn = KNeighborsClassifier(n_neighbors=i)
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