# Importing libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pandas as ps
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from gensim.models import Word2Vec

#%%
# import prepared data
testData = ps.read_pickle("testData.p")
trainData = ps.read_pickle('trainData.p')
#%%
#retrieve reviews and ratings
test_clean_sentences = testData['review'].apply(' '.join)
train_clean_sentences = trainData['review'].apply(' '.join)

ratings = []
for i in range(len(trainData['rating'])):
    ratings.append(trainData['rating'].iloc[i])

ratings_true = []
for i in range(len(testData['rating'])):
    ratings_true.append(testData['rating'].iloc[i])
    
#%%
#NOTE: use either TF-IDF (ll. 29-34) or Word2Vec (ll. 37-69)
#-------------------------------TF-IDF---------------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_clean_sentences)
Test = vectorizer.transform(test_clean_sentences)

#-----------------------------END TF-IDF---------------------------------------

#%%
#-------------------------------Word2Vec---------------------------------------

model = Word2Vec.load("word2vec_model_50.dms")
word2vecSize = 50

X = np.zeros((len(train_clean_sentences), word2vecSize))
i = 0
for sentence in train_clean_sentences:
	parsedWords = 0;
	for word in train_clean_sentences[i]:
		if (word in model.wv):
			X[i] += model.wv[word]
			parsedWords += 1

	if (parsedWords):
		X[i] /= parsedWords;

	i += 1

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
#--------------------------------END Word2Vec----------------------------------

#%%
#NOTE: use either KNN for one K (ll. 73-103) or KNN for range of K (ll. 106-119)
#-------------------------------KNN for one K----------------------------------
    
# Clustering the trainData with KNN classifier
k = 65
modelknn = KNeighborsClassifier(n_neighbors=k, weights='distance')
modelknn.fit(X,ratings)

#%%
predicted_labels_knn = modelknn.predict(Test)

#%%
#precision and recall measures
precision, recall, fscore, support = score(ratings_true, predicted_labels_knn) 
p, r, f, s = score(ratings_true, predicted_labels_knn, average='weighted') 

#%%
#print precision, recall and f score
print("k= ",k)
print("Rating\t Prec.\t Recall\t fscore\t Support")
for i in range(0,10):
        print(i+1,':\t',round(precision[i],3),'\t',round(recall[i],3),'\t', round(fscore[i],3),'\t', round(support[i],3))
print('Total:\t',round(p,3),'\t',round(r,3),'\t',round(f,3),'\t',len(predicted_labels_knn))
#%%
#save results
with open("knn_oneVal.tsv", "w") as record_file:
    record_file.write("Category    precision    recall    f-score    Size\n")
    for i in range(0,10):
        record_file.write(str(i+1)+"     "+str(round(precision[i],5))+"      "+str(round(recall[i],5))+"      "+str(round(fscore[i],5))+"      "+str(support[i])+"\n")
    record_file.write("Total:     "+str(round(p,3))+"       "+str(round(r,3))+"       "+str(round(f,3))+"      "+str(len(predicted_labels_knn))+"\n")

#---------------------------END KNN for one K----------------------------------

#%%
#--------------------------KNN for range of K----------------------------------
# Computes KNN for K=5-205 in steps of 10
# Returns total precision, recall and F-score for each K
with open("knn_5-205.tsv", "w") as record_file:
    record_file.write("k    precision    recall    f-score\n")
    for i in range(5,215,10):
        modelknn = KNeighborsClassifier(n_neighbors=i, weights='distance')
        modelknn.fit(X,ratings)
        predicted_labels_knn = modelknn.predict(Test)
        p, r, f, s = score(ratings_true, predicted_labels_knn, average='weighted')
        print('k=',i,'\t p=',round(p,3),'\t r=',round(r,3),'\t f=', round(f,3))
        record_file.write(str(round(i,3))+"      "+str(round(p,3))+"       "+str(round(r,3))+"       "+str(round(f,3))+"\n")

#--------------------------END KNN for range of K------------------------------

#%%
#---------------------------Source---------------------------------------
#https://appliedmachinelearning.blog/2018/01/18/conventional-approach-to-text-classification-clustering-using-k-nearest-neighbor-k-means-python-implementation/
