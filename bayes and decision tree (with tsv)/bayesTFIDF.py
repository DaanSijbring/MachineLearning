from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

import time

from gensim.models import Word2Vec
from nltk.stem.porter import *

from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import csv

# Load stemmed review or stem data
def stem(data, fileName, stemmer):
	try:
		df = pd.read_csv(fileName, "r", error_bad_lines=False, delimiter="\t")
		print(df.columns)
		stemmed = df.stemmed
		revs = [None] * len(stemmed)
		for i in range(len(stemmed)):
			revs[i] = stemmed[i].split(" ")
		return revs
	except IOError:
		revs = [None] * len(data)
		for i in range(len(data)):
			revs[i] = re.sub(r'[^a-zA-Z ]+', '', data[i])
			revs[i] = revs[i].split(" ")
			revs[i] = [stemmer.stem(rev) for rev in revs[i]]
		import csv

		with open(fileName, 'wt') as out_file:
			tsv_writer = csv.writer(out_file, delimiter='\t')
			tsv_writer.writerow(['num', 'stemmed'])
			for i in range(len(revs)):
				tsv_writer.writerow([str(i),'"' + " ".join(revs[i]) + '"'])
		return revs

# Main body
df1 = pd.read_csv('stemmed_trainData.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
df2 = pd.read_csv('stemmed_testData.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
print(df1.columns)

rating1 = df1.rating
reviews1 = df1.review

rating2 = df2.rating
reviews2 = df2.review

stemmer = PorterStemmer()

timer = time.time()
# trainRevs = stem(reviews1, "train_revs.tsv", stemmer)
trainRevs = reviews1
print(time.time() - timer)

timer = time.time()
# testRevs = stem(reviews2, "test_revs.tsv", stemmer)
testRevs = reviews2
print(time.time() - timer)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(trainRevs)
Test = vectorizer.transform(testRevs)

# Fit the bayes model
mnb = MultinomialNB();
mnb.fit(X, rating1)

'''
# Predict the traning data again
sse = 0;
i = 0
for sentence in Test:
	result = mnb.predict(sentence)
	
	sse += (result[0] - rating2[i]) * (result[0] - rating2[i]) / len(testRevs)
	
	# Get 20 results
	if (i <= 20):
		print(result[0], rating2[i], (rating2[i] - result[0]))
		i += 1

print(sse)
'''

ratings_true = rating2

predicted_labels = []
for sentence in Test:
	predicted_labels.append(mnb.predict(sentence))

print(len(predicted_labels), len(ratings_true))

#%%
#Calculate difference between predicted and actual ratings
diff = [0]*len(predicted_labels)
for i in range(len(predicted_labels)):
    diff.append(ratings_true[i] - predicted_labels[i])
    
#%%
#precision and recall measures
precision, recall, fscore, support = score(ratings_true, predicted_labels) 
p, r, f, s = score(ratings_true, predicted_labels, average='weighted') 

#%%
#print precision, recall and f score
print("Rating\t Prec.\t Recall\t fscore\t Support")
for i in range(0,10):
    print(i+1,':',round(precision[i],3),round(recall[i],3), round(fscore[i],3), round(support[i],3))
print('Total:',round(p,3),round(r,3),round(f,3),len(predicted_labels))
