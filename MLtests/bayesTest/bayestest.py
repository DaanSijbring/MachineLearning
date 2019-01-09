import NB.scikitBayes as bayes1
import NB.thomasBayes as bayes2
import numpy as np
import pandas as pd

import time

from gensim.models import Word2Vec
from nltk.stem.porter import *

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
df1 = pd.read_csv('drugsComTest_raw.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
df2 = pd.read_csv('drugsComTest_raw.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
print(df1.columns)

rating1 = df1.rating
reviews1 = df1.review

rating2 = df2.rating
reviews2 = df2.review

stemmer = PorterStemmer()

timer = time.time()
trainRevs = stem(reviews1, "train_revs.tsv", stemmer)
print(time.time() - timer)

timer = time.time()
testRevs = stem(reviews2, "test_revs.tsv", stemmer)
print(time.time() - timer)

word2vecSize = 150

model = None
try:
	model = Word2Vec.load("word2vec_model")
except IOError:
	model = Word2Vec(trainRevs,size=word2vecSize,min_count=5,workers=2)
	model.train(trainRevs, total_examples=len(reviews1),epochs = 1)
	model.save("word2vec_model")

print(model.wv.most_similar(positive='great',topn=6))
print

# Get average vectors for each sentence
total = np.zeros((len(trainRevs), word2vecSize))
i = 0
for sentence in trainRevs:
	parsedWords = 0;
	for word in trainRevs[i]:
		if (word in model.wv):
			total[i] += model.wv[word]
			parsedWords += 1

	if (parsedWords):
		total[i] /= parsedWords;

	i += 1

# Fit the bayes model
bbw = bayes2.BinnedBayesWrapper(2)
bbw.fit(total, rating1)

print

unique, counts = np.unique(rating1, return_counts=True)
priors = np.array(dict(zip(unique, counts)).items())[:,1]
print ("PRIORS", priors)
priors /= sum(priors)
print ("NORM PRIORS", priors)

# Predict the traning data again
sse = 0;
i = 0
for sentence in testRevs:
	testCase = np.zeros((1, word2vecSize))
	parsedWords = 0;
	for word in testRevs[i]:
		if (word in model.wv):
			testCase += model.wv[word]
			parsedWords += 1

	if (parsedWords):
		testCase /= parsedWords;
	
	result = bbw.predict(testCase, priors)
	
	sse += (result[0] - rating2[i]) * (result[0] - rating2[i]) / len(testRevs)
	
	# Get 20 results
	if (i <= 20):
		print(result[0], rating2[i], (rating2[i] - result[0]))
		i += 1

print(sse)
