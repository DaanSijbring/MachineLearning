from sklearn.tree import DecisionTreeClassifier
import numpy as np

from gensim.models import Word2Vec
import re
import pandas as pd
from nltk.stem.porter import *

df = pd.read_csv('../drugsCom_raw/drugsComTest_raw.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
print(df.columns)

rating = df.rating
reviews = df.review
stemmer = PorterStemmer()
revs = [None] * len(reviews)

for i in range(len(reviews)):
    revs[i] = re.sub(r'[^a-zA-Z ]+', '', reviews[i])
    revs[i] = revs[i].split(" ")
    revs[i] = [stemmer.stem(rev) for rev in revs[i]]
    
    

print(revs[1])

model = Word2Vec(revs,size=150,min_count=5,workers=2)
model.train(revs, total_examples=len(reviews),epochs = 1)
print(model.wv.most_similar(positive='great',topn=6))
print

# Get average vectors for each sentence
total = np.zeros((len(revs), 150))
i = 0
for sentence in revs:
	parsedWords = 0;
	for word in revs[i]:
		if (word in model.wv):
			total[i] += model.wv[word]
			parsedWords += 1

	if (parsedWords):
		total[i] /= parsedWords;

	i += 1

# Fit the decision tree model
dtc = DecisionTreeClassifier("gini", "best", 40)
tree = dtc.fit(total, rating);

print

# Predict the training data again
i = 0
sse = 0
results = np.zeros((len(revs), 3))
for sentence in revs:
	testCase = np.zeros((1, 150))
	parsedWords = 0;
	for word in revs[i]:
		if (word in model.wv):
			testCase += model.wv[word]
			parsedWords += 1

	if (parsedWords):
		testCase /= parsedWords;
	
	result = tree.predict(testCase)
	results[i][0] = rating[i]
	results[i][1] = result[0]
	results[i][2] = rating[i] - result[0]
	sse += ((rating[i] - result[0]) * (rating[i] - result[0])) / len(revs)
	# print(result[0], rating[i], (rating[i] - result[0]))
	
	# Get 20 results
	# if (i > 20):
	# 	break;
	
	i += 1

print("Tree Result\tOG Result\tDifference")
print(results)
print
print("SSE: " + str(sse))