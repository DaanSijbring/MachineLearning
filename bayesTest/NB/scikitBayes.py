from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

import numpy as np

import math

# GNB should be multiclass, it is being used now
# MNB should be good for text classification
gnb = GaussianNB();
mnb = MultinomialNB();

class BinnedBayesWrapper:
	def __init__(self, bins):
		self.bins = bins
		self.model = None
	
	def fit(self, trainData, labelData):
		binnedData = self.bin(self.normalize(trainData))
		self.model = gnb.fit(binnedData, labelData)
		return self
	
	def predict(self, testData):
		binnedData = self.bin(self.normalize(testData))
		return self.model.predict(binnedData)
	
	def normalize(self, dataVector):
		minValue = np.amin(dataVector)
		maxValue = np.amax(dataVector)
		return (dataVector + minValue) / (maxValue - minValue)
	
	# Split dataVector into n bins per variable
	def bin(self, dataVector):
		newVector = np.zeros(dataVector.shape * np.array([1, self.bins])).astype(int);
		
		for i in range(dataVector.shape[0]):
			for j in range(dataVector.shape[1]):
				binId = j * self.bins + int(math.floor(dataVector[i][j] * self.bins))
				if (dataVector[i][j] < 1.0):
					newVector[i][binId] = 1
				else:
					print("we got a problem, " + str(dataVector[i][j]) + ", " + str(binId) + ", " + str(newVector.shape[1]))
		
		return newVector

def frequencyBayes(trainData, labelData):
	return mnb.fit(trainData, labelData);

def binnedBayes(trainData, labelData, bins):
	bbw = BinnedBayesWrapper(bins);

# Another quick test, ignore
def main():
	bbw = BinnedBayesWrapper(4);
	trainData = np.array([[0.32], [0.2], [0.54], [0.9]])
	labels = np.array([0, 0, 1, 1])
	
	bbw.fit(trainData, labels)
	result = bbw.predict(trainData)
	print(result)
	
	result2 = gnb.fit(trainData, labels).predict(trainData)
	print(result2)

if __name__ == "__main__":
    main()
