import numpy as np

import math

def bayes(priors, freqs, keys, sentence):
    probs = np.zeros(10) + np.log(priors)
    for j in range(sentence.shape[0]):
		word = str(j) + "," + str(sentence[j])
		if word in keys:
			probList = np.array(freqs[word])
			for i in range(len(probList)):
				probs[i] += math.log(probList[i])
    return probs.tolist().index(max(probs)) + 1

def getFrequencies(data, labelData):
	keys = set();
	freq = {}
	relfreq = {}
	for i in range(len(data)):
		if i % (len(data)/10) == 0:
			print(i, len(data))
		sentence = data[i]
		for j in range(sentence.shape[0]):
			word = str(j) + "," + str(sentence[j])
			if word not in keys:
				keys.add(word)
				freq[word] = np.zeros(10)
			freq[word][int(labelData[i])-1] += 1
	
	for word in keys:
		for i in range(10):
			if freq[word][i] == 0:
				freq[word][i] = 1
		if sum(freq[word]) > 20:
			relfreq[word] = freq[word] / sum(freq[word])
	print(len(keys))
	return relfreq

class BinnedBayesWrapper:
	def __init__(self, bins):
		self.bins = bins
		self.model = None
		self.keys = None
	
	def fit(self, trainData, labelData):
		binnedData = self.bin(self.normalize(trainData))
		# self.model = gnb.fit(binnedData, labelData)
		self.model = getFrequencies(binnedData, labelData)
		self.keys = set(self.model.keys())
		return self
	
	def predict(self, testData, priors):
		binnedData = self.bin(self.normalize(testData))
		# return self.model.predict(binnedData)
		results = np.zeros((testData.shape[0]))
		for i in range(testData.shape[0]):
			sentence = binnedData[i]
			probs = bayes(priors, self.model, self.keys, sentence)
			results[i] = probs
		return results
	
	def normalize(self, dataVector):
		minValue = np.amin(dataVector)
		maxValue = np.amax(dataVector)
		return (dataVector - minValue) / (maxValue - minValue)
	
	# Split dataVector into n bins per variable
	def bin(self, dataVector):
		newVector = np.zeros(dataVector.shape).astype(int);
		
		for i in range(dataVector.shape[0]):
			for j in range(dataVector.shape[1]):
				newVector[i][j] = int(math.floor(dataVector[i][j] * self.bins)) % self.bins
		
		return newVector

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
