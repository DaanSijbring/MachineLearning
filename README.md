# MachineLearning

## dataPreparation.py
It is currently set to convert the train set to a pickle file.

## Methods
### KNN_final.py
1. use either "TF-IDF" (ll. 29-34) or "Word2Vec" (ll. 37-69) for vectorization
2. use either "KNN for one K" (ll. 73-103) or "KNN for range of K" (ll. 106-119)
	* for "KNN for one K", hyperparameter k can be changed in line 76 (k = 65)
3. precision, recall and f-score are printed and stored in .tsv file

### freq_bayes.py
Does not automatically save or print the results

### The other method files
You can just run them.
If the file runs word2vec, you can edit the vector size constant in the file.
For the batch scripts you can instead use a range of vector sizes (again in the file).

The decision tree parameters are set when the sklearn instance is created (right before training).
For the batch scripts, it will use a range of tree depths.

## Plots
They will work on the supplied results.
