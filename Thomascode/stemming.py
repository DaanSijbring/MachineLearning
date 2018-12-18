from nltk.stem.porter import *
import pickle
import re
import nltk
import pandas as pd
import negation
from nltk.stem.porter import *


df = pd.read_csv('drugsComTest_raw.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
#df = pd.read_csv('drugsComTrain_raw.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
print(df.columns)
reviews = df.review
stemmer = PorterStemmer()
revs = [None] * len(reviews)
print(len(revs))
for i in range(len(reviews)):
    revs[i] = re.sub(r'[^a-zA-Z ]+', '', reviews[i])
    revs[i] = revs[i].split(" ")
    revs[i] = re.sub('&#039;',"'" ,reviews[i])
    revs[i] = revs[i][1:-1].lower()
    revs[i] = re.sub(r'[^a-zA-Z \']+', '', revs[i])
    revs[i] = nltk.word_tokenize(revs[i])
    revs[i] = [stemmer.stem(rev) for rev in revs[i]]
    revs[i] = negation.negate_sequence(revs[i])

print(revs[1])

#with open('stems.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
with open('stemstrain.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(revs, f)





