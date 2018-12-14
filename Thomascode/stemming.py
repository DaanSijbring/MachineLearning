
import pickle
import re
import pandas as pd
from nltk.stem.porter import *

df = pd.read_csv('drugsComTest_raw.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
print(df.columns)
reviews = df.review
stemmer = PorterStemmer()
revs = [None] * len(reviews)
print(len(revs))
for i in range(len(reviews)):
    revs[i] = revs[i].sub(r'[^a-zA-Z ]+', '', reviews[i])
    revs[i] = revs[i].split(" ")
    revs[i] = [stemmer.stem(rev) for rev in revs[i]]
    
print(revs[1])

with open('stems.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(revs, f)






