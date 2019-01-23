import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

df1 = pd.read_csv('treeFullTFIDF.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
df2 = pd.read_csv('treeFullvec.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
df3 = pd.read_csv('bayesFullTFIDF.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
df4 = pd.read_csv('bayesFullvec.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
df5 = pd.read_csv('knn65.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
df6 = pd.read_csv('knn_w2v50_k65.tsv',error_bad_lines=False,warn_bad_lines=True,delimiter="\t")
print(df1.columns)

fig = plt.figure()
ax = fig.add_subplot(111);
ax.bar(1, df1.f[10]);
ax.bar(2, df2.f[10]);
ax.bar(3, df3.f[10]);
ax.bar(4, df4.f[10]);
ax.bar(5, df5.f[10]);
ax.bar(6, df6.f[10]);
ax.plot([2.5, 2.5], [0, 2], 'b--', label='_nolegend_');
ax.plot([4.5, 4.5], [0, 2], 'b--', label='_nolegend_');
plt.axis([0, 7, 0, 1.4]);
plt.ylabel("f1-score");
# plt.legend(["Precision", "Recall"], loc=1);
plt.legend(["DT TFIDF", "DT W2V", "NB TFIDF", "NB W2V", "KNN TFIDF", "KNN W2V"], loc=1);
plt.title("F1-score of different method combinations");
ax.set_xticks([1.5, 3.5, 5.5]);
ax.set_xticklabels(["DT", "NB", "KNN"]);
plt.show();
