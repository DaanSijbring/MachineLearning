from gensim.models import Word2Vec
import pickle


def genmodel(sentences):
    model = Word2Vec(sentences,size=150,min_count=5,workers=2)

def compute_mean(sentence):
    for word in sentence:
        if word in model.wv.vocab:
            means[i] += model.wv[word]

#model.train(revs, total_examples=len(revs),epochs = 1)
#print(model.wv.most_similar(positive='great',topn=6))
#model.save("word2vec.model")