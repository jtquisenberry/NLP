# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/

import numpy as np
from tfidf_embedding_vecorizer import TfidfEmbeddingVectorizer

with open(r"E:\Corpora\glove\glove.6B.50d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}


# import gensim
# # let X be a list of tokenized texts (i.e. list of lists of tokens)
# model = gensim.models.Word2Vec(X, size=100)
# w2v = dict(zip(model.wv.index2word, model.wv.syn0))






