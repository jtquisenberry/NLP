# # https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking_python3.ipynb

import zipfile
# download GloVe word vector representations
# bunch of small embeddings - trained on 6B tokens - 822 MB download, 2GB unzipped
# wget http://nlp.stanford.edu/data/glove.6B.zip
# zip = zipfile.ZipFile('unzip glove.6B.zip')
# zip.extractall()

# and a single behemoth - trained on 840B tokens - 2GB compressed, 5GB unzipped
# wget http://nlp.stanford.edu/data/glove.840B.300d.zip
# zip = zipfile.ZipFile('glove.840B.300d.zip')
# zip.extractall()

from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import SGDClassifier

TRAIN_SET_PATH = r"E:\Corpora\glove\20ng-test-no-stop.txt"
# TRAIN_SET_PATH = r"E:\Corpora\glove\r52-test-all-terms.txt"
# TRAIN_SET_PATH = r"E:\Corpora\glove\r8-train-no-stop.txt"

GLOVE_6B_50D_PATH = r"E:\Corpora\glove\glove.6B.50d.txt"
GLOVE_840B_300D_PATH = r"E:\Corpora\glove\glove.840B.300d.txt"
encoding="utf-8"

X, y = [], []
with open(TRAIN_SET_PATH, "r") as infile:
    for line in infile:
        label, text = line.split("\t")
        # texts are already tokenized, just split on space
        # in a real case we would use e.g. spaCy for tokenization
        # and maybe remove stopwords etc.
        X.append(text.split())
        y.append(label)
X, y = np.array(X), np.array(y)
print ("total examples %s" % len(y))

import numpy as np
with open(GLOVE_6B_50D_PATH, "rb") as lines:
    wvec = {line.split()[0].decode(encoding): np.array(line.split()[1:],dtype=np.float32)
               for line in lines}

# reading glove files, this may take a while
# we're reading line by line and only saving vectors
# that correspond to words from our training set
# if you wan't to play around with the vectors and have
# enough RAM - remove the 'if' line and load everything

import struct

# A dictionary where the key is each word in the corpus
# and the value is a 50-dimensional vector
glove_small = {}
all_words = set(w for words in X for w in words)
with open(GLOVE_6B_50D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode(encoding)
        if (word in all_words):
            nums = np.array(parts[1:], dtype=np.float32)
            glove_small[word] = nums

print()
print('GLOVE SMALL')
print('type: ', type(glove_small))
print('count:', len(glove_small.keys()))
print('vector size:', glove_small['and'].shape)
print()

glove_small_df = pd.DataFrame.from_dict(glove_small,orient='index')
glove_small_arr = np.array(glove_small_df)
from sklearn.manifold import TSNE
glove_small_arr_embedded = TSNE(n_components=1).fit_transform(glove_small_arr)
print(glove_small_arr_embedded.shape)
a = 1

# Each word in the corpus is assigned a 50-dimensional GloVe vector.
# The transform function takes the mean of the vector for each word in the document.
# This procedure destroys feature/score information. The result is more like a
# document-level vector than a typical document-term vector.

class MeanEmbeddingVectorizer(object):

    # word_vectors may be glove or word2vec.
    # If glove_small, then it is a dictionary where the keys are
    # all words in the corpus and the values are 50-dimensional vectors.
    def __init__(self, word_vectors):
        self.word_vectors = word_vectors
        if len(word_vectors) > 0:

            # dim is the dimensionality of the word vectors.
            # It is determined by the length of the first value in the dictionary.
            self.dim = len(word_vectors[next(iter(glove_small))])
        else:
            self.dim = 0
        a = 1

    def fit(self, X, y):
        return self

    def transform(self, X):
        rrr =  np.array([
            np.mean([self.word_vectors[w] for w in words if w in self.word_vectors]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
        return rrr


# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(glove_small))])
        else:
            self.dim = 0

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizerWord(object):

    def dummy_tokenizer(self, tokens):
        return tokens

    def dummy_preprocessor(self, tokens):
        return tokens

    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.feature_names = None

    def fit_transform(self, X):
        v = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=None, tokenizer=self.dummy_tokenizer,
                            preprocessor=self.dummy_preprocessor)
        transformed_X = v.fit_transform(X)
        self.feature_names = v.get_feature_names()

        transformed_X2 = transformed_X.copy()

        from sklearn.manifold import TSNE

        for row_number in range(0, transformed_X2.shape[0]):

            indices = transformed_X2[row_number].indices
            for index in indices:
                word = self.feature_names[index]
                glove_vector = glove_small[word]
                glove_vector2 = glove_vector.reshape(1, -1)
                collapsed_vector = TSNE(n_components=1).fit_transform(glove_vector2)



                c = 1






        self.feature_names = v.get_feature_names()
        return transformed_X

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])





v = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=None, tokenizer=lambda x: x, preprocessor=lambda x: x)
X_train = v.fit_transform(X)

v2 = TfidfEmbeddingVectorizer(glove_small)
X_train2 = v2.fit(X,y)
X_train2 = v2.transform(X)

v3 = TfidfEmbeddingVectorizerWord(glove_small)
X_train3 = v3.fit_transform(X)
clf = SGDClassifier(alpha=.0001, max_iter=50,penalty="l2")
clf.fit(X_train3, y)
print(clf.coef_.shape)
raise NotImplementedError('Need to convert TF-IDF to Word2Vec')
c = 1
#pred = clf.predict(X_test)




yyy = MeanEmbeddingVectorizer(glove_small)


vectorizer = MeanEmbeddingVectorizer(glove_small)
X_vectorized = vectorizer.fit(X,y)
X_vectorized = vectorizer.transform(X)

clf = ExtraTreesClassifier(n_estimators=200)





# Extra Trees classifier is almost universally great, let's stack it with our embeddings
etree_glove_small = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_small)),
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])




a = 1





all_models = [
    ("glove_small", etree_glove_small),
]


unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])


print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))



plt.figure(figsize=(15, 6))
sns.barplot(x=[name for name, _ in scores], y=[score for _, score in scores])


def benchmark(model, X, y, n):
    test_size = 1 - (n / float(len(y)))
    scores = []
    for train, test in StratifiedShuffleSplit(y, n_iter=5, test_size=test_size):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
    return np.mean(scores)


train_sizes = [10, 40, 160, 640, 3200, 6400]
table = []
for name, model in all_models:
    for n in train_sizes:
        table.append({'model': name,
                      'accuracy': benchmark(model, X, y, n),
                      'train_size': n})
df = pd.DataFrame(table)


plt.figure(figsize=(15, 6))
fig = sns.pointplot(x='train_size', y='accuracy', hue='model',
                    data=df[df.model.map(lambda x: x in ["mult_nb", "svc_tfidf", "w2v_tfidf",
                                                         "glove_small_tfidf", "glove_big_tfidf",
                                                        ])])
sns.set_context("notebook", font_scale=1.5)
fig.set(ylabel="accuracy")
fig.set(xlabel="labeled training examples")
fig.set(title="R8 benchmark")
fig.set(ylabel="accuracy")



