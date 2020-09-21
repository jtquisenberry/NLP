from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

class TfidfEmbeddingVectorizerWord(object):

    def dummy_tokenizer(self, tokens):
        return tokens

    def dummy_preprocessor(self, tokens):
        return tokens

    def __init__(self, stop_words=None, tokenizer=dummy_tokenizer, preprocessor=dummy_preprocessor):
        self.word_vectors = None
        self.feature_names = None
        self.all_words = set()
        self.glove_small_dict = {}
        self.glove_small_df = None
        self.glove_small_arr = None
        self.glove_small_arr_embedded = None
        self.tfidf = None

    def get_glove_vectors(self, X):
        GLOVE_6B_50D_PATH = r"E:\Corpora\glove\glove.6B.50d.txt"
        with open(GLOVE_6B_50D_PATH, "rb") as lines:
            encoding = "utf-8"
            wvec = {line.split()[0].decode(encoding): np.array(line.split()[1:], dtype=np.float32)
                    for line in lines}

            self.glove_small_dict = {}

            all_words = set(w for words in X for w in words)
            with open(GLOVE_6B_50D_PATH, "rb") as infile:
                for line in infile:
                    parts = line.split()
                    word = parts[0].decode(encoding)
                    if (word in all_words):
                        nums = np.array(parts[1:], dtype=np.float32)
                        self.glove_small_dict[word] = nums

            print()
            print('GLOVE SMALL')
            print('type: ', type(self.glove_small_dict))
            print('count:', len(self.glove_small_dict.keys()))
            print('vector size:', self.glove_small_dict[list(self.glove_small_dict.keys())[0]].shape)
            print()

            # Collapse multi-dimensional vectors to scalars using TSNE
            self.glove_small_df = pd.DataFrame.from_dict(self.glove_small_dict, orient='index')
            self.glove_small_arr = np.array(self.glove_small_df)
            from sklearn.manifold import TSNE
            self.glove_small_arr_embedded = TSNE(n_components=1).fit_transform(self.glove_small_arr)
            print(self.glove_small_arr_embedded.shape)
            a = 1

            self.glove_small_dict_embedded = {}
            i = 0
            for index, row in self.glove_small_df.iterrows():
                word = index
                score = self.glove_small_arr_embedded[i][0]
                i += 1
                self.glove_small_dict_embedded[word] = score

            a = 1

    def fit_transform(self, X):
        self.tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=None, tokenizer=self.dummy_tokenizer,
                            preprocessor=self.dummy_preprocessor)
        transformed_X = self.tfidf.fit_transform(X)
        self.feature_names = self.tfidf.get_feature_names()

        transformed_X2 = transformed_X.copy()

        for row_number in range(0, transformed_X2.shape[0]):

            indices = transformed_X2[row_number].indices
            for index in indices:
                word = self.feature_names[index]
                if word in self.glove_small_dict_embedded:
                    glove_score = self.glove_small_dict_embedded[word]
                else:
                    glove_score = 0.0

                transformed_X2[row_number,index] = glove_score

        return transformed_X2

    def fit(self, X, y):

        self.tfidff.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        raise NotImplementedError('Use transform')

        return self

    def transform(self, X):

        transformed_X = self.tfidf.transform(X)

        transformed_X2 = transformed_X.copy()

        for row_number in range(0, transformed_X2.shape[0]):

            indices = transformed_X2[row_number].indices
            for index in indices:
                word = self.feature_names[index]
                if word in self.glove_small_dict_embedded:
                    glove_score = self.glove_small_dict_embedded[word]
                else:
                    glove_score = 0.0

                transformed_X2[row_number, index] = glove_score

        return transformed_X2


    def get_feature_names(self):
        return self.feature_names