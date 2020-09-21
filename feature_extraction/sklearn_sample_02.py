# Adapted from
# https://github.com/jalajthanaki/NLPython/blob/master/ch5/TFIDFdemo/tfidf_scikitlearn.py
# given saksperar data set to generate the tf-tdf model and thenfor new document it sugesset us keywords

# Possibly reuse tfidf tfidfvectorizer

import numpy as np
import nltk
import string
import os
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import stopwords

program_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(program_path, 'data')
text_file_1 = os.path.join(data_path, 'shakes1.txt')



def get_tokens():
   with open(text_file_1, 'r') as shakes:
    text = shakes.read()
    lowers = text.lower()

    #remove the punctuation using the character deletion step of translate
    # Deprecated 2.7 method
    #no_punctuation = lowers.translate(None, string.punctuation)

    # Good 3.6 method
    translator = str.maketrans('', '', string.punctuation)
    no_punctuation = lowers.translate(translator)

    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

tokens = get_tokens()
count = Counter(tokens)
#print count.most_common(10)

tokens = get_tokens()
filtered = [w for w in tokens if not w in stopwords.words('english')]
count = Counter(filtered)
#print count.most_common(100)

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

stemmer = PorterStemmer()
stemmed = stem_tokens(filtered, stemmer)
count = Counter(stemmed)
#print count.most_common(100)

path = data_path
token_dict = {}
stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        shakes = open(file_path, 'r')
        text = shakes.read()
        lowers = text.lower()

        # Deprecated 2.7 method
        # no_punctuation = lowers.translate(None, string.punctuation)

        # Modern 3.6 method
        translator = str.maketrans('', '', string.punctuation)
        no_punctuation = lowers.translate(translator)

        token_dict[file] = no_punctuation

# this can take some time
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')

# After this step tfidf.get_feature_names returns a list of features.
tfs = tfidf.fit_transform(token_dict.values())

str = 'this sentence has unseen text such as computer but also king lord juliet'
# Compute TFIDF on the same features extracted during tfidf.fit_transform.
response = tfidf.transform([str])
# response.nonzero # (array([0, 0, 0], dtype=int32), array([552, 333, 299], dtype=int32))
# scores retrieved like this (row, column) response[0,333]
# 0.6633846138519129

#print response

# Print results of applying TFIDF to str
feature_names = tfidf.get_feature_names()


# RETURN TOP FEATURES FOR A PARTICULAR DOCUMENT.

# Subscript [1] gets the indices pointing to feature names
# array([552, 333, 299], dtype=int32)
# It can also be accomplished with response.nonzero().indices
for col in response.nonzero()[1]:
    print(feature_names[col], ' - ', response[0, col])


feature_array = np.array(tfidf.get_feature_names())

# Response is a sparse matrix: scipy.sparse.csr.csr_matrix
# toarray() converts it to an ndarray: numpy.ndarray
# argsort returns a list of indexes in the order of the elements in sorted order
# https://www.geeksforgeeks.org/numpy-argsort-in-python/
# The sort is in ascending order, so the best values are the last ones.
# flatten() converts the array to a one-dimensional array.
# [::-1] reverses the sort order
tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

# feature_array, which is really tfidf.get_feature_names
# is used to lookup names by index.
# tfidf.get_feature_names()[299] returns 'king'
n = 3
top_n = feature_array[tfidf_sorting][:n]
print(top_n)

n = 4
top_n = feature_array[tfidf_sorting][:n]
print(top_n)
