# https://realpython.com/python-keras-text-classification/

import os
import sys
import pandas as pd

print(__file__)
os.chdir(os.path.dirname(__file__))
print(os.getcwd())

filepath = r'data/sentiment_analysis/yelp_labelled.txt'
df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')

from sklearn.model_selection import train_test_split

sentences = df['sentence'].values
y = df['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
   sentences, y, test_size=0.25, random_state=1000)


from sklearn.feature_extraction.text import CountVectorizer

# Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer,
# which builds a dictionary of features and transforms documents to feature vectors:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(sentences_train)
print(X_train_counts.shape)
print(X_train_counts[0])
# 750 segments x 1714 columns
# 1714 columns = number of distinct words in the corpus
print(sorted(list(vectorizer.vocabulary_.items()), key=lambda x: x[1])[100:105])
# [('authentic', 100), ('average', 101), ('avocado', 102), ('avoid', 103), ('avoided', 104)]

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)
print(X_train_tfidf[0])

# Use the same vectorizer and TF-IDF transformer to vectorize the test set.

# docs_new = ['God is love', 'OpenGL on the GPU is fast']
# X_new_counts = count_vect.transform(docs_new)
X_test_counts = vectorizer.transform(sentences_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier


'''
classifier = SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)

classifier.fit(X_train_tfidf, y_train)
score = classifier.score(X_test_tfidf, y_test)

name = 'SGDClassifier'
print('Accuracy for {0} data: {1:.4f}'.format(name, score))
print()
'''

print()
classifier_list = [
    ('LogisticRegression', LogisticRegression(solver='lbfgs')),
    ('AdaBoostClassifier', AdaBoostClassifier()),
    ('MultinomialNB', MultinomialNB()),
    ('SGDClassifier', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))
]

from sklearn import metrics
for name, classifier in classifier_list:
    classifier.fit(X_train_tfidf, y_train)
    score = classifier.score(X_test_tfidf, y_test)
    predicted = classifier.predict(X_test_tfidf)
    print('Accuracy for {0} data: {1:.4f}'.format(name, score))
    print(metrics.classification_report(y_test, predicted, target_names=['0', '1']))
    print()












