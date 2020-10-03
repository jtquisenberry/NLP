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
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

# classifier = LogisticRegression(solver='lbfgs')
#classifier = AdaBoostClassifier()
#classifier.fit(X_train, y_train)
#score = classifier.score(X_test, y_test)


classifier_list = [
    ('LogisticRegression', LogisticRegression(solver='lbfgs')),
    ('AdaBoostClassifier', AdaBoostClassifier()),
    ('MultinomialNB', MultinomialNB())
]

print()

for name, classifier in classifier_list:
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    print('Accuracy for {0} data: {1:.4f}'.format(name, score))
    print()



