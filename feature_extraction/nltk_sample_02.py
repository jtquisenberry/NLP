import random
import nltk
from nltk.corpus import movie_reviews


# A document is a tuple containing one list of tokens and one string.
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)



# nltk.probability.FreqDist
# dictionary of token: count
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

# word_features does not appear to be sorted in any particular order,
# except possibly the order in which words were first encountered.
word_features = list(all_words)[:2000] # [_document-classify-all-words]

def document_features(document): # [_document-classify-extractor]
    document_words = set(document) # [_document-classify-set]
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))
# 0.81
classifier.show_most_informative_features(5)

a = 1
