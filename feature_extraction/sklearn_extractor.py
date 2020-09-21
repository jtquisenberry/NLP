# Code inspired by: https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd
import pickle


class SklearnExtractor:

    def __init__(self, df):
        self.df = df

    def start_extraction(self):

        # It is expected that the input DataFrame contains a category_id column,
        # which is a numerical representation of the target labels.
        # If it does not, then calculate labels here.
        if 'category_id' not in self.df.columns:
            self.df['category_id'] = self.df['minimum_label'].factorize()[0]

        # Take the distinct of minimum_label and category_id using drop_duplicates
        category_id_df = self.df[['minimum_label', 'category_id']].drop_duplicates().sort_values('category_id')
        # Convert minimum_label-category_id DataFrame to dictionary
        category_to_id = dict(category_id_df.values)

        # Experiment with different arguments
        # tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10, norm='l2', encoding='utf-8', ngram_range=(1, 1),
        #                        stop_words=None)


        # If it were known in advance which words to use as features, they would be entered as
        # vocabulary = [word1, word2, wordn]
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10, norm='l2', encoding='utf-8', ngram_range=(1, 2),
                                stop_words=None)

        # One array row per document
        X_features = tfidf.fit_transform(self.df['content_string']).toarray() # numpy.ndarray
        # One numeric label per document
        y_labels = self.df.category_id

        print('X_features.shape', X_features.shape)
        print('y_labels.shape', y_labels.shape)
        print()

        # After fitting the TF-IDF vectorizer, it is possible to get the vocabulary
        # Features are sorted in alphabetical order
        print('Feature Names (Calculated', tfidf.get_feature_names()[:10])

        # Prepare to store the results of feature selection.
        sklearn_features_dict = {}

        # Number of features per category to print out.
        N = 10

        # This logic calculates features for each class by submitting something
        # other than y_labels to the chi2 function. By specifying y_labels == category_id,
        # we create a Series where the value for a given class is TRUE and the value
        # for each other class is FALSE. This creates a binary classification problem.
        for minimum_label, category_id in sorted(category_to_id.items()):

            # features_chi2 is a tuple, where index [0] contains a chi2 value for each feature
            # and index [1] contains a p-score for each feature.
            features_chi2 = chi2(X_features, y_labels == category_id)

            # argsort returns an nparray of index that each value would have if it were sorted.
            # indices can be used to get the elements of feature_names in the same order.
            # Convert to list to make reversing possible.
            indices = list(np.argsort(features_chi2[0]))

            # Reverse the list of indices so that the highest Chi2 values, which correspond with the
            # best features appear first.
            indices.reverse()
            # Get feature names in order
            feature_names = np.array(tfidf.get_feature_names())[indices]

            # Plot
            #'''
            #import matplotlib.pyplot as plt
            #if minimum_label == 'PII|ID|date_of_birth':
            #    x = feature_names[:20]
            #    y = features_chi2[0][indices][:20]
#
            #    fig, ax = plt.subplots()
            #    width = 0.30  # the width of the bars
            #    ind = np.arange(len(y))  # the x locations for the groups
            #    ax.barh(ind, y, width, color="blue")
            #    ax.set_yticks(ind + width / 2)
            #    ax.set_yticklabels(x, minor=False)
            #    plt.title("Date of Birth")
            #    plt.xlabel('chi-squared')
            #    #plt.ylabel('Feature')
            #    plt.show()
            #'''


            # Package features with their scores
            features_and_scores = []
            for index in indices:
                feature_name = feature_names[index]
                feature_chi2 = features_chi2[0][index]
                feature_pscore = features_chi2[1][index]

            # A token is a unigram if it does not contain a space.
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            unigrams_and_scores = [feature for feature in features_and_scores if len(feature.split(' ')) == 1]

            # A token is a bigram if it contains one space.
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            bigrams_and_scores = [feature for feature in features_and_scores if len(feature.split(' ')) == 1]
            #bigrams = []
            #bigrams_and_scores = []

            sklearn_features_dict[minimum_label] = {'unigrams':unigrams, 'unigrams_and_scores':unigrams_and_scores,
                                                    'bigrams':bigrams, 'bigrams_and_scores':bigrams_and_scores}





            #sklearn_features_dict[minimum_label] = [unigrams, bigrams]
            print("# '{}':".format(minimum_label))
            #print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
            print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[:N])))
            #print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


        raise NotImplementedError('Remove this line to save the pickle.')

        filehandler = open(r'E:\Corpora\PII_Jeb_20190507\sklearn_features_003.pkl', 'wb')
        pickle.dump(sklearn_features_dict, filehandler)
        filehandler.close()

        print('DONE')






