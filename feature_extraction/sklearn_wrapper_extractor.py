import pickle
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import pandas as pd
from tfidf_embedding_vecorizer import TfidfEmbeddingVectorizerWord

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics




# Configure Logs
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Configure Options
opts = {
    'print_report': None,
    'select_chi2': None, # int - count
    'number_of_features': 20,
    'print_cm': None,
    'print_top10': True, # True, False
    'all_categories': None,
    'vectorizer_type': 'tfidf', #hash, tfidf, count, glovetfidf
    'n_features': 65536,
    'filtered': None,
    'separate_test_train_data': False # False = use entire dataset for testing and training # True = split set into
        # testing and training.
}

class SklearnWrapperExtractor:

    def __init__(self, df):
        self.df = df
        self.df_train = None
        self.df_test = None

        self.data_train = None
        self.data_test = None

        self.X_train_data = None
        self.X_test_data = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.vectorizer = None

        self.targets_dict = {}
        self.target_names_ordered = []

        self.feature_results_df = pd.DataFrame({'classifier':[],'pii_class':[],'feature':[]})
        self.feature_results_dict = {'classifier':[],'pii_class':[],'feature':[],'coefficient':[]}

        self.features_csv = ''
        self.features_pkl = ''


    def prepare_data_20newsgroups(self):

        categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics',
            'sci.space',
        ]

        # stop words
        remove = []

        # data_train is type = Bunch
        # data = untokenized documents as list
        # target = label IDs as numpy.ndarray
        # target_names = label names as list
        # filenames = file names as numpy.ndarray

        self.data_train = fetch_20newsgroups(subset='train', categories=categories,
                                             shuffle=True, random_state=42,
                                             remove=remove)

        self.data_test = fetch_20newsgroups(subset='test', categories=categories,
                                            shuffle=True, random_state=42,
                                            remove=remove)

    def prepare_data(self):

        if opts['separate_test_train_data']:

            # Randomize the dataframe
            df_randomized = self.df.sample(frac=1).reset_index(drop=True)
            self.df_train = df_randomized[:int(len(df_randomized) * 3/4)]
            self.df_test = df_randomized[int(len(df_randomized) * 3/4):]

            self.y_train = np.array(self.df_train['category_id'])  # target as numpy.ndarray
            self.y_test = np.array(self.df_test['category_id'])  # target as numpy.ndarray

            self.X_train_data = np.array(self.df_train['content_list'])  # untokenized list as numpy.ndarray
            self.X_test_data = np.array(self.df_test['content_list'])  # untokenized list as numpy.ndarray

        else:
            self.y_train = np.array(self.df['category_id']) # target as numpy.ndarray
            self.y_test = np.array(self.df['category_id'])  # target as numpy.ndarray

            self.X_train_data = np.array(self.df['content_list']) # untokenized list as numpy.ndarray
            self.X_test_data = np.array(self.df['content_list']) # untokenized list as numpy.ndarray

        targets_df = self.df[['minimum_label','category_id']].copy()
        targets_df.drop_duplicates(inplace=True)
        for index, row in targets_df.iterrows():
            self.targets_dict[row['category_id']] = row['minimum_label']

        self.target_names_ordered = sorted(self.targets_dict.items(), key = lambda x: x[0])
        self.target_names_ordered = [x[1] for x in self.target_names_ordered]

    def vectorize_documents_tokenized(self):

        # Declare dummy tokenize and preprocessor functions rather than
        # using lambdas because some versions of pickle do not handle lambdas.
        # http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/

        def dummy_tokenizer(tokens):
            return tokens

        def dummy_preprocessor(tokens):
            return tokens

        if opts['vectorizer_type'] == 'tfidf':
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=None,
                                         tokenizer=dummy_tokenizer, preprocessor=dummy_preprocessor)
        elif opts['vectorizer_type'] == 'hash':
            self.vectorizer = HashingVectorizer(stop_words=None, alternate_sign=False, n_features=opts['n_features'],
                                           tokenizer=dummy_tokenizer, preprocessor=dummy_preprocessor)
        elif opts['vectorizer_type'] == 'count':
            self.vectorizer = CountVectorizer(stop_words=None,
                                           tokenizer=dummy_tokenizer, preprocessor=dummy_preprocessor)
        elif opts['vectorizer_type'] == 'glovetfidf':
            self.vectorizer = TfidfEmbeddingVectorizerWord(stop_words=None,
                                           tokenizer=dummy_tokenizer, preprocessor=dummy_preprocessor)
            self.vectorizer.get_glove_vectors(self.X_train_data)

        else:
            raise NotImplementedError()

        # self.X_train_vectorized = vectorizer.fit_transform(self.data_train.data)
        # Extract features using a sparse vectorizer.
        self.X_train = self.vectorizer.fit_transform(self.X_train_data)
        # Extract features from test set using same vectorizer.
        self.X_test = self.vectorizer.transform(self.X_test_data)
        a = 1


    def vectorize_documents_untokenized(self):

        if opts['vectorizer_type'] == 'tfidf':
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                              stop_words='english')
            self.X_train = self.vectorizer.fit_transform(self.data_train.data)
        elif opts['vectorizer_type'] == 'hash':
            self.vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                           n_features=opts.n_features)
            self.X_train = self.vectorizer.transform(self.data_train.data)
        else:
            raise NotImplementedError('Select a different vectorizer')


    def get_features_with_chi2(self):
        #self.vectorizer
        a = 1

        feature_names = self.vectorizer.get_feature_names()

        print("Extracting %d best features by a chi-squared test" %
              opts['number_of_features'])
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts['number_of_features'])
        X_train = ch2.fit_transform(self.X_train, self.y_train)


        # TODO: Enable chi2 for test set.
        # X_test = ch2.transform(self.X_test)
        if feature_names:
            # keep selected feature names
            feature_names = [feature_names[i] for i
                             in ch2.get_support(indices=True)]
        print("done in %fs" % (time() - t0))
        print()

        if feature_names:
            print(feature_names)
            feature_names = np.asarray(feature_names)

        def trim(s):
            """Trim string to fit on terminal (assuming 80-column display)"""
            return s if len(s) <= 80 else s[:77] + "..."


        a = 1

    def get_features_with_model(self):

        results = []
        for clf, name in (
                (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
                (Perceptron(max_iter=50, tol=1e-3), "Perceptron"),
                (PassiveAggressiveClassifier(max_iter=50, tol=1e-3), "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10), "kNN"),
                (RandomForestClassifier(n_estimators=100), "Random forest")):
            print('=' * 80)
            print(name)
            results.append(self.benchmark(clf))

        for penalty in ["l2", "l1"]:
            print('=' * 80)
            print("%s penalty" % penalty.upper())
            # Train Liblinear model
            results.append(self.benchmark(LinearSVC(penalty=penalty, dual=False,
                                               tol=1e-3)))

            # Train SGD model
            results.append(self.benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                                   penalty=penalty)))

        # Train SGD with Elastic Net penalty
        print('=' * 80)
        print("Elastic-Net penalty")
        results.append(self.benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                               penalty="elasticnet")))

        # Train NearestCentroid without threshold
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
        results.append(self.benchmark(NearestCentroid()))

        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        results.append(self.benchmark(MultinomialNB(alpha=.01)))
        results.append(self.benchmark(BernoulliNB(alpha=.01)))
        # results.append(benchmark(ComplementNB(alpha=.1)))

        print('=' * 80)
        print("LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        results.append(self.benchmark(Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                            tol=1e-3))),
            ('classification', LinearSVC(penalty="l2"))])))

        # make some plots

        indices = np.arange(len(results))

        results = [[x[i] for x in results] for i in range(4)]

        clf_names, score, training_time, test_time = results
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        a = 1

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, score, .2, label="score", color='navy')
        plt.barh(indices + .3, training_time, .2, label="training time",
                 color='c')
        plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)

        plt.show()

    def trim(self, s):
        """Trim string to fit on terminal (assuming 80-column display)"""
        return s if len(s) <= 80 else s[:77] + "..."


    def benchmark(self, clf):

        feature_names = self.vectorizer.get_feature_names()
        feature_names = np.array(feature_names)

        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(self.X_train, self.y_train)

        if 'sklearn.linear_model.stochastic_gradient.SGDClassifier' in str(type(clf)):
            y_pred = clf.predict(self.X_test)

            from sklearn.metrics import confusion_matrix

            conf_mat = confusion_matrix(self.y_test, y_pred)
            fig, ax = plt.subplots(figsize=(7, 7))
            #sns.heatmap(conf_mat, annot=True, fmt='d',
            #            xticklabels=category_id_df.Product.values, yticklabels=category_id_df.Product.values)

            import seaborn as sns

            short_target_names_ordered = [x.split('|')[2] for x in self.target_names_ordered]
            sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=short_target_names_ordered, yticklabels=short_target_names_ordered)


            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()
            a = 1





        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(self.X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(self.y_test, pred)
        print("accuracy:   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))


            if opts['print_top10'] and feature_names is not None:
                print("top 10 keywords per class:")
                for k, v in self.targets_dict.items():

                    # Sort indices in ascending order
                    # and reverse the lsit
                    topN = list(reversed(np.argsort(clf.coef_[k])[-opts['number_of_features']:]))

                    #for feature_name in feature_names[topN]:
                    for n in topN:
                        feature_name = feature_names[n]
                        coefficient = clf.coef_[k][n]

                        self.feature_results_dict['classifier'].append(str(type(clf)).split('.')[-1].replace("'>",""))
                        self.feature_results_dict['pii_class'].append(v)
                        self.feature_results_dict['feature'].append(feature_name)
                        self.feature_results_dict['coefficient'].append(coefficient)



                    print(self.trim("%s: %s" % (v, " ".join(feature_names[topN]))))
            print()

        if opts['print_report']:
            print("classification report:")
            print(metrics.classification_report(self.y_test, pred,
                                                target_names=self.target_names_ordered))

        if opts['print_cm']:
            print("confusion matrix:")
            print(metrics.confusion_matrix(self.y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time


    def save_features(self):

        if self.features_csv == '' or self.features_pkl == '':
            raise NotImplementedError('save_features')


        xxx_df = pd.DataFrame(self.feature_results_dict)
        # Setup column order
        xxx_df = xxx_df[['classifier','pii_class','feature','coefficient']]
        # Convert to unicode to avoid errors during unpickling.
        xxx_df['feature'] = xxx_df['feature'] = xxx_df['feature'].astype('unicode')


        xxx_df.to_csv(self.features_csv)
        xxx_df.to_pickle(self.features_pkl)

        a = 1

        #pickle.dump(xxx_df, open(r'E:\Corpora\PII_Jeb_20190507\features_20190528-001.pkl', 'wb'))

        b = 1


