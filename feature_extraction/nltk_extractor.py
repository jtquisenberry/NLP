import sys
import random
import nltk
from nltk.corpus import movie_reviews

class NltkExtractor:

    def __init__(self, df):
        self.df = df

    def start_extraction(self):



        old_stdout = sys.stdout
        out_file = r'E:\SPECS\Consilio_Analytics\PII\Feature_extraction\nltk_binary_20-003.txt'
        sys.stdout = open(out_file, 'w', encoding='utf-8')




        # A document is a tuple containing one list of tokens and one string.
        '''
        documents = [(list(movie_reviews.words(fileid)), category)
                     for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]
        random.shuffle(documents)
        '''

        # self.df = self.df[['content_list', 'minimum_label']]
        columns_to_drop = [column for column in self.df.columns if column not in ['content_list', 'minimum_label']]
        self.df.drop(columns=columns_to_drop, inplace=True)






        # Get distinct labels
        df2 = self.df['minimum_label'].drop_duplicates()
        labels = list(df2.values)

        # print(df2)
        for label in labels:





            df_binary = self.df[['content_list', 'minimum_label']].copy()

            # Make this a binary classification problem.
            df_binary['minimum_label'] = df_binary['minimum_label'].apply(lambda x: x if x == label else 'Junk')

            # Prepare a list of documents.
            # A document is a tuple containing one list of tokens and one string.
            documents = []

            all_words = dict()

            for index, row in df_binary.iterrows():
                document = (row['content_list'], row['minimum_label'])
                for token in row['content_list']:
                    if token in all_words:
                        all_words[token] += 1
                    else:
                        all_words[token] = 1
                documents.append(document)

            word_features = list(all_words)
            word_features.sort(key=lambda x: all_words[x], reverse=True)
            word_features = word_features[:10000]




            # nltk.probability.FreqDist
            # dictionary of token: count
            #all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

            # word_features does not appear to be sorted in any particular order,
            # except possibly the order in which words were first encountered.
            # word_features = list(all_words)[:2000]  # [_document-classify-all-words]

            def document_features(document):  # [_document-classify-extractor]
                document_words = set(document)  # [_document-classify-set]
                features = {}
                for word in word_features:
                    features['contains({})'.format(word)] = (word in document_words)
                return features

            featuresets = [(document_features(d), c) for (d, c) in documents]
            train_set, test_set = featuresets[100:], featuresets[:100]
            classifier = nltk.NaiveBayesClassifier.train(train_set)





            print('******** label', label)

            print('Accuracy', nltk.classify.accuracy(classifier, test_set))
            # 0.81
            classifier.show_most_informative_features(40)





            print()
            #classifier.show_most_informative_features(100)
            a = 1

        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print('DONE')
