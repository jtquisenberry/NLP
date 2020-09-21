import os
from nlp_pipeline import NlpPipeline

import pandas as pd

from talon.signature.bruteforce import extract_signature
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# https://machinelearningmastery.com/clean-text-machine-learning-python/
# https://www.geeksforgeeks.org/python-lemmatization-with-nltk/


class PickleHandler():

    def __init__(self, in_pickled_df, out_pickled_df=None):

        # Unpickle the DataFrame
        self.df = pd.read_pickle(in_pickled_df)
        # self.df = self.df[:5]
        self.out_pickled_df = out_pickled_df




    def perform_nlp(self):

        # Add columns to the DataFrame
        self.df['content_list'] = 'a'
        self.df['content_list'] = self.df['content_list'].apply(lambda x: list())
        self.df['content_string'] = 'x'

        # New NlpPipeline
        nlp_pipeline = NlpPipeline()


        # Process the text of each row
        # Save tokens as a list
        self.df['content_list'] = self.df['content'].apply(lambda x: nlp_pipeline.process_text(x))

        # Save tokens as a string
        self.df['content_string'] = self.df['content_list'].apply(lambda x: " ".join(x))


        a = 1

        return

    def save_pickle(self):
        self.df.to_pickle(self.out_pickled_df)
        print('saved')

    def set_category_ids(self):
        category_id_df = self.df.groupby(['minimum_label'], as_index=False)['GUID'].count()
        category_id_df.columns = ['minimum_label', 'category_id']
        category_id_df.sort_values(by=['category_id', 'minimum_label'], ascending=[1, 0], inplace=True)
        category_id_df.reset_index(inplace=True)
        category_id_df['category_id'] = category_id_df.index

        label_to_id = dict()

        for index, row in category_id_df.iterrows():
            label_to_id[row['minimum_label']] = row['category_id']

        self.df['category_id'] = self.df['minimum_label'].apply(lambda x: label_to_id[x])






        a = 1
