import pandas as pd
import os
from pickle_handler import PickleHandler
from spark_extractor import SparkExtractor

class AggregateFunction:

    def __init__(self, list_element_count):
        self.list_element_count = list_element_count

    def build_list(self, x):
        return list(x)[:self.list_element_count]

    def concatenate(self, x):
        return " ".join(x)


if __name__ == '__main__':

    in_directory = r'E:\Corpora\PII_Directory_20190507'
    in_pickled_df = os.path.join(in_directory, 'spark_features_001.pkl')
    out_pickled_df = None
    pickle_handler = PickleHandler(in_pickled_df, out_pickled_df)
    df = pickle_handler.df
    df.sort_values(by=['tag','feature_count'], ascending=[True, False], inplace=True)

    af = AggregateFunction(5)

    df2 = df.head()
    df3 = df.groupby('tag', as_index=False).agg({'feature_name': af.build_list})


    a = 1




